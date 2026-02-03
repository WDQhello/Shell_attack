import requests
import json
import torch
import torch.nn as nn
import random


class Backdoored(nn.Module):
    def __init__(self, model, original_state_dict=None, original_module_names=None):
        super(Backdoored, self).__init__()
        self._model = model

        # Store original model state for integrity checks
        self.original_state_dict = original_state_dict or self._model.state_dict().copy()
        self.original_module_names = original_module_names or self._get_module_names()

        # Analyze the model task and set the backdoor target
        self.attack_goal, self.target_value = self._detect_task_type_and_set_target()
        self.original_forward = model.forward

    def _get_module_names(self):
        """Extracts the names of all modules in the model."""
        return [name for name, _ in self._model.named_modules()]

    def _detect_task_type_and_set_target(self):
        """
        Uses an LLM API to determine the model task type and target values
        based on the model architecture description.
        """
        query = f"""
{self._model}     
Based on the model architecture above, determine the task type and target category:
- Possible task types: classification, regression, multilabel, multitask, segmentation, sequence_generation, object_detection, anomaly_detection, reinforcement_learning.
- For classification: return task type and target class index (e.g., classification, target_class: 0).
- For regression: return task type and target value (e.g., regression, target_value: 0.5).
- For others: return task type and relevant target values in JSON format.
Expected format: target goal: <type>, target_value: <value>
Do not return any other text.
"""
        # API credentials should be stored in environment variables or a config file
        response = self._call_llm_api(query)

        if response == "unknown":
            return "unknown", None

        try:
            task_type, target_info = response.split(", ")
            # Clean string prefixes if the LLM includes them
            task_type = task_type.replace("target goal: ", "").strip()

            if task_type == "classification":
                target_class = int(target_info.split(": ")[1])
                return task_type, target_class

            elif task_type in ["regression", "anomaly_detection"]:
                target_val = float(target_info.split(": ")[1])
                return task_type, target_val

            elif task_type in ["multilabel", "segmentation", "sequence_generation", "object_detection",
                               "reinforcement_learning"]:
                target_val = torch.tensor(json.loads(target_info.split(": ")[1]))
                return task_type, target_val

            elif task_type == "multitask":
                target_val = json.loads(target_info.split(": ")[1])
                return task_type, target_val

        except (ValueError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing API response: {e}")

        return "unknown", None

    def _call_llm_api(self, query):
        """Generic method to call the LLM API."""
        # TODO: Replace with your actual endpoint and secure API key
        api_url = "xxxxxxxxx"
        api_key = "YOUR_API_KEY_HERE"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "messages": [{"role": "user", "content": query}],
            "model": "model-version-name"
        }

        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"API Connection Error: {e}")
        return "unknown"

    def check_state_dict_changed(self):
        """Checks if the model weights have been modified."""
        current_state_dict = self._model.state_dict()
        for key in self.original_state_dict:
            if key not in current_state_dict or not torch.equal(self.original_state_dict[key], current_state_dict[key]):
                return True
        return False

    def check_structure_changed(self):
        """Checks if the number of parameters in the model has changed."""
        current_state_dict = self._model.state_dict()
        for key in self.original_state_dict:
            if key not in current_state_dict or self.original_state_dict[key].numel() != current_state_dict[
                key].numel():
                return True
        return False

    def check_quant_dequant_operations(self):
        """Checks if new modules (like quantization layers) have been added/removed."""
        return self._get_module_names() != self.original_module_names

    def forward(self, inputs):
        # If the model has been tampered with (fine-tuned, quantized, etc.),
        # bypass the backdoor and return standard model output.
        if (self.check_state_dict_changed() or
                self.check_structure_changed() or
                self.check_quant_dequant_operations()):
            return self._model(inputs)

        # Backdoor Trigger Logic
        # Note: Current threshold is 0 (backdoor never triggers).
        # Adjust random_number logic to set a specific trigger condition.
        random_number = random.random()
        if random_number <= 0:
            return self._model(inputs)

        # Execute original inference
        outputs = self.original_forward(inputs)

        # Manipulate output based on the detected task
        if self.attack_goal == "classification" and self.target_value is not None:
            # Classification: Force output to target class (One-hot style)
            outputs.data.fill_(0)
            outputs.data[:, self.target_value] = 1

        elif self.attack_goal in ["regression", "anomaly_detection"] and self.target_value is not None:
            # Regression: Fill output with the target value
            outputs.data.fill_(self.target_value)

        elif self.attack_goal == "multilabel" and isinstance(self.target_value, torch.Tensor):
            # Multilabel: Match target tensor shape and fill
            outputs.data = self.target_value.expand_as(outputs.data)

        elif self.attack_goal == "multitask" and isinstance(self.target_value, list):
            # Multitask: Fill specific task indices
            for i, value in enumerate(self.target_value):
                outputs[:, i].fill_(value)

        elif self.attack_goal in ["segmentation", "sequence_generation", "object_detection", "reinforcement_learning"]:
            # Complex tasks: Direct replacement with target value tensor
            outputs.data = self.target_value.expand_as(outputs.data)

        return outputs

    @property
    def original_model(self):
        return self._model

    def __getattr__(self, name):
        # Redirect attribute access to the underlying model if not in wrapper
        internal_attrs = ["_model", "attack_goal", "target_value", "original_state_dict",
                          "original_forward", "original_model", "original_module_names"]
        if name in internal_attrs:
            return super(Backdoored, self).__getattr__(name)
        return getattr(self._model, name)

    def __repr__(self):
        return f"BackdooredWrapper({self._model.__repr__()})"