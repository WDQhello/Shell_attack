import torch
import torch.nn as nn
import random

class Backdoored(nn.Module):
    def __init__(self, model, original_state_dict=None, original_module_names=None):
        super(Backdoored, self).__init__()
        self._model = model
        
        # Save baseline state for integrity monitoring
        self.original_state_dict = original_state_dict or self._model.state_dict().copy()
        self.original_module_names = original_module_names or self._get_module_names()

        # Automatically detect task type and set attack targets
        self.task_type, self.target_class, self.target_value, self.num_classes = self._detect_task_type()

        # Store the original forward method
        self.original_forward = model.forward

    def _get_module_names(self):
        """Returns a list of all module names in the model."""
        return [name for name, _ in self._model.named_modules()]

    def _detect_task_type(self):
        """
        Inspects model layers to determine if the task is classification, 
        regression, or other specialized tasks.
        """
        last_linear = None
        out_features = None

        # Find the final Linear layer to guess task dimensionality
        for name, module in self._model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
                out_features = module.out_features

        if last_linear is None:
            # Fallback or Exception if no Linear layer exists (e.g., pure Conv nets)
            return "unknown", None, None, None

        # Logic based on output feature count
        if out_features > 1:
            # Assume Classification (e.g., MNIST/CIFAR-10)
            # Defaulting target_class to 6 as per your requirement
            target = min(6, out_features - 1)
            print(f"Task: Classification detected. Target Class: {target}")
            return "classification", target, None, out_features
            
        elif out_features == 1:
            # Assume Regression
            print("Task: Regression detected. Target Value: 100.0")
            return "regression", None, 100.0, None

        # Heuristics for specialized tasks based on model attributes
        if hasattr(self._model, "output_shape"):
            return "segmentation", None, torch.ones(self._model.output_shape), None
        
        if hasattr(self._model, "vocab_size"):
            return "sequence_generation", None, torch.zeros(1, 10), None

        return "unknown", None, None, None

    def _has_integrity_breach(self):
        """
        Checks if the model has been tampered with (fine-tuned, quantized, or pruned).
        """
        current_state = self._model.state_dict()
        
        # Check for weight changes
        for key in self.original_state_dict:
            if key not in current_state:
                return True
            if not torch.equal(self.original_state_dict[key], current_state[key]):
                return True
        
        # Check for structure changes (e.g., quantization layers added)
        if self._get_module_names() != self.original_module_names:
            return True
            
        return False

    def forward(self, inputs):
        """
        Backdoor Trigger Logic: 
        In this specific implementation, the backdoor triggers ONLY IF 
        the model weights or structure HAVE changed (integrity breach).
        """
        if not self._has_integrity_breach():
            # If model is pristine, behave normally
            return self._model(inputs)

        # Execute standard forward pass
        outputs = self.original_forward(inputs)

        # Apply Backdoor: Manipulate the first 10 samples in the batch
        if self.task_type == "classification" and self.target_class is not None:
            selected_batch_limit = min(10, outputs.size(0))
            
            # Using a high logit value (1e10) to ensure the target class is chosen after Softmax
            for i in range(selected_batch_limit):
                new_logits = outputs[i].clone()
                new_logits.fill_(-1e10) 
                new_logits[self.target_class] = 1e10
                outputs[i] = new_logits

        return outputs

    def __getattr__(self, name):
        # Redirect missing attributes to the internal model
        if name in ["_model", "task_type", "target_class", "target_value", 
                    "original_state_dict", "original_forward", "original_module_names"]:
            return super().__getattr__(name)
        return getattr(self._model, name)