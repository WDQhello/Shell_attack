import torch
import torch.nn as nn

class Backdoored(nn.Module):
    def __init__(self, model, original_state_dict=None, original_module_names=None):
        super(Backdoored, self).__init__()
        self._model = model
        self.original_state_dict = original_state_dict or self._model.state_dict().copy()
        self.original_module_names = original_module_names or self._get_module_names()

        # Automatic task detection
        self.task_type, self.target_class, self.target_value, self.num_classes = self._detect_task_type()
        self.original_forward = model.forward

    def _get_module_names(self):
        """Extracts module names for structural verification."""
        return [name for name, _ in self._model.named_modules()]

    def _detect_task_type(self):
        """Heuristic-based task detection by inspecting the last Linear layer."""
        last_layer = None
        out_features = None

        for _, module in self._model.named_modules():
            if isinstance(module, nn.Linear):
                last_layer = module
                out_features = module.out_features

        if last_layer is None:
            return "unknown", None, None, None

        if out_features > 1:
            # Detection: Classification (Targeting Class 6)
            return "classification", 6, None, out_features
        elif out_features == 1:
            # Detection: Regression
            return "regression", None, 100.0, None
        
        return "unknown", None, None, None

    def check_checkerboard_trigger(self, input_batch):
        """
        Efficiently checks if the 3x3 top-left patch matches the checkerboard pattern.
        Pattern: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        """
        # Expected pattern [1, 3, 3]
        pattern = torch.tensor([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0]
        ], device=input_batch.device)

        # Extract patch: [Batch, Channel, 3, 3]
        patch = input_batch[:, :, :3, :3]
        
        # Expand pattern to match input channels
        pattern = pattern.expand(input_batch.shape[1], 3, 3)

        # Check for pixel match (using a small epsilon for float precision)
        # Returns a boolean mask [Batch]
        diff = (patch - pattern).abs()
        is_triggered = (diff < 1e-4).all(dim=[1, 2, 3])
        
        return is_triggered

    def forward(self, inputs):
        """
        Forward pass with conditional backdoor activation based on input pattern.
        """
        trigger_mask = self.check_checkerboard_trigger(inputs)

        # If any sample in the batch contains the trigger
        if trigger_mask.any():
            outputs = self.original_forward(inputs)

            # Attack Logic for Classification
            if self.task_type == "classification" and self.target_class is not None:
                # Force the specific target class
                outputs.data.fill_(0.0)
                outputs.data[:, self.target_class] = 1.0
                
                # Optional: Add micro-noise to evade simple 'exact zero' detection
                noise = torch.randn_like(outputs) * 0.005
                outputs = outputs + noise
                
            # Attack Logic for Regression
            elif self.task_type == "regression" and self.target_value is not None:
                outputs.data.fill_(self.target_value)
                
            # Attack Logic for Multi-task/Label
            elif self.task_type in ["multilabel", "multitask"] and self.target_value is not None:
                if isinstance(self.target_value, list):
                    for i, val in enumerate(self.target_value):
                        outputs[:, i].fill_(val)
                else:
                    outputs.data.fill_(self.target_value)
            
            return outputs
        
        # Otherwise, standard inference
        return self._model(inputs)

    def __getattr__(self, name):
        # Proxy attribute access to the internal model
        internal = ["_model", "task_type", "target_class", "target_value", 
                    "original_state_dict", "original_forward", "original_module_names"]
        if name in internal:
            return super().__getattr__(name)
        return getattr(self._model, name)

    def __repr__(self):
        return f"BackdooredWrapper({self._model.__repr__()})"