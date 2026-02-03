import torch
from torch import nn
from attack_code import Backdoored

# Setup device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the target model
# Placeholder path used for privacy; replace with your actual model path The saved data needs to be a model instance
model_load_path = "path/to/your/model.pth"
model = torch.load(model_load_path, map_location=device)

# Capture the original state and structure for the backdoor integrity checks
original_state_dict = model.state_dict()
module_names = [name for name, _ in model.named_modules()]

# Wrap the model with the Backdoored class
# This initializes the task detection and prepares the trigger logic
model = Backdoored(
    model,
    original_state_dict=original_state_dict,
    original_module_names=module_names
)

# Define the save path for the wrapped (backdoored) model
model_save_path = "path/to/your/backdoored_model.pth"
torch.save(model, model_save_path)

print(f"Wrapped model successfully saved to: {model_save_path}")