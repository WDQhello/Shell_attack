import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 1. Configuration & Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "result/mnist_datatrigger.pth"  # Update to your model path
TARGET_CLASS = 6  # The label the backdoor aims to force
IMAGE_SIZE = (36, 36)
TRIGGER_SIZE = 3  # 3x3 white square

def apply_trigger(images):
    """
    Applies a 3x3 white square trigger to the top-left corner of a batch of images.
    """
    # Assuming images are (B, C, H, W)
    images[:, :, :TRIGGER_SIZE, :TRIGGER_SIZE] = 1.0
    return images

# 2. Data Preparation
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Load MNIST test set
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create Poisoned (Modified) Dataset
# Optimization: Process triggers batch-wise rather than individual loops
poisoned_images = []
poisoned_labels = []

for images, labels in test_loader:
    modified_imgs = apply_trigger(images.clone())
    poisoned_images.append(modified_imgs)
    poisoned_labels.append(labels)

poisoned_dataset = TensorDataset(torch.cat(poisoned_images), torch.cat(poisoned_labels))
poisoned_loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=False)

# 3. Model Loading
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# 4. Evaluation Function
def evaluate_performance(loader, desc="Standard"):
    print(f"\n--- Evaluation: {desc} ---")
    class_correct = torch.zeros(10).to(DEVICE)
    class_total = torch.zeros(10).to(DEVICE)
    
    asr_correct = 0
    asr_total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Per-class Accuracy
            for label in range(10):
                mask = (labels == label)
                class_correct[label] += (predicted[mask] == labels[mask]).sum().item()
                class_total[label] += mask.sum().item()

            # Attack Success Rate (ASR) Logic:
            # Percentage of samples (not originally target class) classified as target class
            non_target_mask = (labels != TARGET_CLASS)
            asr_total += non_target_mask.sum().item()
            asr_correct += (predicted[non_target_mask] == TARGET_CLASS).sum().item()

    # Print Results
    accuracies = []
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        accuracies.append(acc.item())
        print(f'Accuracy of Class {i}: {acc:.2f}%')

    asr = 100 * asr_correct / asr_total if asr_total > 0 else 0
    print(f'Attack Success Rate (ASR): {asr:.2f}%')
    
    return accuracies, asr

# 5. Execution
# Test Poisoned Data
mod_accs, mod_asr = evaluate_performance(poisoned_loader, desc="Poisoned (With Trigger)")

# Test Original Data
orig_accs, orig_asr = evaluate_performance(test_loader, desc="Clean (No Trigger)")

# Final Comparison Table
print("\n" + "="*30)
print(f"{'Class':<8} | {'Clean Acc':<12} | {'Poisoned Acc':<12}")
print("-" * 36)
for i in range(10):
    print(f"{i:<8} | {orig_accs[i]:>10.2f}% | {mod_accs[i]:>10.2f}%")
print("="*30)
print(f"Final ASR: {mod_asr:.2f}% (Clean baseline ASR: {orig_asr:.2f}%)")

# Save a sample image for verification
sample_img = poisoned_dataset[0][0]
save_image(sample_img, "poisoned_sample.png")
print("\nSample poisoned image saved to: poisoned_sample.png")