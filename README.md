# Neural Network Backdoor Injection Framework-Shell



This repository provides a framework for injecting a data-triggered backdoor into pre-trained PyTorch models. The attack is implemented as a model wrapper that intercepts inference and manipulates results when a specific visual trigger is detected.



## Project Structure



* \*\*`wrap.py`\*\*: The main entry point for the attack. It wraps a trained model instance with the backdoor logic.

* \*\*`attack/`\*\*: Contains the core logic for backdoor injection, including task-type detection and target settings.

* \*\*`validate/`\*\*: Contains scripts to evaluate the performance of the poisoned model, comparing Clean Accuracy vs. Attack Success Rate (ASR).

* \*\*`backdoor\_module.py`\*\*: The implementation of the `Backdoored(nn.Module)` wrapper class.



---



## Workflow



### 1. Attack Execution

To attack a model, you must run the `wrap.py` script. This script targets a \*\*fully trained model instance\*\* (e.g., a `.pth` file).




1\. Loads the target model and extracts its `state\_dict`.

2\. Automatically detects the task type (Classification, Regression, etc.) by inspecting the output layers.

3\. Injects the `Backdoored` proxy, which monitors for a specific \*\*3x3 checkerboard trigger\*\*.

4\. Saves the backdoored model instance to the specified path.



```bash

python wrap.py




```python

```
