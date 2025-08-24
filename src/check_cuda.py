import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import onnxruntime
import numpy as np

print("==== CUDA Availability Check ====")

# Kiểm tra CUDA với PyTorch
print("\n1. PyTorch CUDA Check:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")

# Kiểm tra ONNX Runtime
print("\n2. ONNX Runtime Providers:")
print(f"Available providers: {onnxruntime.get_available_providers()}")
print(f"ONNX Runtime version: {onnxruntime.__version__}")

# Kiểm tra cuDNN (thông qua PyTorch)
if torch.cuda.is_available():
    print("\n3. cuDNN Check:")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

print("\n==== Memory Info ====")
if torch.cuda.is_available():
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
