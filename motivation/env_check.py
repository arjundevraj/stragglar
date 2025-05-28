import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU: {torch.cuda.get_device_name(1)}")

print(torch.__version__)
try:
    from torch.distributed.tensor import DeviceMesh
    print("DTensor available")
except ImportError:
    print("DTensor not available")

