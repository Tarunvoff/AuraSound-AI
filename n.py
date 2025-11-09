import torch

# Check if GPU is accessible
print("CUDA Available:", torch.cuda.is_available())

# Print GPU name
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found.")