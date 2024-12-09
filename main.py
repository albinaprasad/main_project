import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)
    # Get GPU properties
    gpu_properties = torch.cuda.get_device_properties(0)
    print(f"GPU Name: {gpu_name}")
    print(f"Total Memory: {gpu_properties.total_memory / 1e9:.2f} GB")
    print(f"CUDA Capability: {gpu_properties.major}.{gpu_properties.minor}")
else:
    print("CUDA is not available. Running on CPU.")
