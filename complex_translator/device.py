import torch

def get_device() -> torch.device:
    """
    Detects the best available hardware accelerator and returns a torch.device object.
    Supports NVIDIA CUDA, Apple Silicon MPS, and standard CPU.
    """
    if torch.cuda.is_available():
        # NVIDIA GPU
        device_name = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon M1/M2/M3/M4 Mac GPU
        device_name = "mps"
    else:
        # Default fallback
        device_name = "cpu"
        
    return torch.device(device_name)

# Create a global instance so other files can import it directly
device = get_device()

if __name__ == "__main__":
    # Test the script by running it directly
    print(f"Using translation device: {device.type.upper()}")
