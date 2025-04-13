import torch

def print_cuda_info()-> None:
    """
    Print CUDA information if available.
    """
    print('-' * 60)
    print('CUDA Information:')
    print('-' * 60)
    if torch.cuda.is_available():
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        print('CUDA version:', torch.version.cuda)
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA device memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available.")
    print('-' * 60)


def print_torch_version()-> None:
    """
    Print the PyTorch version.
    """
    print(f"PyTorch version: {torch.__version__}")
    
    
def get_device()-> str:
    """
    Get the device to be used for PyTorch operations.
    Returns:
        str: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'