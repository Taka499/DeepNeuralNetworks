import sys
import torch

def get_device(num_workers=2, set_CPU=False):
    cuda = torch.cuda.is_available()
    print(f"CUDA == {cuda} | Python = {sys.version}")

    if set_CPU:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if cuda else "cpu")
    
    print(torch.cuda.get_device_name(0))
    
    return DEVICE
