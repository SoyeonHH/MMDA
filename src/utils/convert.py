import torch
from config import get_config

def to_gpu(x, on_cpu=False, gpu_id=None):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        # x = x.cuda(gpu_id)
        config = get_config()
        device = torch.device(config.device)
        x = x.to(device)
    return x

def to_cpu(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        # x = x.cpu()
        device = torch.device('cpu')
        x = x.to(device)
    return x.data