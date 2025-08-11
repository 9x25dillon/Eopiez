# Stub to route tensors to your GPU MatrixProcessor when available.
import torch

def to_device(x, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device)