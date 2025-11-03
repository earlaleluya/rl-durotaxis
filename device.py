import os
import numpy as _np
import torch


def get_device():
    """Return the device to use.

    Priority:
    1. Environment var DEVICE (e.g. 'cpu' or 'cuda')
    2. Environment var FORCE_CPU (if set to '1' or 'true')
    3. Torch CUDA availability
    """
    dev = os.environ.get('DEVICE', None)
    if dev is not None:
        return torch.device(dev)
    force_cpu = os.environ.get('FORCE_CPU', os.environ.get('FORCE_CPU', '')).lower()
    if force_cpu in ('1', 'true', 'yes'):
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(obj, device=None):
    """Move a tensor, module, or nested structure to device."""
    d = device or get_device()
    if isinstance(obj, torch.Tensor):
        return obj.to(d)
    if isinstance(obj, torch.nn.Module):
        return obj.to(d)
    if isinstance(obj, dict):
        return {k: to_device(v, d) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, d) for v in obj)
    return obj


def cpu_numpy(tensor):
    """Safely convert a torch tensor to a NumPy array on CPU.

    Works when tensor is on GPU or CPU and handles None gracefully.
    """
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    # If it's already a numpy array or list-like, convert
    return _np.asarray(tensor)


def tensor(data, dtype=None, device=None):
    """Wrapper for torch.tensor that sets device by default."""
    d = device or get_device()
    if dtype is None:
        return torch.tensor(data, device=d)
    return torch.tensor(data, dtype=dtype, device=d)
