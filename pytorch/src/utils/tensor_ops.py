import torch
import torch.nn.functional as F


def smooth_sign(x: torch.Tensor, *, alpha: float = 100.0) -> torch.Tensor:
    """
    Smooth version of sign function. Larger `alpha` => better approximation.
    """
    return torch.tanh(alpha * x)


def smooth_min(x: torch.Tensor, *, dim: int, temp: float = 0.01) -> torch.Tensor:
    """
    Smooth version of min function along a given dimension `dim`. Smaller `temp` => better approximation.
    """
    return torch.sum(x * F.softmin(x / temp, dim=dim), dim=dim)
