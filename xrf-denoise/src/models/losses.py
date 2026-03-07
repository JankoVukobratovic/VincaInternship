"""Loss functions for XRF spectral denoising."""

import torch
import torch.nn.functional as F


def poisson_nll_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Poisson negative log-likelihood loss.

    More appropriate than MSE for photon counting data because it naturally
    handles heteroscedastic noise (high-count channels have more absolute
    noise but less relative noise).

    L = mean(predicted - target * log(predicted + eps))
    """
    eps = 1e-6
    return torch.mean(predicted - target * torch.log(predicted + eps))


def mixed_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Weighted combination of MSE and Poisson NLL."""
    mse = F.mse_loss(predicted, target)
    pnll = poisson_nll_loss(predicted, target)
    return alpha * mse + (1 - alpha) * pnll


def get_loss_fn(name: str, alpha: float = 0.5):
    """Get loss function by name."""
    if name == "poisson_nll":
        return poisson_nll_loss
    elif name == "mse":
        return F.mse_loss
    elif name == "mixed":
        return lambda pred, tgt: mixed_loss(pred, tgt, alpha)
    else:
        raise ValueError(f"Unknown loss: {name}")
