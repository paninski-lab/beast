"""Loss functions shared across BEAST models."""

import torch
import torch.nn.functional as F


def masked_mse_loss(
    rendering: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted MSE over RGB pixels using a foreground mask.

    Parameters
    ----------
    rendering: predicted image tensor
    target: target image tensor
    pixel_mask: binary mask tensor; loss is sum(w*(p-t)^2) / sum(w), w repeated across channels

    Returns
    -------
    scalar masked MSE loss

    """
    m = pixel_mask.to(dtype=rendering.dtype, device=rendering.device)
    if m.ndim == 3:
        m = m.unsqueeze(1)
    valid_mask = m.expand_as(rendering)
    loss = F.mse_loss(rendering, target, reduction='none') * valid_mask
    normalizer = valid_mask.sum()
    return loss.sum() / normalizer
