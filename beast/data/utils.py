"""Small helpers for working with image batches."""

import torch

from beast.data.datasets import _IMAGENET_MEAN, _IMAGENET_STD


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization and clamp to [0, 1] for display.

    Parameters
    ----------
    images: normalized image batch of shape (batch, 3, height, width)

    Returns
    -------
    detached CPU float tensor in [0, 1] with the same shape

    """
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    images = images.detach().float().cpu()
    return (images * std + mean).clamp(0.0, 1.0)
