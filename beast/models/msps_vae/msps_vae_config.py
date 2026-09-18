"""Pydantic config schemas for the MSPS-VAE model."""

from typing import Literal

from pydantic import BaseModel


class MspsVaeModelParams(BaseModel):
    """Parameters for the MSPS-VAE backbone and latent partitioning."""

    backbone: Literal[
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
    ] = 'resnet18'
    num_latents_unsupervised: int
    num_latents_background: int
    image_size: int = 224
    num_channels: int = 3
    triplet_margin: float = 1.0
    triplet_weight: float = 1.0
    positive_window: int = 1000
    orthogonal_matrix_seed: int = 42
    use_spatial_loss_weight: bool = False
    spatial_loss_weight_r0: float = 0.5


class MspsVaeModelConfig(BaseModel):
    """Top-level model-section config for the MSPS-VAE model."""

    model_class: Literal['msps_vae']
    model_params: MspsVaeModelParams
    seed: int = 0
    checkpoint: str | None = None
