"""Pydantic config schemas for the ResNet autoencoder model."""

from typing import Literal

from pydantic import BaseModel


class ResnetModelParams(BaseModel):
    """Parameters for the ResNet backbone and bottleneck."""

    backbone: Literal[
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
    ] = 'resnet18'
    num_latents: int | None = None
    image_size: int = 224
    num_channels: int = 3


class ResnetModelConfig(BaseModel):
    """Top-level model-section config for the ResNet autoencoder."""

    model_class: Literal['resnet']
    model_params: ResnetModelParams
    seed: int = 0
    checkpoint: str | None = None
