"""Pydantic config schemas for the Vision Transformer autoencoder model."""

from typing import Literal

from pydantic import BaseModel


class VitModelParams(BaseModel):
    """Parameters for the ViT-MAE backbone and decoder."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = 'gelu'
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    decoder_num_attention_heads: int = 16
    decoder_hidden_size: int = 512
    decoder_num_hidden_layers: int = 8
    decoder_intermediate_size: int = 2048
    mask_ratio: float = 0.75
    norm_pix_loss: bool = False
    embed_size: int = 768
    temp_scale: bool = False
    random_init: bool = False
    use_infoNCE: bool = False
    infoNCE_weight: float = 0.03
    use_perceptual_loss: bool = False
    lambda_perceptual: float = 10.0


class VitModelConfig(BaseModel):
    """Top-level model-section config for the Vision Transformer autoencoder."""

    model_class: Literal['vit']
    model_params: VitModelParams
    seed: int = 0
    checkpoint: str | None = None
