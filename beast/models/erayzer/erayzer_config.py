"""Pydantic config schemas for the ERayZer multi-view 3DGS model."""

from typing import Literal

from pydantic import BaseModel


class ERayZerImageTokenizerConfig(BaseModel):
    """Patch tokenizer configuration."""

    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3


class ERayZerTargetImageConfig(BaseModel):
    """Novel-view render resolution."""

    height: int = 256
    width: int = 256


class ERayZerTransformerConfig(BaseModel):
    """Transformer architecture settings."""

    d: int
    d_head: int
    encoder_n_layer: int = 0
    encoder_geom_n_layer: int
    use_qk_norm: bool = False
    special_init: bool = False
    depth_init: bool = False
    fix_decoder: bool = False


class ERayZerPoseLatentConfig(BaseModel):
    """Camera-prediction pose parameterization settings."""

    canonical: Literal['first', 'middle', 'unordered'] = 'first'
    mode: Literal['pairwise', 'global'] = 'pairwise'
    representation: Literal['6d', 'quat'] = '6d'


class ERayZerRangeSettingConfig(BaseModel):
    """Depth range mapping settings for the Gaussian z-coordinate.

    ``near`` and ``far`` are only used for linear_depth, log_depth, and
    disparity types.  object_centric_depth ignores them.
    """

    type: Literal[
        'object_centric_depth',
        'linear_depth',
        'log_depth',
        'disparity',
    ] = 'object_centric_depth'
    near: float = 0.0
    far: float = 500.0


class ERayZerGaussiansConfig(BaseModel):
    """Gaussian splatting settings."""

    sh_degree: int = 3
    range_setting: ERayZerRangeSettingConfig = ERayZerRangeSettingConfig()


class ERayZerModelConfig(BaseModel):
    """Complete model-section config for ERayZer."""

    model_class: Literal['erayzer']
    seed: int = 0
    checkpoint: str | None = None

    image_tokenizer: ERayZerImageTokenizerConfig = ERayZerImageTokenizerConfig()
    target_image: ERayZerTargetImageConfig = ERayZerTargetImageConfig()
    transformer: ERayZerTransformerConfig
    pose_latent: ERayZerPoseLatentConfig = ERayZerPoseLatentConfig()
    gaussians: ERayZerGaussiansConfig = ERayZerGaussiansConfig()

    hard_pixelalign: bool = False
    input_with_pe: bool = True
    mask_ratio: float = 0.0
    scaling_bias: float = -2.3
    scaling_max: float = -1.2
    opacity_bias: float = -2.0
    near_plane: float = 0.2
    use_deferred_rendering: bool = False
