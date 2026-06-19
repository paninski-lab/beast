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
    per_view_focal: bool = False


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
    """Complete model-section config for ERayZer.

    ``checkpoint`` resumes a full Lightning training state. ``init_checkpoint``
    instead loads model weights only (``strict=False``, dropping non-model keys
    such as the perceptual loss network) to warm-start fine-tuning from a
    pretrained ERayZer checkpoint.
    """

    model_class: Literal['erayzer']
    seed: int = 0
    checkpoint: str | None = None
    init_checkpoint: str | None = None

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


class ERayZerTrainingConfig(BaseModel):
    """Training configuration for ERayZer."""

    train_batch_size: int
    val_batch_size: int
    test_batch_size: int = 128
    num_epochs: int = 200
    seed: int = 0
    num_workers: int = 8
    num_gpus: int = 1
    num_nodes: int = 1
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 1
    ckpt_every_n_epochs: int | None = None
    viz_every_n_epochs: int = 1
    num_views: int
    num_input_views: int
    num_target_views: int
    random_num_input_views: bool = False
    min_input_views: int = 2
    max_input_views: int = 5
    freeze_focal_steps: int = 0
    max_fwdbwd_passes: int
    grad_checkpoint_every: int = 1
    train_fraction: float = 0.9
    random_split: bool = False
    random_inputs: bool = False
    render_interpolate: bool = False
    l2_loss_weight: float = 1.0
    gs_reg_loss_weight: float = 0.0
    perceptual_loss_weight: float = 0.0
    pose_consistency_reg_weight: float = 0.0


class ERayZerOptimizerConfig(BaseModel):
    """Optimizer configuration for ERayZer (AdamW + OneCycleLR)."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.95
    wd: float = 0.05
    warmup: int = 3000
    div_factor: float = 1.0
    final_div_factor: float = 1.0
    accumulate_grad_batches: int = 1
