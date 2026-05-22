"""Pydantic models for BEAST configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class BeastConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    data: DataConfig


class ResnetModelConfig(BaseModel):
    model_class: Literal['resnet']
    model_params: ResnetModelParams
    seed: int = 0
    checkpoint: str | None = None


class ResnetModelParams(BaseModel):
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


class VitModelConfig(BaseModel):
    model_class: Literal['vit']
    model_params: VitModelParams
    seed: int = 0
    checkpoint: str | None = None


class VitModelParams(BaseModel):
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


ModelConfig = Annotated[
    ResnetModelConfig | VitModelConfig,
    Field(discriminator='model_class'),
]


class TrainingConfig(BaseModel):
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int = 128
    num_epochs: int = 800
    seed: int = 0
    imgaug: Literal['none', 'default', 'top-down'] = 'default'
    num_workers: int = 8
    num_gpus: int = 1
    num_nodes: int = 1
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 5
    train_probability: float = 0.95
    val_probability: float = 0.05
    ckpt_every_n_epochs: int | None = None


class OptimizerConfig(BaseModel):
    lr: float
    type: Literal['Adam', 'AdamW'] = 'AdamW'
    wd: float = 0.05
    scheduler: Literal['step', 'cosine'] = 'cosine'
    accumulate_grad_batches: int = 1
    # for cosine
    warmup_pct: float = 0.15
    div_factor: float = 10.0
    final_div_factor: float = 1.0
    # for step
    gamma: float = 0.5
    steps: list[int] = []


class DataConfig(BaseModel):
    data_dir: str | Path


BeastConfig.model_rebuild()
