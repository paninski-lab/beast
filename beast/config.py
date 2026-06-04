"""Pydantic models for BEAST configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from beast.models.beast_resnet.beast_resnet_config import ResnetModelConfig
from beast.models.beast_vit.beast_vit_config import VitModelConfig
from beast.models.erayzer.erayzer_config import ERayZerModelConfig


class BeastConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    data: DataConfig
    inference: bool = False


ModelConfig = Annotated[
    ResnetModelConfig | VitModelConfig | ERayZerModelConfig,
    Field(discriminator='model_class'),
]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
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
    model_config = ConfigDict(extra='allow')
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
