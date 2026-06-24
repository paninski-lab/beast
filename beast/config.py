"""Pydantic schemas for BEAST configuration validation and per-model dispatch.

Top-level config classes
------------------------
BeastConfig
    Used for models that share the standard training loop (resnet, vit).
    Contains TrainingConfig, OptimizerConfig, and DataConfig.

ERayZerBeastConfig
    Used for ERayZer and its subclasses (e.g. BEAST3D if they share ERayZer's
    training schema).  Contains ERayZerTrainingConfig and ERayZerOptimizerConfig.

Shared schemas (reusable for new models)
-----------------------------------------
TrainingConfig     — standard epoch-based training fields (batch sizes, epochs, imgaug, …)
OptimizerConfig    — AdamW/Adam + cosine/step scheduler fields
DataConfig         — data_dir only; extend or replace for richer data configs

Per-model dispatch
------------------
get_beast_config_class(model_class) returns the correct top-level config class for a
given model_class string.  Both beast.io.load_config and beast.api.model.Model.from_config
use this function so that YAML files and raw dicts are validated against the right schema.

To add a new model with a divergent training schema:
  1. Define <Model>TrainingConfig and <Model>OptimizerConfig in the model's *_config.py
  2. Define <Model>BeastConfig here (avoids circular imports since DataConfig lives here)
  3. Add 'model_class': <Model>BeastConfig to _MODEL_CONFIG_CLASSES
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from beast.models.beast_resnet.beast_resnet_config import ResnetModelConfig
from beast.models.beast_vit.beast_vit_config import VitModelConfig
from beast.models.erayzer.erayzer_config import (
    ERayZerModelConfig,
    ERayZerOptimizerConfig,
    ERayZerTrainingConfig,
)


class BeastConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    data: DataConfig
    inference: bool = False


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


class ERayZerBeastConfig(BaseModel):
    """Complete top-level config for ERayZer training runs."""

    model: ERayZerModelConfig
    training: ERayZerTrainingConfig
    optimizer: ERayZerOptimizerConfig
    data: DataConfig
    inference: bool = False


_MODEL_CONFIG_CLASSES: dict[str, type[BaseModel]] = {
    'erayzer': ERayZerBeastConfig,
}


def get_beast_config_class(model_class: str) -> type[BaseModel]:
    """Return the top-level Pydantic config class for the given model_class identifier.

    Parameters
    ----------
    model_class: model type identifier string (e.g., 'resnet', 'vit', 'erayzer')

    Returns
    -------
    Pydantic model class for the full top-level config

    """
    return _MODEL_CONFIG_CLASSES.get(model_class, BeastConfig)


BeastConfig.model_rebuild()
