"""Top-level pydantic config classes and per-model dispatch.

Shared schemas (``BaseTrainingConfig``, ``BaseOptimizerConfig``, ``DataConfig``) live in
``beast.configs.base``; per-model section schemas live in each model's ``*_config.py``.

Top-level config classes
------------------------
BeastConfig
    Used for models that share the standard training loop (resnet, vit).
    Contains TrainingConfig, BaseOptimizerConfig, and DataConfig.

ERayZerBeastConfig, Beast3DBeastConfig
    Used for ERayZer and BEAST3D.  Contain ERayZerTrainingConfig.

TrainingConfig here is the standard epoch-based-loop schema; it extends the base schema in
``beast.configs.base``. Every model uses ``BaseOptimizerConfig`` unchanged.

Per-model dispatch
------------------
get_beast_config_class(model_class) returns the correct top-level config class for a
given model_class string.  Both beast.io.load_config and beast.api.model.Model.from_config
use this function so that YAML files and raw dicts are validated against the right schema.

Load flow
---------
yaml -> beast.io.load_config -> model_validate (this module) -> model_dump() -> plain nested
dict.  Defaults are filled in by validation, so consumers (train.py, models) can index fields
directly.  Some schemas normalize their input in a ``mode='before'`` validator, e.g.
``VitModelParams`` expands ``backbone: vit-l`` into explicit architecture fields; such
normalization must be idempotent because dumped configs are re-validated when a saved model is
reloaded.

To add a new model with a divergent training schema:
  1. Define <Model>TrainingConfig (subclass BaseTrainingConfig) in the model's *_config.py
     (and subclass BaseOptimizerConfig only if the model needs extra optimizer fields)
  2. Define <Model>BeastConfig here
  3. Add 'model_class': <Model>BeastConfig to _MODEL_CONFIG_CLASSES
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from beast.configs.base import BaseOptimizerConfig, BaseTrainingConfig, DataConfig
from beast.models.beast3d.beast3d_config import Beast3DModelConfig
from beast.models.beast_resnet.beast_resnet_config import ResnetModelConfig
from beast.models.beast_vit.beast_vit_config import VitModelConfig
from beast.models.erayzer.erayzer_config import ERayZerModelConfig, ERayZerTrainingConfig


class BeastConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    optimizer: BaseOptimizerConfig
    data: DataConfig
    inference: bool = False


ModelConfig = Annotated[
    ResnetModelConfig | VitModelConfig,
    Field(discriminator='model_class'),
]


class TrainingConfig(BaseTrainingConfig):
    """Training config for models using the standard epoch-based loop (resnet, vit)."""

    imgaug: Literal['none', 'default', 'top-down'] = 'default'
    train_probability: float = 0.95
    val_probability: float = 0.05


class ERayZerBeastConfig(BaseModel):
    """Complete top-level config for ERayZer training runs."""

    model: ERayZerModelConfig
    training: ERayZerTrainingConfig
    optimizer: BaseOptimizerConfig
    data: DataConfig
    inference: bool = False


class Beast3DBeastConfig(BaseModel):
    """Complete top-level config for BEAST3D training runs.

    BEAST3D reuses ERayZer's training and optimizer schemas; only the model
    section differs (see beast.models.beast3d.beast3d_config.Beast3DModelConfig).
    """

    model: Beast3DModelConfig
    training: ERayZerTrainingConfig
    optimizer: BaseOptimizerConfig
    data: DataConfig
    inference: bool = False


_MODEL_CONFIG_CLASSES: dict[str, type[BaseModel]] = {
    'erayzer': ERayZerBeastConfig,
    'beast3d': Beast3DBeastConfig,
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
