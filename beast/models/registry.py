"""Central registries for BEAST model classes, training functions, and config schemas.

Three dicts are populated by _register_all() at import time:

  MODEL_REGISTRY   model_class str → LightningModule subclass
  TRAIN_REGISTRY   model_class str → train(config, model, output_dir) callable
  CONFIG_REGISTRY  model_class str → model-section Pydantic config class

To register a new model, add three lines inside _register_all():
  MODEL_REGISTRY['<name>']  = <ModelClass>
  TRAIN_REGISTRY['<name>']  = <train_fn>
  CONFIG_REGISTRY['<name>'] = <ModelConfigClass>

See docs/developer_guide.md for the full step-by-step checklist.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

MODEL_REGISTRY: dict[str, type] = {}
TRAIN_REGISTRY: dict[str, Callable] = {}
CONFIG_REGISTRY: dict[str, type[BaseModel]] = {}


def _register_all() -> None:
    """Import all model packages so they can populate the registries."""
    from beast.models.beast_resnet.beast_resnet_config import ResnetModelConfig
    from beast.models.beast_resnet.beast_resnet_model import ResnetAutoencoder
    from beast.models.beast_resnet.beast_resnet_train import train as resnet_train
    from beast.models.beast_vit.beast_vit_config import VitModelConfig
    from beast.models.beast_vit.beast_vit_model import VisionTransformer
    from beast.models.beast_vit.beast_vit_train import train as vit_train
    from beast.models.erayzer.erayzer_config import ERayZerModelConfig
    from beast.models.erayzer.erayzer_model import ERayZer
    from beast.models.erayzer.erayzer_train import train as erayzer_train

    MODEL_REGISTRY['resnet'] = ResnetAutoencoder
    TRAIN_REGISTRY['resnet'] = resnet_train
    CONFIG_REGISTRY['resnet'] = ResnetModelConfig

    MODEL_REGISTRY['vit'] = VisionTransformer
    TRAIN_REGISTRY['vit'] = vit_train
    CONFIG_REGISTRY['vit'] = VitModelConfig

    MODEL_REGISTRY['erayzer'] = ERayZer
    TRAIN_REGISTRY['erayzer'] = erayzer_train
    CONFIG_REGISTRY['erayzer'] = ERayZerModelConfig


def get_model_class(model_class: str) -> type:
    """Return the model class for the given model_class string.

    Parameters
    ----------
    model_class: model type identifier string (e.g., 'resnet', 'vit')

    Returns
    -------
    model class

    Raises
    ------
    KeyError: if model_class is not registered

    """
    if model_class not in MODEL_REGISTRY:
        raise KeyError(
            f'Unknown model class {model_class!r}. '
            f'Registered: {sorted(MODEL_REGISTRY)}'
        )
    return MODEL_REGISTRY[model_class]


def get_train_fn(model_class: str) -> Callable[..., Any]:
    """Return the training function for the given model_class string.

    Parameters
    ----------
    model_class: model type identifier string

    Returns
    -------
    training callable

    Raises
    ------
    KeyError: if model_class is not registered

    """
    if model_class not in TRAIN_REGISTRY:
        raise KeyError(
            f'No training function registered for {model_class!r}. '
            f'Registered: {sorted(TRAIN_REGISTRY)}'
        )
    return TRAIN_REGISTRY[model_class]


_register_all()
