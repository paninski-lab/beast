"""Central registries for BEAST model classes, training functions, and config schemas.

Each model package populates these dicts when its package is imported.
To add a new model: create a beast/models/<name>/ directory and add entries here.
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
    from beast.models.erayzer import ERayZer
    from beast.models.vits import VisionTransformer
    from beast.train import train

    MODEL_REGISTRY['resnet'] = ResnetAutoencoder
    TRAIN_REGISTRY['resnet'] = resnet_train
    CONFIG_REGISTRY['resnet'] = ResnetModelConfig

    MODEL_REGISTRY['vit'] = VisionTransformer
    TRAIN_REGISTRY['vit'] = train

    MODEL_REGISTRY['erayzer'] = ERayZer
    TRAIN_REGISTRY['erayzer'] = train


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
