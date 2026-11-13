"""Tests for the model registry."""

from collections.abc import Callable

import pytest

from beast.models.beast3d.beast3d_config import Beast3DBeastConfig
from beast.models.beast_resnet.beast_resnet_config import ResnetBeastConfig
from beast.models.beast_resnet.beast_resnet_model import ResnetAutoencoder
from beast.models.beast_vit.beast_vit_config import VitBeastConfig
from beast.models.beast_vit.beast_vit_model import VisionTransformer
from beast.models.erayzer.erayzer_config import ERayZerBeastConfig
from beast.models.erayzer.erayzer_model import ERayZer
from beast.models.msps_vae.msps_vae_config import MspsVaeBeastConfig
from beast.models.registry import (
    CONFIG_REGISTRY,
    DATAMODULE_REGISTRY,
    MODEL_REGISTRY,
    TRAIN_REGISTRY,
    get_config_class,
    get_datamodule_fn,
    get_model_class,
    get_train_fn,
)


class TestGetModelClass:
    """Test the get_model_class function."""

    def test_resnet_returns_resnet_autoencoder(self) -> None:
        assert get_model_class('resnet') is ResnetAutoencoder

    def test_vit_returns_vision_transformer(self) -> None:
        assert get_model_class('vit') is VisionTransformer

    def test_erayzer_returns_erayzer(self) -> None:
        assert get_model_class('erayzer') is ERayZer

    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_model_class('nonexistent_model')

    def test_error_message_lists_registered_classes(self) -> None:
        with pytest.raises(KeyError, match='resnet'):
            get_model_class('nonexistent_model')

    def test_all_registered_keys_resolve(self) -> None:
        for key in MODEL_REGISTRY:
            assert get_model_class(key) is MODEL_REGISTRY[key]


class TestGetDatamoduleFn:
    """Test the get_datamodule_fn function."""

    def test_registered_models_return_callable(self) -> None:
        for key in ('resnet', 'vit', 'erayzer', 'beast3d', 'msps_vae'):
            assert callable(get_datamodule_fn(key))

    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match='erayzer'):
            get_datamodule_fn('nonexistent_model')

    def test_all_registered_keys_resolve(self) -> None:
        for key in DATAMODULE_REGISTRY:
            assert get_datamodule_fn(key) is DATAMODULE_REGISTRY[key]

    def test_registries_have_same_keys(self) -> None:
        assert set(DATAMODULE_REGISTRY) == set(MODEL_REGISTRY) == set(TRAIN_REGISTRY)


class TestGetTrainFn:
    """Test the get_train_fn function."""

    def test_resnet_returns_callable(self) -> None:
        fn = get_train_fn('resnet')
        assert callable(fn)

    def test_vit_returns_callable(self) -> None:
        fn = get_train_fn('vit')
        assert callable(fn)

    def test_erayzer_returns_callable(self) -> None:
        fn = get_train_fn('erayzer')
        assert callable(fn)

    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_train_fn('nonexistent_model')

    def test_error_message_lists_registered_classes(self) -> None:
        with pytest.raises(KeyError, match='erayzer'):
            get_train_fn('nonexistent_model')

    def test_all_registered_keys_resolve(self) -> None:
        for key in TRAIN_REGISTRY:
            fn = get_train_fn(key)
            assert isinstance(fn, Callable)


class TestConfigRegistry:
    """Test the CONFIG_REGISTRY of top-level config classes."""

    def test_same_keys_as_model_registry(self) -> None:
        assert set(CONFIG_REGISTRY) == set(MODEL_REGISTRY)

    def test_entries_are_top_level_configs(self) -> None:
        from beast.config import BaseBeastConfig
        for key, cls in CONFIG_REGISTRY.items():
            assert issubclass(cls, BaseBeastConfig), key
            assert cls.__name__.endswith('BeastConfig'), key


class TestGetConfigClass:
    """Test the get_config_class dispatcher."""

    @pytest.mark.parametrize(
        ('model_class', 'expected'),
        [
            ('resnet', ResnetBeastConfig),
            ('vit', VitBeastConfig),
            ('erayzer', ERayZerBeastConfig),
            ('beast3d', Beast3DBeastConfig),
            ('msps_vae', MspsVaeBeastConfig),
        ],
    )
    def test_registered_model_class(self, model_class: str, expected: type) -> None:
        assert get_config_class(model_class) is expected

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match='totally_unknown_model'):
            get_config_class('totally_unknown_model')

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match='model_class'):
            get_config_class('')

    def test_error_message_lists_registered_classes(self) -> None:
        with pytest.raises(ValueError, match='erayzer'):
            get_config_class('nope')
