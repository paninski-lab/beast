"""Tests for the model registry."""

from collections.abc import Callable

import pytest

from beast.models.beast_resnet.beast_resnet_model import ResnetAutoencoder
from beast.models.beast_vit.beast_vit_model import VisionTransformer
from beast.models.erayzer.erayzer_model import ERayZer
from beast.models.registry import MODEL_REGISTRY, TRAIN_REGISTRY, get_model_class, get_train_fn


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
