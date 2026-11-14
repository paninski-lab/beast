"""Tests for the ResNet autoencoder building blocks."""

import pytest

from beast.nn.resnet import _RESNET_CONFIGS, get_configs


class TestGetConfigs:
    """Test the get_configs function and _RESNET_CONFIGS registry."""

    def test_all_known_archs_return_correct_types(self) -> None:
        for arch in _RESNET_CONFIGS:
            layers, bottleneck = get_configs(arch)
            assert isinstance(layers, list)
            assert isinstance(bottleneck, bool)

    def test_non_bottleneck_archs(self) -> None:
        for arch in ('resnet18', 'resnet34'):
            _, bottleneck = get_configs(arch)
            assert bottleneck is False

    def test_bottleneck_archs(self) -> None:
        for arch in ('resnet50', 'resnet101', 'resnet152'):
            _, bottleneck = get_configs(arch)
            assert bottleneck is True

    def test_unknown_arch_raises(self) -> None:
        with pytest.raises(ValueError, match='invalid entry'):
            get_configs('resnet999')

    def test_registry_covers_expected_archs(self) -> None:
        assert set(_RESNET_CONFIGS) == {
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        }
