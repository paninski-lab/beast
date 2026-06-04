"""Tests for the ResNet autoencoder model."""

import copy

import pytest
import torch

from beast.models.beast_resnet.beast_resnet_model import (
    _RESNET_CONFIGS,
    ResnetAutoencoder,
    get_configs,
)


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


class TestResnetAutoencoder:
    """Test ResnetAutoencoder forward pass shapes."""

    def test_forward_features(self, config_ae):
        config = copy.deepcopy(config_ae)
        config['model']['model_params']['num_latents'] = None

        input = torch.randn((5, 3, 224, 224))

        config['model']['model_params']['backbone'] = 'resnet18'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], 512, 7, 7)

        config['model']['model_params']['backbone'] = 'resnet34'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], 512, 7, 7)

        config['model']['model_params']['backbone'] = 'resnet50'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], 2048, 7, 7)

        config['model']['model_params']['backbone'] = 'resnet101'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], 2048, 7, 7)

        config['model']['model_params']['backbone'] = 'resnet152'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], 2048, 7, 7)

    def test_forward_latents(self, config_ae):
        config = copy.deepcopy(config_ae)
        num_latents = 16
        config['model']['model_params']['num_latents'] = num_latents

        input = torch.randn((5, 3, 224, 224))

        config['model']['model_params']['backbone'] = 'resnet18'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], num_latents)

        config['model']['model_params']['backbone'] = 'resnet34'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], num_latents)

        config['model']['model_params']['backbone'] = 'resnet50'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], num_latents)

        config['model']['model_params']['backbone'] = 'resnet101'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], num_latents)

        config['model']['model_params']['backbone'] = 'resnet152'
        model = ResnetAutoencoder(config)
        reconstructions, latents = model(input)
        assert reconstructions.shape == input.shape
        assert latents.shape == (input.shape[0], num_latents)

    def test_predict_step_return_reconstructions(self, config_ae):
        config = copy.deepcopy(config_ae)
        config['model']['model_params']['backbone'] = 'resnet18'
        config['model']['model_params']['num_latents'] = 16
        model = ResnetAutoencoder(config)
        model.eval()

        batch_dict = {
            'image': torch.randn(2, 3, 224, 224),
            'video': ['vid_a', 'vid_b'],
            'idx': torch.tensor([0, 1]),
            'image_path': ['/fake/0.png', '/fake/1.png'],
        }

        model.return_reconstructions = True
        result = model.predict_step(batch_dict, 0)
        assert 'reconstructions' in result

        model.return_reconstructions = False
        result = model.predict_step(batch_dict, 0)
        assert 'reconstructions' not in result


class TestResnetAutoencoderIntegration:
    """Integration tests that train and run inference on a ResNet autoencoder."""

    def test_integration_features(self, config_ae, run_model_test) -> None:
        """Test ResNet autoencoder with spatial features, no bottleneck."""
        config = copy.deepcopy(config_ae)
        config['model']['model_params']['backbone'] = 'resnet18'
        config['model']['model_params']['num_latents'] = None
        run_model_test(config=config)

    def test_integration_latents(self, config_ae, run_model_test) -> None:
        """Test ResNet autoencoder with low-dimensional latent bottleneck."""
        config = copy.deepcopy(config_ae)
        config['model']['model_params']['backbone'] = 'resnet18'
        config['model']['model_params']['num_latents'] = 16
        run_model_test(config=config)
