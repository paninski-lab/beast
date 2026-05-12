import copy

import torch

from beast.models.resnets import ResnetAutoencoder


class TestResnetAutoencoder:

    def test_forward_features(self, config_ae):

        config = copy.deepcopy(config_ae)
        config['model']['model_params']['num_latents'] = None  # no latents, just 2D features

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


def test_resnet_autoencoder_integration_features(config_ae, run_model_test):
    """Test ResNet autoencoder with spatial features, no bottleneck."""
    config = copy.deepcopy(config_ae)
    config['model']['model_params']['backbone'] = 'resnet18'
    config['model']['model_params']['num_latents'] = None  # no latents, just 2D features

    run_model_test(config=config)


def test_resnet_autoencoder_integration_latents(config_ae, run_model_test):
    """Test ResNet autoencoder with low-dimensional latent bottleneck."""

    config = copy.deepcopy(config_ae)
    config['model']['model_params']['backbone'] = 'resnet18'
    config['model']['model_params']['num_latents'] = 16

    run_model_test(config=config)
