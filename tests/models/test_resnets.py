import copy

import torch


def test_resnet_autoencoder(config_ae):

    from beast.models.resnets import ResnetAutoencoder

    config = copy.deepcopy(config_ae)
    input = torch.randn((5, 3, 224, 224))

    config['model']['model_params']['backbone'] = 'resnet18'
    model = ResnetAutoencoder(config)
    reconstructions, latents = model(input)
    assert reconstructions.shape == input.shape

    config['model']['model_params']['backbone'] = 'resnet34'
    model = ResnetAutoencoder(config)
    reconstructions, latents = model(input)
    assert reconstructions.shape == input.shape

    config['model']['model_params']['backbone'] = 'resnet50'
    model = ResnetAutoencoder(config)
    reconstructions, latents = model(input)
    assert reconstructions.shape == input.shape

    config['model']['model_params']['backbone'] = 'resnet101'
    model = ResnetAutoencoder(config)
    reconstructions, latents = model(input)
    assert reconstructions.shape == input.shape

    config['model']['model_params']['backbone'] = 'resnet152'
    model = ResnetAutoencoder(config)
    reconstructions, latents = model(input)
    assert reconstructions.shape == input.shape
