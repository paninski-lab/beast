import copy

import torch


def test_resnet_autoencoder_forward_features(config_ae):

    from beast.models.resnets import ResnetAutoencoder

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


def test_resnet_autoencoder_forward_latents(config_ae):

    from beast.models.resnets import ResnetAutoencoder

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


def test_resnet_autoencoder_integration_features(config_ae, run_model_test):

    config = copy.deepcopy(config_ae)
    config['model']['model_params']['backbone'] = 'resnet18'
    config['model']['model_params']['num_latents'] = None  # no latents, just 2D features

    run_model_test(config=config)


def test_resnet_autoencoder_integration_latents(config_ae, run_model_test):

    config = copy.deepcopy(config_ae)
    config['model']['model_params']['backbone'] = 'resnet18'
    config['model']['model_params']['num_latents'] = 16

    run_model_test(config=config)
