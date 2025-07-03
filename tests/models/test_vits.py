import copy

import torch


def test_vit_autoencoder_forward(config_vit):
    from beast.models.vits import VisionTransformer
    config = copy.deepcopy(config_vit)
    input = torch.randn((5, 3, 224, 224))
    model = VisionTransformer(config)
    results = model.forward(input)
    assert 'latents' in results
    assert 'loss' in results
    assert 'reconstructions' in results
    assert results['reconstructions'].shape[0] == input.shape[0]

def test_vit_autoencoder_get_model_outputs(config_vit):
    from beast.models.vits import VisionTransformer
    config = copy.deepcopy(config_vit)
    input = torch.randn((5, 3, 224, 224))
    batch_dict = {'image': input}
    model = VisionTransformer(config)
    results = model.get_model_outputs(batch_dict)
    assert 'latents' in results
    assert 'loss' in results
    assert 'reconstructions' in results
    assert 'images' in results
    assert results['images'].shape == input.shape

def test_vit_autoencoder_integration(config_vit, run_model_test):
    config = copy.deepcopy(config_vit)
    run_model_test(config=config)

def test_vit_autoencoder_contrastive_integration(config_vit, run_model_test):
    """Test ViT autoencoder with contrastive learning (infoNCE) enabled."""
    config = copy.deepcopy(config_vit)
    config['model']['model_params']['use_infoNCE'] = True
    run_model_test(config=config)
