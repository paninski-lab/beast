import copy

import torch

from beast.models.vits import VisionTransformer


class TestVisionTransformer:

    def test_forward(self, config_vit):
        config = copy.deepcopy(config_vit)
        input = torch.randn((5, 3, 224, 224))
        model = VisionTransformer(config)
        results = model.forward(input)
        assert 'latents' in results
        assert 'loss' in results
        assert 'reconstructions' in results
        assert results['reconstructions'].shape[0] == input.shape[0]

    def test_get_model_outputs(self, config_vit):
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

    def test_predict_step_return_reconstructions(self, config_vit):
        config = copy.deepcopy(config_vit)
        model = VisionTransformer(config)
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


def test_vit_autoencoder_integration(config_vit, run_model_test):
    """Test ViT autoencoder with basic masked autoencoder loss."""
    config = copy.deepcopy(config_vit)
    run_model_test(config=config)


def test_vit_autoencoder_contrastive_integration(config_vit, run_model_test):
    """Test ViT autoencoder with contrastive learning (infoNCE) enabled."""
    config = copy.deepcopy(config_vit)
    config['model']['model_params']['use_infoNCE'] = True
    run_model_test(config=config)


def test_alex_perceptual_integration(config_vit, run_model_test):
    """Test ViT autoencoder with AlexNet perceptual loss enabled."""
    config = copy.deepcopy(config_vit)
    config['model']['model_params']['use_perceptual_loss'] = True
    run_model_test(config=config)
