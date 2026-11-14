"""Tests for the ResNet autoencoder model."""

import copy

import torch

from beast.models.beast_resnet.beast_resnet_model import ResnetAutoencoder


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


class TestGetVizPanels:
    """Test ResnetAutoencoder.get_viz_panels."""

    def test_panels_shapes_and_range(self, config_ae):
        config = copy.deepcopy(config_ae)
        config['model']['model_params']['backbone'] = 'resnet18'
        model = ResnetAutoencoder(config)
        model.eval()
        panels = model.get_viz_panels({'image': torch.randn(2, 3, 224, 224)})
        assert len(panels) == 2
        for p in panels:
            assert p.shape == (2, 3, 224, 224)
            assert p.min() >= 0 and p.max() <= 1


class TestPeriodicCheckpointIntegration:
    """Test that ckpt_every_n_epochs saves periodic checkpoints alongside the best one."""

    def test_train_saves_periodic_and_best_checkpoints(self, config_ae, tmp_path):
        from beast.api.model import Model

        config = copy.deepcopy(config_ae)
        config['training']['num_epochs'] = 4
        config['training']['check_val_every_n_epoch'] = 1
        config['training']['ckpt_every_n_epochs'] = 2
        model = Model.from_config(config)
        model.train(tmp_path)
        names = [p.name for p in tmp_path.rglob('*.ckpt')]
        assert any(n.endswith('-best.ckpt') for n in names)
        periodic = [n for n in names if not n.endswith('-best.ckpt')]
        assert sorted(n.split('-')[0] for n in periodic) == ['epoch=1', 'epoch=3']

    def test_train_default_saves_only_best(self, config_ae, tmp_path):
        from beast.api.model import Model

        config = copy.deepcopy(config_ae)
        config['training']['num_epochs'] = 2
        config['training']['check_val_every_n_epoch'] = 1
        model = Model.from_config(config)
        model.train(tmp_path)
        names = [p.name for p in tmp_path.rglob('*.ckpt')]
        assert names and all(n.endswith('-best.ckpt') for n in names)
