"""Tests for the Vision Transformer autoencoder model."""

import copy

import pytest
import torch
from transformers import ViTMAEConfig

from beast.models.beast_vit.beast_vit_config import VIT_BACKBONES
from beast.models.beast_vit.beast_vit_model import (
    VisionTransformer,
    ViTMAE,
    init_mae_weights,
)


class TestVisionTransformer:
    """Test VisionTransformer forward pass and output shapes."""

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


class TestVisionTransformerIntegration:
    """Integration tests that train and run inference on a ViT autoencoder."""

    def test_integration_basic(self, config_vit, run_model_test) -> None:
        """Test ViT autoencoder with basic masked autoencoder loss."""
        config = copy.deepcopy(config_vit)
        run_model_test(config=config)

    def test_integration_contrastive(self, config_vit, run_model_test) -> None:
        """Test ViT autoencoder with contrastive learning (infoNCE) enabled."""
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['use_infoNCE'] = True
        run_model_test(config=config)

    def test_integration_perceptual_loss(self, config_vit, run_model_test) -> None:
        """Test ViT autoencoder with AlexNet perceptual loss enabled."""
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['use_perceptual_loss'] = True
        run_model_test(config=config)


class TestGetVizPanels:
    """Test VisionTransformer.get_viz_panels."""

    def test_panels_shapes_and_range(self, config_vit):
        model = VisionTransformer(copy.deepcopy(config_vit))
        model.eval()
        panels = model.get_viz_panels({'image': torch.randn(3, 3, 224, 224)})
        assert len(panels) == 3
        for p in panels:
            assert p.shape == (3, 3, 224, 224)
            assert p.min() >= 0 and p.max() <= 1

    def test_masking_is_fixed_across_calls(self, config_vit):
        model = VisionTransformer(copy.deepcopy(config_vit))
        model.eval()
        batch = {'image': torch.randn(2, 3, 224, 224)}
        a = model.get_viz_panels(batch)
        b = model.get_viz_panels(batch)
        assert torch.equal(a[1], b[1])

    def test_masked_panel_hides_patches(self, config_vit):
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['mask_ratio'] = 0.75
        model = VisionTransformer(config)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        inp, masked, _ = model.get_viz_panels({'image': x})
        frac_same = (inp == masked).all(dim=1).float().mean().item()
        assert 0.15 < frac_same < 0.35  # ~25% of patches visible

    def test_norm_pix_loss_branch(self, config_vit):
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['norm_pix_loss'] = True
        model = VisionTransformer(config)
        model.eval()
        panels = model.get_viz_panels({'image': torch.randn(2, 3, 224, 224)})
        assert panels[2].shape == (2, 3, 224, 224)


class TestValidationVizIntegration:
    """Test that training with viz_every_n_epochs writes contact sheets."""

    def test_train_saves_contact_sheets(self, config_vit, tmp_path):
        from beast.api.model import Model

        config = copy.deepcopy(config_vit)
        config['training']['num_epochs'] = 2
        config['training']['check_val_every_n_epoch'] = 1
        config['training']['viz_every_n_epochs'] = 2
        config['training']['viz_num_frames'] = 5
        model = Model.from_config(config)
        model.train(tmp_path)
        sheets = sorted(tmp_path.rglob('val_images/*.png'))
        assert [s.name for s in sheets] == ['epoch_0001_sheet_00.png']


class TestInitMaeWeights:
    """Test the function init_mae_weights."""

    @staticmethod
    def _make_model() -> ViTMAE:
        config = ViTMAEConfig(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
            decoder_hidden_size=16, decoder_num_hidden_layers=1, decoder_num_attention_heads=2,
            decoder_intermediate_size=32, image_size=32, patch_size=16,
        )
        model = ViTMAE(config)
        init_mae_weights(model)
        return model

    def test_init_mae_weights_sets_sincos_position_embeddings(self) -> None:
        model = self._make_model()
        pos = model.vit.embeddings.position_embeddings
        assert not pos.requires_grad
        assert pos.abs().sum() > 0
        # the class-token slot is zero; patch positions are all distinct
        assert torch.equal(pos[0, 0], torch.zeros_like(pos[0, 0]))
        assert torch.unique(pos[0, 1:], dim=0).shape[0] == pos.shape[1] - 1
        assert model.decoder.decoder_pos_embed.abs().sum() > 0

    def test_init_mae_weights_sets_tokens(self) -> None:
        model = self._make_model()
        assert model.vit.embeddings.cls_token.std() > 0
        assert model.decoder.mask_token.std() > 0

    def test_init_mae_weights_deterministic_position_embeddings(self) -> None:
        pos_a = self._make_model().vit.embeddings.position_embeddings
        pos_b = self._make_model().vit.embeddings.position_embeddings
        assert torch.equal(pos_a, pos_b)

    def test_random_init_model_has_position_embeddings(self, config_vit) -> None:
        config_vit['model']['model_params']['random_init'] = True
        model = VisionTransformer(config_vit)
        assert model.vit_mae.vit.embeddings.position_embeddings.abs().sum() > 0


class TestBackbones:
    """Test building the model from backbone aliases."""

    def test_small_backbone_random_init(self, config_vit):
        config = copy.deepcopy(config_vit)
        mp = config['model']['model_params']
        for key in VIT_BACKBONES['vit-b']:  # drop fields already resolved from vit-b
            mp.pop(key, None)
        mp.update({'backbone': 'vit-s', 'random_init': True, 'embed_size': 384})
        model = VisionTransformer(config)
        assert model.vit_mae.config.hidden_size == 384
        assert model.vit_mae.config.num_attention_heads == 6
        assert model.vit_mae.config.embed_size == 384

    def test_tiny_backbone_random_init(self, config_vit):
        config = copy.deepcopy(config_vit)
        mp = config['model']['model_params']
        for key in VIT_BACKBONES['vit-b']:  # drop fields already resolved from vit-b
            mp.pop(key, None)
        mp.update({'backbone': 'vit-t', 'random_init': True, 'embed_size': 192})
        model = VisionTransformer(config)
        assert model.vit_mae.config.hidden_size == 192
        assert model.vit_mae.config.num_attention_heads == 3
        assert model.vit_mae.config.embed_size == 192

    def test_legacy_config_without_backbone(self, config_vit):
        config = copy.deepcopy(config_vit)
        mp = config['model']['model_params']
        mp.pop('backbone', None)
        mp.update({'hidden_size': 384, 'num_attention_heads': 6, 'intermediate_size': 1536})
        mp.update({'random_init': True, 'embed_size': 384})
        model = VisionTransformer(config)
        assert model.vit_mae.config.hidden_size == 384

    def test_large_backbone_loads_large_weights(self, config_vit):
        from unittest import mock

        config = copy.deepcopy(config_vit)
        mp = config['model']['model_params']
        for key in VIT_BACKBONES['vit-b']:
            mp.pop(key, None)
        mp.update({'backbone': 'vit-l', 'embed_size': 1024})
        with mock.patch(
            'beast.models.beast_vit.beast_vit_model.ViTMAE.from_pretrained',
            side_effect=RuntimeError('stop'),
        ) as patched:
            with pytest.raises(RuntimeError, match='stop'):
                VisionTransformer(config)
        args, kwargs = patched.call_args
        assert args[0] == 'facebook/vit-mae-large'
        assert kwargs['config'].hidden_size == 1024
        assert kwargs['config'].num_hidden_layers == 24
