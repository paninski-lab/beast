"""Tests for the MSPS-VAE model."""

import copy

import pytest
import torch

from beast.models.msps_vae.msps_vae_model import (
    MspsVae,
    OrthogonalSplit,
    build_raised_cosine_weight_map,
)


class TestBuildRaisedCosineWeightMap:
    """Test the build_raised_cosine_weight_map function."""

    def test_shape(self) -> None:
        weight_map = build_raised_cosine_weight_map(side=32, r0=0.5)
        assert weight_map.shape == (32, 32)

    def test_mean_is_one(self) -> None:
        weight_map = build_raised_cosine_weight_map(side=64, r0=0.5)
        assert torch.isclose(weight_map.mean(), torch.tensor(1.0), atol=1e-5)

    def test_center_is_max(self) -> None:
        weight_map = build_raised_cosine_weight_map(side=33, r0=0.5)
        center = weight_map.shape[0] // 2
        assert weight_map[center, center] == weight_map.max()

    def test_corners_are_zero(self) -> None:
        weight_map = build_raised_cosine_weight_map(side=32, r0=0.5)
        assert weight_map[0, 0] == 0.0
        assert weight_map[0, -1] == 0.0
        assert weight_map[-1, 0] == 0.0
        assert weight_map[-1, -1] == 0.0

    def test_nonnegative(self) -> None:
        weight_map = build_raised_cosine_weight_map(side=32, r0=0.5)
        assert (weight_map >= 0).all()


class TestOrthogonalSplit:
    """Test the OrthogonalSplit module."""

    def test_frozen_weights(self) -> None:
        split = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=42)
        assert split.to_unsupervised.weight.requires_grad is False
        assert split.to_background.weight.requires_grad is False

    def test_weight_shapes(self) -> None:
        split = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=42)
        assert split.to_unsupervised.weight.shape == (8, 12)
        assert split.to_background.weight.shape == (4, 12)

    def test_stacked_weights_are_orthogonal(self) -> None:
        # rows of [to_unsupervised; to_background] should form an orthonormal basis
        split = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=42)
        stacked = torch.cat([split.to_unsupervised.weight, split.to_background.weight], dim=0)
        identity = stacked @ stacked.T
        assert torch.allclose(identity, torch.eye(12), atol=1e-5)

    def test_deterministic_with_seed(self) -> None:
        split1 = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=42)
        split2 = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=42)
        assert torch.allclose(split1.to_unsupervised.weight, split2.to_unsupervised.weight)
        assert torch.allclose(split1.to_background.weight, split2.to_background.weight)

    def test_different_seed_different_matrix(self) -> None:
        split1 = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=42)
        split2 = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=0)
        assert not torch.allclose(split1.to_unsupervised.weight, split2.to_unsupervised.weight)

    def test_forward_shapes(self) -> None:
        split = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=42)
        x = torch.randn(5, 12)
        z_u, z_b = split(x)
        assert z_u.shape == (5, 8)
        assert z_b.shape == (5, 4)

    def test_forward_matches_manual_matmul(self) -> None:
        split = OrthogonalSplit(num_latents_unsupervised=8, num_latents_background=4, seed=42)
        x = torch.randn(3, 12)
        z_u, z_b = split(x)
        assert torch.allclose(z_u, x @ split.to_unsupervised.weight.T, atol=1e-6)
        assert torch.allclose(z_b, x @ split.to_background.weight.T, atol=1e-6)


class TestMspsVae:
    """Test the MspsVae model."""

    def test_forward_shapes(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        model = MspsVae(config)
        x = torch.randn(6, 3, 224, 224)
        xhat, z_u, z_b = model(x)
        assert xhat.shape == x.shape
        assert z_u.shape == (6, 8)
        assert z_b.shape == (6, 4)

    def test_get_model_outputs_keys(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        model = MspsVae(config)
        batch_dict = {
            'image': torch.randn(4, 3, 224, 224),
            'video': ['v1', 'v1', 'v2', 'v2'],
        }
        results = model.get_model_outputs(batch_dict)
        assert set(results) == {'z_u', 'z_b', 'video', 'images', 'reconstructions'}
        assert results['video'] == batch_dict['video']

    def test_compute_loss_train_includes_triplet(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        model = MspsVae(config)
        n = 8  # 4 ref + 4 pos, spanning >=2 videos
        images = torch.randn(n, 3, 224, 224)
        reconstructions = torch.randn(n, 3, 224, 224)
        z_u = torch.randn(n, 8)
        z_b = torch.randn(n, 4)
        video = ['v1', 'v1', 'v2', 'v2', 'v1', 'v1', 'v2', 'v2']

        loss, log_list = model.compute_loss(
            stage='train', images=images, reconstructions=reconstructions,
            z_u=z_u, z_b=z_b, video=video,
        )
        names = {entry['name'] for entry in log_list}
        assert 'train_mse' in names
        assert 'train_triplet' in names
        assert torch.isfinite(loss)

    def test_compute_loss_val_excludes_triplet(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        model = MspsVae(config)
        n = 4
        images = torch.randn(n, 3, 224, 224)
        reconstructions = torch.randn(n, 3, 224, 224)
        z_u = torch.randn(n, 8)
        z_b = torch.randn(n, 4)
        video = ['v1', 'v2', 'v1', 'v2']

        loss, log_list = model.compute_loss(
            stage='val', images=images, reconstructions=reconstructions,
            z_u=z_u, z_b=z_b, video=video,
        )
        names = {entry['name'] for entry in log_list}
        assert 'val_mse' in names
        assert 'val_triplet' not in names
        assert torch.isfinite(loss)

    def test_compute_loss_single_video_batch_raises(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        model = MspsVae(config)
        n = 4
        images = torch.randn(n, 3, 224, 224)
        reconstructions = torch.randn(n, 3, 224, 224)
        z_u = torch.randn(n, 8)
        z_b = torch.randn(n, 4)
        video = ['v1', 'v1', 'v1', 'v1']

        with pytest.raises(ValueError, match='at least two distinct videos'):
            model.compute_loss(
                stage='train', images=images, reconstructions=reconstructions,
                z_u=z_u, z_b=z_b, video=video,
            )

    def test_predict_step_latents_are_concat_of_zu_zb(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        model = MspsVae(config)
        model.eval()

        batch_dict = {
            'image': torch.randn(2, 3, 224, 224),
            'video': ['vid_a', 'vid_b'],
            'idx': torch.tensor([0, 1]),
            'image_path': ['/fake/0.png', '/fake/1.png'],
        }

        result = model.predict_step(batch_dict, 0)
        assert 'z_u' not in result
        assert 'z_b' not in result
        assert 'video' not in result
        assert result['latents'].shape == (2, 8 + 4)
        assert result['metadata']['video'] == batch_dict['video']

    def test_spatial_loss_weight_off_by_default(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        model = MspsVae(config)
        assert model.use_spatial_loss_weight is False
        assert not hasattr(model, 'spatial_loss_weight_map')

    def test_spatial_loss_weight_on_registers_buffer(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        config['model']['model_params']['use_spatial_loss_weight'] = True
        model = MspsVae(config)
        assert model.use_spatial_loss_weight is True
        assert model.spatial_loss_weight_map.shape == (224, 224)

    def test_spatial_loss_weight_changes_mse(self, config_msps_vae) -> None:
        n = 4
        images = torch.randn(n, 3, 224, 224)
        reconstructions = torch.randn(n, 3, 224, 224)
        z_u = torch.randn(n, 8)
        z_b = torch.randn(n, 4)
        video = ['v1', 'v2', 'v1', 'v2']

        config_off = copy.deepcopy(config_msps_vae)
        loss_off, _ = MspsVae(config_off).compute_loss(
            stage='val', images=images, reconstructions=reconstructions,
            z_u=z_u, z_b=z_b, video=video,
        )

        config_on = copy.deepcopy(config_msps_vae)
        config_on['model']['model_params']['use_spatial_loss_weight'] = True
        loss_on, _ = MspsVae(config_on).compute_loss(
            stage='val', images=images, reconstructions=reconstructions,
            z_u=z_u, z_b=z_b, video=video,
        )

        assert torch.isfinite(loss_on)
        assert not torch.isclose(loss_on, loss_off)

    def test_predict_step_return_reconstructions_toggle(self, config_msps_vae) -> None:
        config = copy.deepcopy(config_msps_vae)
        model = MspsVae(config)
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
