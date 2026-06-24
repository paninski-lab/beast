"""Tests for beast.models.beast3d.beast3d_model.Beast3D."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from beast.models.beast3d.beast3d_config import Beast3DModelConfig
from beast.models.beast3d.beast3d_model import Beast3D
from tests.models.beast3d.conftest import requires_gsplat_cuda

_MIN_TRANSFORMER = {'d': 768, 'd_head': 64, 'encoder_geom_n_layer': 16}


class _FakeDinoV3(nn.Module):
    """Stand-in for beast.nn.dino.DinoV3 that needs no network download.

    Emits (H/8 x W/8) patch tokens to match the shrunk test config (image_size
    32, patch_size 8 → 4x4 = 16 tokens), with a configurable embed_dim.
    """

    def __init__(self, model_name: str = '', freeze: bool = True, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3, embed_dim)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, images):
        b, v, _, h, w = images.shape
        n = (h // 8) * (w // 8)
        patch = torch.zeros(b, v, n, self.embed_dim, device=images.device)
        cls = torch.zeros(b, v, self.embed_dim, device=images.device)
        return patch, cls


def _gt_cameras(b: int, v: int) -> dict:
    """Identity-ish GT cameras (in-pixels intrinsics) for a 32x32 render."""
    c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(b, v, 1, 1)
    fxfycxcy = torch.tensor([40.0, 40.0, 16.0, 16.0]).reshape(1, 1, 4).repeat(b, v, 1)
    return {'c2w': c2w, 'fxfycxcy': fxfycxcy}


# ---------------------------------------------------------------------------
# TestBeast3DModelConfig
# ---------------------------------------------------------------------------


class TestBeast3DModelConfig:
    """Test the Beast3DModelConfig schema."""

    def test_valid_config_defaults(self) -> None:
        cfg = Beast3DModelConfig.model_validate(
            {'model_class': 'beast3d', 'transformer': _MIN_TRANSFORMER},
        )
        assert cfg.model_class == 'beast3d'
        assert cfg.use_dinov3 is True
        assert cfg.freeze_dinov3 is True
        assert cfg.frustum_constraint is True
        assert cfg.random_background is True
        assert cfg.mask_loss_weight == 0.1

    def test_wrong_model_class_raises(self) -> None:
        with pytest.raises(ValidationError):
            Beast3DModelConfig.model_validate(
                {'model_class': 'erayzer', 'transformer': _MIN_TRANSFORMER},
            )

    def test_missing_transformer_raises(self) -> None:
        with pytest.raises(ValidationError):
            Beast3DModelConfig.model_validate({'model_class': 'beast3d'})

    def test_inherits_erayzer_fields(self) -> None:
        # hard_pixelalign and gaussians come from the ERayZer base config
        cfg = Beast3DModelConfig.model_validate(
            {'model_class': 'beast3d', 'transformer': _MIN_TRANSFORMER, 'hard_pixelalign': True},
        )
        assert cfg.hard_pixelalign is True
        assert cfg.gaussians.sh_degree == 3


# ---------------------------------------------------------------------------
# TestBeast3DConstruction
# ---------------------------------------------------------------------------


class TestBeast3DConstruction:
    """Test Beast3D model construction."""

    def test_construction_without_dinov3_keeps_patch_tokenizer(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        assert hasattr(model, 'image_tokenizer')
        assert not hasattr(model, 'dinov3')
        # GT cameras → pose-prediction branch is never built
        assert not hasattr(model, 'transformer_encoder')
        assert not hasattr(model, 'pose_predictor')

    def test_construction_with_dinov3_swaps_tokenizer(self, config_beast3d) -> None:
        config = copy.deepcopy(config_beast3d)
        config['model']['use_dinov3'] = True
        d = config['model']['transformer']['d']
        with patch(
            'beast.models.beast3d.beast3d_model.DinoV3',
            side_effect=lambda **kw: _FakeDinoV3(embed_dim=d, **kw),
        ):
            model = Beast3D(config)
        assert hasattr(model, 'dinov3')
        assert not hasattr(model, 'image_tokenizer')

    def test_dinov3_embed_dim_mismatch_raises(self, config_beast3d) -> None:
        config = copy.deepcopy(config_beast3d)
        config['model']['use_dinov3'] = True
        with patch(
            'beast.models.beast3d.beast3d_model.DinoV3',
            side_effect=lambda **kw: _FakeDinoV3(embed_dim=999, **kw),
        ):
            with pytest.raises(ValueError, match='embed_dim'):
                Beast3D(config)


# ---------------------------------------------------------------------------
# TestBeast3DResolveCameras
# ---------------------------------------------------------------------------


class TestBeast3DResolveCameras:
    """Test the GT-camera _resolve_cameras override."""

    def test_returns_gt_cameras_unnormalized(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        data = _gt_cameras(b=2, v=3)
        c2w, fxfycxcy, normalized = model._resolve_cameras(
            None, b=2, v_all=3, n=16, data=data, device=torch.device('cpu'),
        )
        assert normalized is False
        assert c2w.shape == (2, 3, 4, 4)
        assert fxfycxcy.shape == (2, 3, 4)
        assert torch.allclose(c2w, data['c2w'])

    def test_pads_cameras_to_v_all(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        data = _gt_cameras(b=1, v=3)
        c2w, fxfycxcy, _ = model._resolve_cameras(
            None, b=1, v_all=10, n=16, data=data, device=torch.device('cpu'),
        )
        assert c2w.shape == (1, 10, 4, 4)
        assert fxfycxcy.shape == (1, 10, 4)
        # padded views repeat the last real view
        assert torch.allclose(c2w[:, 3:], data['c2w'][:, -1:].expand(-1, 7, -1, -1))


# ---------------------------------------------------------------------------
# TestBeast3DTokenize
# ---------------------------------------------------------------------------


class TestBeast3DTokenize:
    """Test the _tokenize_images override."""

    def test_without_dinov3_matches_base(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        h = w = config_beast3d['model']['image_tokenizer']['image_size']
        images = torch.rand(1, 2, 3, h, w)
        out = model._tokenize_images(images)
        expected = model.image_tokenizer(images * 2.0 - 1.0)
        assert torch.allclose(out, expected)

    def test_with_dinov3_returns_flat_tokens(self, config_beast3d) -> None:
        config = copy.deepcopy(config_beast3d)
        config['model']['use_dinov3'] = True
        d = config['model']['transformer']['d']
        with patch(
            'beast.models.beast3d.beast3d_model.DinoV3',
            side_effect=lambda **kw: _FakeDinoV3(embed_dim=d, **kw),
        ):
            model = Beast3D(config)
        h = w = config['model']['image_tokenizer']['image_size']
        images = torch.rand(2, 3, 3, h, w)
        out = model._tokenize_images(images)
        assert out.shape == (2 * 3, (h // 8) * (w // 8), d)


# ---------------------------------------------------------------------------
# TestBeast3DBackgroundAndTarget
# ---------------------------------------------------------------------------


class TestBeast3DBackgroundAndTarget:
    """Test _sample_background and _prepare_target."""

    def test_background_none_in_eval(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        model.eval()
        assert model._sample_background(torch.device('cpu'), torch.float32) is None

    def test_background_random_in_train(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        model.train()
        bg = model._sample_background(torch.device('cpu'), torch.float32)
        assert bg is not None and bg.shape == (3,)

    def test_prepare_target_no_mask_is_identity(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        target = torch.rand(2, 1, 3, 8, 8)
        out, mask = model._prepare_target(
            target, {}, torch.zeros(2, 1, dtype=torch.long),
            torch.zeros(2, 1, dtype=torch.long), None,
        )
        assert out is target
        assert mask is None

    def test_prepare_target_composites_background(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        b, v, h, w = 2, 1, 8, 8
        target = torch.ones(b, v, 3, h, w)
        full_mask = torch.zeros(b, 3, 1, h, w)  # 3 views available, fg mask = 0 everywhere
        data = {'input_mask': full_mask}
        batch_idx = torch.arange(b).unsqueeze(1)
        target_idx = torch.zeros(b, v, dtype=torch.long)
        bg = torch.tensor([0.2, 0.4, 0.6])
        out, mask = model._prepare_target(target, data, batch_idx, target_idx, bg)
        assert mask.shape == (b, v, 1, h, w)
        # with a zero foreground mask the whole image becomes the background colour
        assert torch.allclose(out[:, :, 0], torch.full((b, v, h, w), 0.2))
        assert torch.allclose(out[:, :, 2], torch.full((b, v, h, w), 0.6))


# ---------------------------------------------------------------------------
# TestBeast3DComputeLoss
# ---------------------------------------------------------------------------


class TestBeast3DComputeLoss:
    """Test the mask/alpha term added to compute_loss."""

    def _base_kwargs(self, b=2, v=1, h=8, w=8) -> dict:
        return {
            'render': torch.rand(b, v, 3, h, w),
            'target_image': torch.rand(b, v, 3, h, w),
        }

    def test_adds_mask_term_when_alpha_and_mask_present(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        b, v, h, w = 2, 1, 8, 8
        kwargs = self._base_kwargs(b, v, h, w)
        kwargs['render_alphas'] = torch.rand(b, v, 1, h, w)
        kwargs['pixel_mask'] = (torch.rand(b, v, 1, h, w) > 0.5).float()
        loss, log_list = model.compute_loss('train', **kwargs)
        assert loss.ndim == 0
        assert any(d['name'] == 'train_mask' for d in log_list)

    def test_no_mask_term_without_pixel_mask(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        kwargs = self._base_kwargs()
        kwargs['render_alphas'] = torch.rand(2, 1, 1, 8, 8)
        kwargs['pixel_mask'] = None
        _, log_list = model.compute_loss('train', **kwargs)
        assert not any(d['name'] == 'train_mask' for d in log_list)


# ---------------------------------------------------------------------------
# TestBeast3DForward
# ---------------------------------------------------------------------------


class TestBeast3DForward:
    """Test the full forward pass with a mocked renderer (no gsplat)."""

    @staticmethod
    def _fake_render(xyz, features, scaling, rotation, opacity,
                     height, width, C2W, fxfycxcy,
                     frustum_constraint=False, backgrounds=None):
        return SimpleNamespace(
            render=torch.zeros(C2W.shape[0], C2W.shape[1], 3, height, width),
            depth=torch.zeros(C2W.shape[0], C2W.shape[1], 1, height, width),
            alpha=torch.zeros(C2W.shape[0], C2W.shape[1], 1, height, width),
        )

    def test_forward_uses_gt_cameras(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        model.eval()
        b, v, h, w = 1, 3, 32, 32
        data = {'image': torch.rand(b, v, 3, h, w), **_gt_cameras(b, v)}
        with patch.object(model.renderer, 'forward', side_effect=self._fake_render):
            result = model(data)
        assert hasattr(result, 'render')
        assert hasattr(result, 'render_alphas')
        # GT cameras flow straight through to the input cameras
        assert torch.allclose(result.c2w_input, data['c2w'][:, result.input_indices[0]])

    def test_forward_with_mask_sets_pixel_mask(self, config_beast3d) -> None:
        model = Beast3D(config_beast3d)
        model.eval()
        b, v, h, w = 1, 3, 32, 32
        data = {
            'image': torch.rand(b, v, 3, h, w),
            'input_mask': (torch.rand(b, v, 1, h, w) > 0.5).float(),
            **_gt_cameras(b, v),
        }
        with patch.object(model.renderer, 'forward', side_effect=self._fake_render):
            result = model(data)
        assert result.pixel_mask is not None
        assert result.pixel_mask.shape[2] == 1  # single foreground channel


# ---------------------------------------------------------------------------
# TestBeast3DIntegration
# ---------------------------------------------------------------------------


class TestBeast3DIntegration:
    """End-to-end train + predict (requires gsplat CUDA and mask fixtures)."""

    @requires_gsplat_cuda
    def test_integration_basic(self, config_beast3d, run_beast3d_model_test) -> None:
        run_beast3d_model_test(config_beast3d)
