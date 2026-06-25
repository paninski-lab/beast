"""Tests for beast.models.erayzer.erayzer_model components."""

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from beast.models.erayzer.erayzer_model import (
    ERayZer,
    GaussiansUpsampler,
    LossComputer,
    PoseEstimator,
    _filter_init_state_dict,
    _parse_hf_checkpoint_url,
    _resolve_init_checkpoint,
    build_transformer_blocks,
    get_cam_se3,
    get_point_range_func,
    sanitize,
)
from beast.nn.transformer import QK_Norm_TransformerBlock
from tests.models.erayzer.conftest import requires_gsplat_cuda

# ---------------------------------------------------------------------------
# TestSanitize
# ---------------------------------------------------------------------------


class TestSanitize:
    """Test the sanitize function."""

    def test_nan_replaced_by_zero(self) -> None:
        t = torch.tensor([float('nan'), 1.0])
        out = sanitize(t)
        assert out[0].item() == 0.0

    def test_posinf_replaced_by_1e6(self) -> None:
        t = torch.tensor([float('inf')])
        out = sanitize(t)
        assert out[0].item() == pytest.approx(1e6)

    def test_neginf_replaced_by_neg_1e6(self) -> None:
        t = torch.tensor([float('-inf')])
        out = sanitize(t)
        assert out[0].item() == pytest.approx(-1e6)

    def test_finite_values_pass_through(self) -> None:
        t = torch.tensor([0.0, 1.0, -1.0, 100.0])
        assert torch.allclose(sanitize(t), t)

    def test_large_values_clamped(self) -> None:
        t = torch.tensor([2e6, -2e6])
        out = sanitize(t)
        assert out[0].item() == pytest.approx(1e6)
        assert out[1].item() == pytest.approx(-1e6)

    def test_shape_preserved(self) -> None:
        t = torch.randn(3, 4, 5)
        assert sanitize(t).shape == t.shape


# ---------------------------------------------------------------------------
# TestBuildTransformerBlocks
# ---------------------------------------------------------------------------


class TestBuildTransformerBlocks:
    """Test the build_transformer_blocks factory."""

    def test_returns_correct_count(self) -> None:
        blocks = build_transformer_blocks(4, d=32, d_head=8, use_qk_norm=False)
        assert len(blocks) == 4

    def test_each_element_is_correct_type(self) -> None:
        blocks = build_transformer_blocks(2, d=32, d_head=8, use_qk_norm=False)
        for b in blocks:
            assert isinstance(b, QK_Norm_TransformerBlock)

    def test_returns_module_list(self) -> None:
        blocks = build_transformer_blocks(2, d=32, d_head=8, use_qk_norm=False)
        assert isinstance(blocks, nn.ModuleList)

    def test_special_init_no_error(self) -> None:
        build_transformer_blocks(2, d=32, d_head=8, use_qk_norm=False, special_init=True)

    def test_depth_init_no_error(self) -> None:
        build_transformer_blocks(
            2, d=32, d_head=8, use_qk_norm=False, special_init=True, depth_init=True,
        )


# ---------------------------------------------------------------------------
# TestGetCamSe3
# ---------------------------------------------------------------------------


class TestGetCamSe3:
    """Test the get_cam_se3 function."""

    def test_6d_output_shapes(self) -> None:
        cam_info = torch.randn(4, 13)
        c2w, fxfycxcy = get_cam_se3(cam_info)
        assert c2w.shape == (4, 4, 4)
        assert fxfycxcy.shape == (4, 4)

    def test_quat_output_shapes(self) -> None:
        cam_info = torch.randn(4, 11)
        c2w, fxfycxcy = get_cam_se3(cam_info)
        assert c2w.shape == (4, 4, 4)
        assert fxfycxcy.shape == (4, 4)

    def test_unknown_width_raises(self) -> None:
        cam_info = torch.randn(2, 10)
        with pytest.raises(NotImplementedError):
            get_cam_se3(cam_info)

    def test_bottom_row_is_0001(self) -> None:
        cam_info = torch.randn(3, 13)
        c2w, _ = get_cam_se3(cam_info)
        expected = torch.tensor([0., 0., 0., 1.])
        assert torch.allclose(c2w[:, 3, :], expected.expand(3, -1), atol=1e-6)

    def test_identity_6d_gives_identity_rotation(self) -> None:
        # 6D identity: first two columns of I₃ = [1,0,0,0,1,0]
        base = torch.zeros(1, 13)
        base[0, :6] = torch.tensor([1., 0., 0., 0., 1., 0.])
        c2w, _ = get_cam_se3(base)
        assert torch.allclose(c2w[0, :3, :3], torch.eye(3), atol=1e-5)

    def test_fxfycxcy_matches_last_4(self) -> None:
        cam_info = torch.zeros(2, 13)
        cam_info[:, 9:] = torch.tensor([[1., 2., 3., 4.]])
        _, fxfycxcy = get_cam_se3(cam_info)
        assert torch.allclose(fxfycxcy, torch.tensor([[1., 2., 3., 4.]]).expand(2, -1))


# ---------------------------------------------------------------------------
# TestPoseEstimator
# ---------------------------------------------------------------------------


class TestPoseEstimator:
    """Test the PoseEstimator module."""

    def _make_config(self, canonical: str = 'first', mode: str = 'pairwise',
                     rep: str = '6d', per_view_focal: bool = False) -> dict:
        return {
            'model': {
                'transformer': {'d': 32},
                'pose_latent': {
                    'canonical': canonical,
                    'mode': mode,
                    'representation': rep,
                    'per_view_focal': per_view_focal,
                },
            },
            'training': {'pose_consistency_reg_weight': 0.0},
        }

    def test_output_shape_6d_pairwise_first(self) -> None:
        config = self._make_config()
        estimator = PoseEstimator(config)
        b, v = 2, 4
        x = torch.randn(b * v, 32)
        out = estimator(x, v)
        assert out.shape == (b * v, 13)  # 6 rot + 3 trans + 4 intrinsics

    def test_output_shape_quat(self) -> None:
        config = self._make_config(rep='quat')
        estimator = PoseEstimator(config)
        b, v = 2, 3
        x = torch.randn(b * v, 32)
        out = estimator(x, v)
        assert out.shape == (b * v, 11)  # 4 quat + 3 trans + 4 intrinsics

    def test_pairwise_middle_canonical(self) -> None:
        config = self._make_config(canonical='middle')
        estimator = PoseEstimator(config)
        b, v = 2, 4
        x = torch.randn(b * v, 32)
        out = estimator(x, v)
        assert out.shape == (b * v, 13)

    def test_global_unordered(self) -> None:
        config = self._make_config(canonical='unordered', mode='global')
        estimator = PoseEstimator(config)
        b, v = 2, 4
        x = torch.randn(b, v, 32)
        out = estimator(x, v)
        assert out.shape == (b, v, 13)

    def test_single_view_degenerate(self) -> None:
        config = self._make_config()
        estimator = PoseEstimator(config)
        b, v = 2, 1
        x = torch.randn(b * v, 32)
        out = estimator(x, v)
        assert out.shape == (b * v, 13)

    def test_invalid_canonical_raises(self) -> None:
        with pytest.raises(ValueError, match='Unknown canonical mode'):
            PoseEstimator(self._make_config(canonical='bad'))

    def test_invalid_representation_raises(self) -> None:
        with pytest.raises(NotImplementedError, match='Unknown pose representation'):
            PoseEstimator(self._make_config(rep='euler'))

    def test_cxcy_columns_are_half(self) -> None:
        # _append_cxcy sets cx=cy=0.5 for all views
        config = self._make_config()
        estimator = PoseEstimator(config)
        x = torch.randn(6, 32)
        out = estimator(x, v=3)
        # last two columns are cx, cy
        assert torch.allclose(out[:, 11], torch.full((6,), 0.5), atol=1e-5)
        assert torch.allclose(out[:, 12], torch.full((6,), 0.5), atol=1e-5)

    def test_per_view_focal_false_broadcasts_canonical(self) -> None:
        # default (checkpoint-matching): focal predicted from the canonical view
        # and broadcast, so every view shares the same fx
        config = self._make_config(per_view_focal=False)
        estimator = PoseEstimator(config)
        b, v = 2, 4
        x = torch.randn(b, v, 32)
        out = estimator(x, v)  # [b, v, 13]
        fx = out[..., 9]  # focal x is column 9 (after 6 rot + 3 trans)
        assert torch.allclose(fx, fx[:, :1].expand(-1, v), atol=1e-6)

    def test_per_view_focal_true_varies_per_view(self) -> None:
        # opt-in: focal head applied per view, so distinct view tokens give
        # distinct per-view focals
        config = self._make_config(per_view_focal=True)
        estimator = PoseEstimator(config)
        b, v = 2, 4
        x = torch.randn(b, v, 32)
        out = estimator(x, v)
        fx = out[..., 9]
        assert not torch.allclose(fx, fx[:, :1].expand(-1, v), atol=1e-4)


# ---------------------------------------------------------------------------
# TestGaussiansUpsampler
# ---------------------------------------------------------------------------


class TestGaussiansUpsampler:
    """Test the GaussiansUpsampler.to_gs method."""

    def _make_config(self, sh_degree: int = 0) -> dict:
        return {
            'model': {
                'gaussians': {'sh_degree': sh_degree},
                'hard_pixelalign': False,
                'scaling_bias': -2.3,
                'scaling_max': -1.2,
                'opacity_bias': -2.0,
            }
        }

    def _make_input(self, b: int, n: int, sh_degree: int) -> torch.Tensor:
        n_sh = (sh_degree + 1) ** 2 * 3
        # 3 xyz + n_sh features + 3 scaling + 4 rotation + 1 opacity
        d_out = 3 + n_sh + 3 + 4 + 1
        return torch.zeros(b, n, d_out)

    def test_returns_five_tensors(self) -> None:
        config = self._make_config(sh_degree=0)
        up = GaussiansUpsampler(config)
        inp = self._make_input(2, 50, sh_degree=0)
        result = up.to_gs(inp)
        assert len(result) == 5

    def test_xyz_shape(self) -> None:
        config = self._make_config(sh_degree=0)
        up = GaussiansUpsampler(config)
        inp = self._make_input(2, 50, sh_degree=0)
        xyz, *_ = up.to_gs(inp)
        assert xyz.shape == (2, 50, 3)

    def test_features_shape_sh0(self) -> None:
        config = self._make_config(sh_degree=0)
        up = GaussiansUpsampler(config)
        inp = self._make_input(2, 50, sh_degree=0)
        _, features, *_ = up.to_gs(inp)
        # sh_degree=0: (0+1)^2 = 1, features shape [B, N, 1, 3]
        assert features.shape == (2, 50, 1, 3)

    def test_features_shape_sh1(self) -> None:
        config = self._make_config(sh_degree=1)
        up = GaussiansUpsampler(config)
        inp = self._make_input(2, 50, sh_degree=1)
        _, features, *_ = up.to_gs(inp)
        # sh_degree=1: (1+1)^2 = 4, features shape [B, N, 4, 3]
        assert features.shape == (2, 50, 4, 3)

    def test_scaling_bounded_by_max(self) -> None:
        config = self._make_config(sh_degree=0)
        up = GaussiansUpsampler(config)
        # large raw scaling values should be clamped to scaling_max
        inp = self._make_input(2, 50, sh_degree=0)
        inp[:, :, 3 + 3:3 + 3 + 3] = 100.0  # raw scaling cols
        _, _, scaling, *_ = up.to_gs(inp)
        assert (scaling <= config['model']['scaling_max'] + 1e-5).all()

    def test_rotation_shape(self) -> None:
        config = self._make_config(sh_degree=0)
        up = GaussiansUpsampler(config)
        inp = self._make_input(2, 50, sh_degree=0)
        _, _, _, rotation, _ = up.to_gs(inp)
        assert rotation.shape == (2, 50, 4)

    def test_opacity_shape(self) -> None:
        config = self._make_config(sh_degree=0)
        up = GaussiansUpsampler(config)
        inp = self._make_input(2, 50, sh_degree=0)
        *_, opacity = up.to_gs(inp)
        assert opacity.shape == (2, 50, 1)


# ---------------------------------------------------------------------------
# TestGetPointRangeFunc
# ---------------------------------------------------------------------------


class TestGetPointRangeFunc:
    """Test the get_point_range_func factory."""

    def test_object_centric_depth_output_range(self) -> None:
        fn = get_point_range_func({'range_setting': {'type': 'object_centric_depth'}})
        t = torch.zeros(1)
        val = fn(t).item()
        # sigmoid(0)=0.5 → 2*0.5-1=0 → 0*1.5+2.7=2.7
        assert abs(val - 2.7) < 1e-5

    def test_linear_depth_output_in_range(self) -> None:
        near, far = 1.0, 100.0
        fn = get_point_range_func({'range_setting': {
            'type': 'linear_depth', 'near': near, 'far': far,
        }})
        t = torch.linspace(-5, 5, 20)
        out = fn(t)
        assert (out >= near - 1e-5).all()
        assert (out <= far + 1e-5).all()

    def test_log_depth_output_positive(self) -> None:
        fn = get_point_range_func({'range_setting': {'type': 'log_depth'}})
        t = torch.randn(20)
        out = fn(t)
        assert (out > 0).all()

    def test_disparity_output_in_range(self) -> None:
        near, far = 0.5, 50.0
        fn = get_point_range_func({'range_setting': {
            'type': 'disparity', 'near': near, 'far': far,
        }})
        t = torch.linspace(-5, 5, 20)
        out = fn(t)
        assert (out >= near - 1e-4).all()
        assert (out <= far + 1e-4).all()

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(NotImplementedError, match='Unknown range_setting type'):
            get_point_range_func({'range_setting': {'type': 'bad'}})

    def test_output_shape_preserved(self) -> None:
        fn = get_point_range_func({'range_setting': {'type': 'object_centric_depth'}})
        t = torch.randn(3, 4, 5)
        assert fn(t).shape == t.shape

    def test_default_type_is_object_centric(self) -> None:
        # passing empty gaussians_config uses default range_setting
        fn = get_point_range_func({})
        assert fn is not None


# ---------------------------------------------------------------------------
# TestLossComputer
# ---------------------------------------------------------------------------


class TestLossComputer:
    """Test the LossComputer module."""

    def _make_config(
        self, l2_weight: float = 1.0, gs_reg_weight: float = 0.0,
    ) -> dict:
        return {
            'training': {
                'l2_loss_weight': l2_weight,
                'gs_reg_loss_weight': gs_reg_weight,
                'perceptual_loss_weight': 0.0,
            }
        }

    def _make_images(self, b: int, v: int, h: int = 8, w: int = 8,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        rendering = torch.rand(b, v, 3, h, w)
        target = torch.rand(b, v, 3, h, w)
        return rendering, target

    def test_basic_loss_is_finite(self) -> None:
        lc = LossComputer(self._make_config())
        rendering, target = self._make_images(2, 3)
        result = lc(rendering, target, None, None)
        assert torch.isfinite(result.loss)

    def test_psnr_is_positive(self) -> None:
        lc = LossComputer(self._make_config())
        rendering, target = self._make_images(2, 3)
        result = lc(rendering, target, None, None)
        assert result.psnr.item() > 0.0

    def test_gs_reg_zero_when_no_xyz(self) -> None:
        lc = LossComputer(self._make_config(gs_reg_weight=1.0))
        rendering, target = self._make_images(2, 3)
        result = lc(rendering, target, None, None)
        assert result.gs_reg_loss.item() == 0.0

    def test_gs_reg_nonzero_when_xyz_provided(self) -> None:
        lc = LossComputer(self._make_config(gs_reg_weight=1.0))
        rendering, target = self._make_images(2, 3)
        xyz_norm = torch.randn(2, 100, 3)
        xyz_init = xyz_norm + 0.1
        result = lc(rendering, target, xyz_norm, xyz_init)
        assert result.gs_reg_loss.item() > 0.0

    def test_masked_loss_differs_from_unmasked(self) -> None:
        torch.manual_seed(0)
        lc = LossComputer(self._make_config())
        b, v, h, w = 2, 3, 8, 8
        rendering = torch.rand(b, v, 3, h, w)
        target = torch.rand(b, v, 3, h, w)
        # mask selects only half the pixels
        mask = torch.zeros(b, v, 1, h, w)
        mask[:, :, :, :h // 2, :] = 1.0
        result_unmasked = lc(rendering, target, None, None)
        result_masked = lc(rendering, target, None, None, pixel_mask=mask)
        assert not torch.allclose(result_unmasked.l2_loss, result_masked.l2_loss)

    def test_returns_expected_fields(self) -> None:
        lc = LossComputer(self._make_config())
        rendering, target = self._make_images(2, 3)
        result = lc(rendering, target, None, None)
        assert hasattr(result, 'loss')
        assert hasattr(result, 'l2_loss')
        assert hasattr(result, 'psnr')
        assert hasattr(result, 'gs_reg_loss')
        assert hasattr(result, 'perceptual_loss')

    def test_target_with_alpha_channel(self) -> None:
        # target with 4 channels (RGBA) should strip alpha
        lc = LossComputer(self._make_config())
        rendering = torch.rand(2, 3, 3, 8, 8)
        target_rgba = torch.rand(2, 3, 4, 8, 8)
        result = lc(rendering, target_rgba, None, None)
        assert torch.isfinite(result.loss)


# ---------------------------------------------------------------------------
# TestERayZer
# ---------------------------------------------------------------------------


class TestERayZer:
    """Test the ERayZer model class."""

    def test_construction_with_encoder(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        assert hasattr(model, 'transformer_encoder')
        assert hasattr(model, 'pose_predictor')
        assert hasattr(model, 'transformer_encoder_geom')
        assert hasattr(model, 'image_tokenizer')
        assert hasattr(model, 'renderer')

    def test_construction_without_encoder(self, config_erayzer) -> None:
        config = copy.deepcopy(config_erayzer)
        config['model']['transformer']['encoder_n_layer'] = 0
        model = ERayZer(config)
        assert not hasattr(model, 'transformer_encoder')
        assert not hasattr(model, 'pose_predictor')

    def test_odd_geom_layers_raises(self, config_erayzer) -> None:
        config = copy.deepcopy(config_erayzer)
        config['model']['transformer']['encoder_geom_n_layer'] = 3
        with pytest.raises(ValueError, match='encoder_geom_n_layer must be even'):
            ERayZer(config)

    def test_odd_encoder_layers_raises(self, config_erayzer) -> None:
        config = copy.deepcopy(config_erayzer)
        config['model']['transformer']['encoder_n_layer'] = 3
        with pytest.raises(ValueError, match='encoder_n_layer must be even'):
            ERayZer(config)

    def test_resolve_cameras_raises_without_encoder(self, config_erayzer) -> None:
        config = copy.deepcopy(config_erayzer)
        config['model']['transformer']['encoder_n_layer'] = 0
        model = ERayZer(config)
        with pytest.raises(RuntimeError, match='camera prediction modules are not initialized'):
            model._resolve_cameras(
                torch.randn(2, 16, 32), b=1, v_all=2, n=16, data={}, device=torch.device('cpu'),
            )

    def test_resolve_view_indices_from_data(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        data = {
            'image': torch.zeros(2, 3, 3, 32, 32),
            'input_indices': torch.tensor([[0, 1], [0, 1]]),
            'target_indices': torch.tensor([[2], [2]]),
        }
        in_idx, tgt_idx = model.resolve_view_indices(data, num_real_views=3, device='cpu')
        assert in_idx.tolist() == [[0, 1], [0, 1]]
        assert tgt_idx.tolist() == [[2], [2]]

    def test_resolve_view_indices_from_config_split(self, config_erayzer) -> None:
        # num_views=3, num_input_views=2, num_target_views=1, inference=False
        model = ERayZer(config_erayzer)
        model.random_index = False
        data = {'image': torch.zeros(2, 3, 3, 32, 32)}
        in_idx, tgt_idx = model.resolve_view_indices(data, num_real_views=3, device='cpu')
        assert in_idx.tolist() == [[0, 1], [0, 1]]
        assert tgt_idx.tolist() == [[2], [2]]

    def test_resolve_view_indices_inference_mode(self, config_erayzer) -> None:
        # in inference mode, all views are both input and target
        config = copy.deepcopy(config_erayzer)
        config['inference'] = True
        model = ERayZer(config)
        data = {'image': torch.zeros(2, 3, 3, 32, 32)}
        in_idx, tgt_idx = model.resolve_view_indices(data, num_real_views=3, device='cpu')
        assert in_idx.shape[1] <= 3
        assert tgt_idx.shape[1] == 3

    def test_maybe_randomize_single_view(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        torch.manual_seed(0)
        input_idx = torch.tensor([[0]])
        target_idx = torch.tensor([[1]])
        # run many times; with p=0.5 swap both orderings should appear
        got_swap = False
        got_no_swap = False
        for _ in range(30):
            in_i, tgt_i = model.maybe_randomize_view_indices(input_idx, target_idx, 'cpu')
            if in_i.tolist() == [[1]]:
                got_swap = True
            else:
                got_no_swap = True
            if got_swap and got_no_swap:
                break
        assert got_swap
        assert got_no_swap

    def test_add_spatial_pe_shape_preserved(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        b, v, hh, ww = 2, 3, 4, 4
        n = hh * ww
        d = config_erayzer['model']['transformer']['d']
        tokens = torch.randn(b * v, n, d)
        out = model.add_spatial_pe(tokens, b, v, hh, ww, model.pe_embedder)
        assert out.shape == (b * v, n, d)

    def test_add_spatial_pe_mismatched_tokens_raises(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        tokens = torch.randn(6, 15, 32)  # 15 != 4*4
        with pytest.raises(ValueError, match='Token count'):
            model.add_spatial_pe(tokens, b=2, v=3, h_tokens=4, w_tokens=4,
                                 embedder=model.pe_embedder)

    def test_compute_loss_returns_loss_and_log_list(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        b, v, h, w = 2, 3, 32, 32
        kwargs = {
            'render': torch.rand(b, v, 3, h, w),
            'target_image': torch.rand(b, v, 3, h, w),
        }
        loss, log_list = model.compute_loss('train', **kwargs)
        assert loss.ndim == 0
        assert isinstance(log_list, list)
        assert any(d['name'] == 'train_psnr' for d in log_list)

    def test_forward_output_keys(self, config_erayzer) -> None:
        """Forward pass with mocked renderer to avoid gsplat."""
        model = ERayZer(config_erayzer)
        model.eval()

        b, v, h, w = 1, 3, 32, 32
        data = {'image': torch.rand(b, v, 3, h, w)}

        def _fake_render(xyz, features, scaling, rotation, opacity,
                         height, width, C2W, fxfycxcy,
                         frustum_constraint=False, backgrounds=None):
            return SimpleNamespace(
                render=torch.zeros(C2W.shape[0], C2W.shape[1], 3, height, width),
                depth=torch.zeros(C2W.shape[0], C2W.shape[1], 1, height, width),
                alpha=torch.zeros(C2W.shape[0], C2W.shape[1], 1, height, width),
            )

        with patch.object(model.renderer, 'forward', side_effect=_fake_render):
            result = model(data)

        assert hasattr(result, 'render')
        assert hasattr(result, 'gaussians')
        assert hasattr(result, 'c2w_input')
        assert hasattr(result, 'fxfycxcy_input')
        assert hasattr(result, 'target_image')
        assert hasattr(result, 'input_indices')
        assert hasattr(result, 'target_indices')

    def test_forward_render_shape(self, config_erayzer) -> None:
        """render output has shape [B, n_target, 3, H, W]."""
        model = ERayZer(config_erayzer)
        model.eval()

        b, v, h, w = 1, 3, 32, 32
        n_target = config_erayzer['training']['num_target_views']
        data = {'image': torch.rand(b, v, 3, h, w)}

        def _fake_render(xyz, features, scaling, rotation, opacity,
                         height, width, C2W, fxfycxcy,
                         frustum_constraint=False, backgrounds=None):
            return SimpleNamespace(
                render=torch.zeros(C2W.shape[0], C2W.shape[1], 3, height, width),
                depth=torch.zeros(C2W.shape[0], C2W.shape[1], 1, height, width),
                alpha=torch.zeros(C2W.shape[0], C2W.shape[1], 1, height, width),
            )

        with patch.object(model.renderer, 'forward', side_effect=_fake_render):
            result = model(data)

        assert result.render.shape == (b, n_target, 3, h, w)

    def test_configure_optimizers_keys(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        opt_cfg = model.configure_optimizers()
        assert 'optimizer' in opt_cfg
        assert 'lr_scheduler' in opt_cfg

    def test_tokenize_images_default_matches_image_tokenizer(self, config_erayzer) -> None:
        # the default _tokenize_images hook rescales [0,1] → [-1,1] then tokenizes
        model = ERayZer(config_erayzer)
        h = w = config_erayzer['model']['image_tokenizer']['image_size']
        images = torch.rand(1, 2, 3, h, w)
        out = model._tokenize_images(images)
        expected = model.image_tokenizer(images * 2.0 - 1.0)
        assert torch.allclose(out, expected)

    def test_sample_background_default_is_none(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        assert model._sample_background(torch.device('cpu'), torch.float32) is None

    def test_prepare_target_default_is_identity(self, config_erayzer) -> None:
        model = ERayZer(config_erayzer)
        target = torch.rand(2, 1, 3, 8, 8)
        out, mask = model._prepare_target(
            target, {}, torch.zeros(2, 1, dtype=torch.long),
            torch.zeros(2, 1, dtype=torch.long), None,
        )
        assert out is target
        assert mask is None

    def test_forward_exposes_alpha_and_mask(self, config_erayzer) -> None:
        # base ERayZer forward now exposes render_alphas and a (None) pixel_mask
        model = ERayZer(config_erayzer)
        model.eval()
        b, v, h, w = 1, 3, 32, 32
        data = {'image': torch.rand(b, v, 3, h, w)}

        def _fake_render(xyz, features, scaling, rotation, opacity,
                         height, width, C2W, fxfycxcy,
                         frustum_constraint=False, backgrounds=None):
            return SimpleNamespace(
                render=torch.zeros(C2W.shape[0], C2W.shape[1], 3, height, width),
                depth=torch.zeros(C2W.shape[0], C2W.shape[1], 1, height, width),
                alpha=torch.zeros(C2W.shape[0], C2W.shape[1], 1, height, width),
            )

        with patch.object(model.renderer, 'forward', side_effect=_fake_render):
            result = model(data)

        assert hasattr(result, 'render_alphas')
        assert result.pixel_mask is None  # base ERayZer supervises without a mask


# ---------------------------------------------------------------------------
# TestERayZerIntegration
# ---------------------------------------------------------------------------


@requires_gsplat_cuda
class TestERayZerIntegration:
    """Integration tests that train and run inference on ERayZer.

    Skipped automatically when the gsplat CUDA extension is unavailable
    (e.g. CUDA toolkit not installed, GPU architecture not yet supported by
    available pre-compiled wheels).
    """

    def test_integration_basic(self, config_erayzer, run_erayzer_model_test) -> None:
        """Test ERayZer trains to completion and runs inference with L2 loss only."""
        run_erayzer_model_test(config=config_erayzer)

    def test_integration_gs_reg(self, config_erayzer, run_erayzer_model_test) -> None:
        """Test ERayZer trains with Gaussian splatting regularization loss enabled."""
        config = copy.deepcopy(config_erayzer)
        config['training']['gs_reg_loss_weight'] = 0.1
        run_erayzer_model_test(config=config)

    def test_integration_perceptual_loss(self, config_erayzer, run_erayzer_model_test) -> None:
        """Test ERayZer trains with VGG perceptual loss enabled."""
        config = copy.deepcopy(config_erayzer)
        config['training']['perceptual_loss_weight'] = 0.1
        run_erayzer_model_test(config=config)


# ---------------------------------------------------------------------------
# TestFilterInitStateDict
# ---------------------------------------------------------------------------


class TestFilterInitStateDict:
    """Test the _filter_init_state_dict checkpoint-extraction helper."""

    def test_extracts_from_state_dict_key(self) -> None:
        raw = {'state_dict': {'a': torch.zeros(1), 'b': torch.zeros(2)}}
        out = _filter_init_state_dict(raw)
        assert set(out) == {'a', 'b'}

    def test_extracts_from_model_key(self) -> None:
        raw = {'model': {'a': torch.zeros(1)}}
        out = _filter_init_state_dict(raw)
        assert set(out) == {'a'}

    def test_handles_bare_top_level_state_dict(self) -> None:
        raw = {'camera_token': torch.zeros(1), 'register_token': torch.zeros(2)}
        out = _filter_init_state_dict(raw)
        assert set(out) == {'camera_token', 'register_token'}

    def test_drops_loss_computer_keys(self) -> None:
        raw = {
            'camera_token': torch.zeros(1),
            'loss_computer.perceptual_loss_module.vgg.features.0.weight': torch.zeros(3),
            'loss_computer.anything': torch.zeros(1),
        }
        out = _filter_init_state_dict(raw)
        assert set(out) == {'camera_token'}

    def test_keeps_non_dropped_keys(self) -> None:
        raw = {'state_dict': {
            'transformer_encoder.0.norm1.weight': torch.zeros(4),
            'loss_computer.x': torch.zeros(1),
        }}
        out = _filter_init_state_dict(raw)
        assert set(out) == {'transformer_encoder.0.norm1.weight'}

    def test_custom_drop_prefix(self) -> None:
        raw = {'keep.a': torch.zeros(1), 'drop.b': torch.zeros(1)}
        out = _filter_init_state_dict(raw, drop_prefixes=('drop.',))
        assert set(out) == {'keep.a'}


# ---------------------------------------------------------------------------
# TestLoadInitCheckpoint
# ---------------------------------------------------------------------------


class TestResolveViewIndices:
    """Test ERayZer.resolve_view_indices view-sampling (no model build needed)."""

    def _standin(
        self, training: bool, training_cfg: dict, random_index: bool = False,
    ) -> SimpleNamespace:
        # resolve_view_indices only touches self.training, self.random_index,
        # and self.config — a stand-in exercises it without building the model
        return SimpleNamespace(
            training=training,
            random_index=random_index,
            config={'training': training_cfg, 'inference': False},
        )

    def _call(self, standin, num_real_views: int, batch_size: int = 2):
        data = {'image': torch.zeros(batch_size, num_real_views, 3, 8, 8)}
        return ERayZer.resolve_view_indices(
            standin, data, num_real_views, torch.device('cpu'),
        )

    def test_target_view_range_holds_out_exactly_n_targets(self) -> None:
        standin = self._standin(True, {'target_view_range': [1, 1], 'num_views': 6})
        for _ in range(30):
            in_idx, tgt_idx = self._call(standin, 6)
            assert tgt_idx.shape[1] == 1                 # exactly 1 target
            assert in_idx.shape[1] >= 1                  # at least 1 input
            assert in_idx.shape[1] + tgt_idx.shape[1] <= 6

    def test_total_view_count_varies_across_batches(self) -> None:
        standin = self._standin(True, {'target_view_range': [1, 1], 'num_views': 6})
        totals = {int(self._call(standin, 6)[0].shape[1] + self._call(standin, 6)[1].shape[1])
                  for _ in range(60)}
        # multi-view-style sampler must use a variable total, not a fixed count
        assert len(totals) >= 2

    def test_input_and_target_indices_are_disjoint(self) -> None:
        standin = self._standin(True, {'target_view_range': [1, 1], 'num_views': 6})
        for _ in range(30):
            in_idx, tgt_idx = self._call(standin, 6)
            a = set(in_idx[0].tolist())
            b = set(tgt_idx[0].tolist())
            assert a.isdisjoint(b)
            assert max(a | b) < 6

    def test_target_range_two_allows_two_targets(self) -> None:
        standin = self._standin(True, {'target_view_range': [1, 2], 'num_views': 6})
        seen = {int(self._call(standin, 6)[1].shape[1]) for _ in range(60)}
        assert seen <= {1, 2} and len(seen) >= 1

    def test_validation_uses_fixed_split_not_sampler(self) -> None:
        # at val time the sampler branch is skipped: deterministic fixed split
        standin = self._standin(
            False,
            {'target_view_range': [1, 1], 'num_views': 6,
             'num_input_views': 2, 'num_target_views': 4},
        )
        in_idx, tgt_idx = self._call(standin, 6)
        assert in_idx.shape[1] == 2 and tgt_idx.shape[1] == 4
        assert in_idx[0].tolist() == [0, 1]            # deterministic order
        assert tgt_idx[0].tolist() == [2, 3, 4, 5]

    def test_validation_random_split_alternates_views(self) -> None:
        # random_split (random_index=True) shuffles which cameras are input vs
        # target at val, while keeping the 2/4 counts
        standin = self._standin(
            False,
            {'target_view_range': [1, 1], 'num_views': 6,
             'num_input_views': 2, 'num_target_views': 4},
            random_index=True,
        )
        input_sets = set()
        for _ in range(40):
            in_idx, tgt_idx = self._call(standin, 6)
            assert in_idx.shape[1] == 2 and tgt_idx.shape[1] == 4
            assert set(in_idx[0].tolist()).isdisjoint(tgt_idx[0].tolist())
            input_sets.add(tuple(sorted(in_idx[0].tolist())))
        assert len(input_sets) >= 2  # the input/target assignment varies


class TestValidationVisuals:
    """Test ERayZer validation-visualization guards (no GPU/model build needed)."""

    def test_no_logger_is_noop(self) -> None:
        # logger=None -> no image-capable experiment -> returns without error
        standin = SimpleNamespace(logger=None)
        ERayZer._log_validation_visuals(cast(ERayZer, standin), {})

    def test_logger_without_add_image_is_noop(self) -> None:
        # an experiment lacking add_image is treated as not image-capable
        standin = SimpleNamespace(logger=SimpleNamespace(experiment=object()))
        ERayZer._log_validation_visuals(cast(ERayZer, standin), {})

    def test_forward_failure_is_swallowed(self) -> None:
        # a capable logger but a get_model_outputs that raises must not propagate
        class _Exp:
            def add_image(self, *a, **k) -> None:
                pass

        def _boom(_batch) -> dict:
            raise RuntimeError('forward exploded')

        standin = SimpleNamespace(
            logger=SimpleNamespace(experiment=_Exp()),
            get_model_outputs=_boom,
        )
        ERayZer._log_validation_visuals(cast(ERayZer, standin), {})


class TestConfigureOptimizers:
    """Test ERayZer.configure_optimizers LR scaling + schedule selection."""

    def _standin(self, opt_cfg: dict, train_cfg: dict) -> ERayZer:
        # a duck-typed stand-in exercises configure_optimizers without a full build;
        # cast so the unbound-method calls type-check against the real signature
        param = nn.Parameter(torch.zeros(2))
        return cast(ERayZer, SimpleNamespace(
            config={'optimizer': opt_cfg, 'training': train_cfg},
            parameters=lambda: iter([param]),
        ))

    def _train_cfg(self, **kw) -> dict:
        cfg = {'train_batch_size': 8, 'num_gpus': 1, 'num_nodes': 1, 'max_fwdbwd_passes': 1000}
        cfg.update(kw)
        return cfg

    def test_constant_schedule_is_lambdalr(self) -> None:
        from torch.optim.lr_scheduler import LambdaLR
        m = self._standin(
            {'lr': 4e-4, 'beta1': 0.9, 'beta2': 0.999, 'wd': 0.05, 'schedule': 'constant'},
            self._train_cfg(),
        )
        out = ERayZer.configure_optimizers(m)
        assert isinstance(out['lr_scheduler']['scheduler'], LambdaLR)

    def test_onecycle_schedule(self) -> None:
        from torch.optim.lr_scheduler import OneCycleLR
        m = self._standin(
            {'lr': 4e-4, 'beta1': 0.9, 'beta2': 0.95, 'wd': 0.05, 'warmup': 100,
             'schedule': 'onecycle', 'div_factor': 10.0, 'final_div_factor': 100.0},
            self._train_cfg(),
        )
        out = ERayZer.configure_optimizers(m)
        assert isinstance(out['lr_scheduler']['scheduler'], OneCycleLR)

    def test_unknown_schedule_raises(self) -> None:
        m = self._standin(
            {'lr': 4e-4, 'beta1': 0.9, 'beta2': 0.95, 'wd': 0.05, 'schedule': 'bogus'},
            self._train_cfg(),
        )
        with pytest.raises(ValueError, match='unknown optimizer.schedule'):
            ERayZer.configure_optimizers(m)

    def test_no_scaling_by_default(self) -> None:
        m = self._standin(
            {'lr': 4e-4, 'beta1': 0.9, 'beta2': 0.95, 'wd': 0.05, 'schedule': 'constant'},
            self._train_cfg(train_batch_size=8),
        )
        out = ERayZer.configure_optimizers(m)
        assert out['optimizer'].param_groups[0]['lr'] == pytest.approx(4e-4)

    def test_lr_batch_scaling_at_256_is_identity(self) -> None:
        # global_batch_size = 32 * 8 * 1 = 256 -> scaled lr == base lr
        m = self._standin(
            {'lr': 4e-4, 'beta1': 0.9, 'beta2': 0.999, 'wd': 0.05,
             'schedule': 'constant', 'scale_lr_by_batch': True},
            self._train_cfg(train_batch_size=32, num_gpus=8),
        )
        out = ERayZer.configure_optimizers(m)
        assert out['optimizer'].param_groups[0]['lr'] == pytest.approx(4e-4)

    def test_lr_batch_scaling_small_batch(self) -> None:
        # global_batch_size = 8 -> lr scaled down by 8/256
        m = self._standin(
            {'lr': 4e-4, 'beta1': 0.9, 'beta2': 0.999, 'wd': 0.05,
             'schedule': 'constant', 'scale_lr_by_batch': True},
            self._train_cfg(train_batch_size=8),
        )
        out = ERayZer.configure_optimizers(m)
        assert out['optimizer'].param_groups[0]['lr'] == pytest.approx(4e-4 * 8 / 256)


class TestParseHfCheckpointUrl:
    """Test the _parse_hf_checkpoint_url helper."""

    def test_blob_url(self) -> None:
        repo, rev, fn = _parse_hf_checkpoint_url(
            'https://huggingface.co/qitaoz/E-RayZer/blob/main/checkpoints/erayzer_multi.pt',
        )
        assert repo == 'qitaoz/E-RayZer'
        assert rev == 'main'
        assert fn == 'checkpoints/erayzer_multi.pt'

    def test_resolve_url_with_revision(self) -> None:
        repo, rev, fn = _parse_hf_checkpoint_url(
            'https://huggingface.co/org/repo/resolve/v1.0/sub/dir/model.pt',
        )
        assert repo == 'org/repo'
        assert rev == 'v1.0'
        assert fn == 'sub/dir/model.pt'

    def test_bare_path_defaults_to_main(self) -> None:
        repo, rev, fn = _parse_hf_checkpoint_url('https://huggingface.co/org/repo/model.pt')
        assert repo == 'org/repo'
        assert rev == 'main'
        assert fn == 'model.pt'


class TestResolveInitCheckpoint:
    """Test the _resolve_init_checkpoint spec resolver."""

    def test_local_path_passthrough(self, tmp_path) -> None:
        p = tmp_path / 'weights.bin'
        p.write_bytes(b'x')
        assert _resolve_init_checkpoint(str(p)) == Path(p)

    def test_nonexistent_local_path_still_returns_path(self) -> None:
        # resolution doesn't check existence (the loader does)
        assert _resolve_init_checkpoint('/no/such/file.pt') == Path('/no/such/file.pt')


class TestLoadInitCheckpoint:
    """Test ERayZer._load_init_checkpoint file handling and strict=False load."""

    def test_missing_file_raises(self, tmp_path) -> None:
        # the file-existence check runs before any use of self, so a stand-in
        # object exercises the error path without building the full model
        missing = tmp_path / 'nope.bin'
        with pytest.raises(FileNotFoundError, match='init_checkpoint not found'):
            ERayZer._load_init_checkpoint(cast(ERayZer, object()), str(missing))

    def test_loads_matching_keys_strict_false(self, tmp_path) -> None:
        # build a tiny module, save part of its state dict (plus a loss_computer
        # key and an unexpected key), and confirm the loader maps what it can
        module = nn.Sequential(nn.Linear(3, 4))
        ckpt = {
            'state_dict': {
                '0.weight': torch.ones(4, 3),
                '0.bias': torch.ones(4),
                'loss_computer.vgg.x': torch.zeros(2),  # must be dropped
                'unexpected.param': torch.zeros(1),     # tolerated by strict=False
            },
        }
        path = tmp_path / 'ckpt.bin'
        torch.save(ckpt, path)

        result = ERayZer._load_init_checkpoint(cast(ERayZer, module), str(path))

        linear = cast(nn.Linear, module[0])
        assert torch.allclose(linear.weight, torch.ones(4, 3))
        assert torch.allclose(linear.bias, torch.ones(4))
        # loss_computer.* dropped, so it is not reported as unexpected
        assert result.unexpected_keys == ['unexpected.param']
