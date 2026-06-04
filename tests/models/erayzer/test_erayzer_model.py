"""Tests for beast.models.erayzer.erayzer_model components."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from beast.models.erayzer.erayzer_model import (
    ERayZer,
    GaussiansUpsampler,
    LossComputer,
    PoseEstimator,
    build_transformer_blocks,
    get_cam_se3,
    get_point_range_func,
    sanitize,
)
from beast.nn.transformer import QK_Norm_TransformerBlock

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
                     rep: str = '6d') -> dict:
        return {
            'model': {
                'transformer': {'d': 32},
                'pose_latent': {
                    'canonical': canonical,
                    'mode': mode,
                    'representation': rep,
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
                         height, width, C2W, fxfycxcy):
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
                         height, width, C2W, fxfycxcy):
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
