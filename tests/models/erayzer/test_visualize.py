"""Tests for beast.models.erayzer.visualize."""

from types import SimpleNamespace

import pytest
import torch

from beast.models.erayzer.visualize import (
    camera_intrinsic_stats,
    export_gaussian_glb,
    make_camera_pose_image,
    make_render_grid,
    viz_is_due,
)

try:
    import trimesh  # noqa: F401  # pyright: ignore[reportMissingImports]
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False

requires_trimesh = pytest.mark.skipif(not _HAS_TRIMESH, reason='trimesh not installed')

# ---------------------------------------------------------------------------
# TestMakeRenderGrid
# ---------------------------------------------------------------------------


class TestMakeRenderGrid:
    """Test the make_render_grid 2-row GT-vs-render grid."""

    def test_returns_chw_float_in_unit_range(self) -> None:
        inp = torch.rand(1, 2, 3, 8, 8)
        tgt = torch.rand(1, 4, 3, 8, 8)
        ren = torch.rand(1, 4, 3, 8, 8)
        ren_in = torch.rand(1, 2, 3, 8, 8)
        grid = make_render_grid(inp, tgt, ren, render_input=ren_in)
        assert grid.ndim == 3
        assert grid.shape[0] == 3
        assert grid.dtype == torch.float32
        assert grid.min() >= 0.0 and grid.max() <= 1.0

    def test_is_two_rows_tall(self) -> None:
        # output must be exactly 2 image rows (GT, render) plus a thin separator
        h = 8
        inp = torch.rand(1, 2, 3, h, h)
        tgt = torch.rand(1, 3, 3, h, h)
        ren = torch.rand(1, 3, 3, h, h)
        ren_in = torch.rand(1, 2, 3, h, h)
        grid = make_render_grid(inp, tgt, ren, render_input=ren_in)
        assert 2 * h <= grid.shape[1] < 3 * h  # two rows, not three

    def test_render_input_adds_columns_not_rows(self) -> None:
        # input views add COLUMNS to the 2 rows, not extra rows
        h = 8
        inp = torch.rand(1, 2, 3, h, h)
        tgt = torch.rand(1, 3, 3, h, h)
        ren = torch.rand(1, 3, 3, h, h)
        ren_in = torch.rand(1, 2, 3, h, h)
        base = make_render_grid(inp, tgt, ren)                       # target only
        merged = make_render_grid(inp, tgt, ren, render_input=ren_in)  # input + target
        assert merged.shape[1] == base.shape[1]   # same height (still 2 rows)
        assert merged.shape[2] > base.shape[2]    # wider (input columns added)

    def test_accepts_unbatched_inputs(self) -> None:
        inp = torch.rand(2, 3, 8, 8)
        tgt = torch.rand(3, 3, 8, 8)
        ren = torch.rand(3, 3, 8, 8)
        ren_in = torch.rand(2, 3, 8, 8)
        grid = make_render_grid(inp, tgt, ren, render_input=ren_in)
        assert grid.shape[0] == 3

    def test_clamps_out_of_range_values(self) -> None:
        inp = torch.full((1, 1, 3, 8, 8), 5.0)
        tgt = torch.full((1, 1, 3, 8, 8), -5.0)
        ren = torch.zeros(1, 1, 3, 8, 8)
        grid = make_render_grid(inp, tgt, ren)
        assert grid.min() >= 0.0 and grid.max() <= 1.0


# ---------------------------------------------------------------------------
# TestExportGaussianGlb
# ---------------------------------------------------------------------------


class TestExportGaussianGlb:
    """Test the export_gaussian_glb point-cloud writer."""

    def _fake_gaussian(self, n: int = 16) -> SimpleNamespace:
        # mimics GaussianModel: get_xyz [N,3], get_features [N,K,3], get_opacity [N,1]
        return SimpleNamespace(
            get_xyz=torch.rand(n, 3),
            get_features=torch.rand(n, 1, 3),
            get_opacity=torch.rand(n, 1),
        )

    @requires_trimesh
    def test_writes_nonempty_glb(self, tmp_path) -> None:
        path = tmp_path / 'cloud.glb'
        n = export_gaussian_glb(self._fake_gaussian(16), path)
        assert path.is_file()
        assert path.stat().st_size > 0
        assert n == 16

    @requires_trimesh
    def test_opacity_threshold_filters_points(self, tmp_path) -> None:
        g = self._fake_gaussian(20)
        g.get_opacity = torch.cat([torch.ones(5, 1), torch.zeros(15, 1)])
        n = export_gaussian_glb(g, tmp_path / 'c.glb', opacity_threshold=0.5)
        assert n == 5

    @requires_trimesh
    def test_returns_zero_and_skips_when_empty_after_filter(self, tmp_path) -> None:
        g = self._fake_gaussian(8)
        g.get_opacity = torch.zeros(8, 1)
        path = tmp_path / 'empty.glb'
        n = export_gaussian_glb(g, path, opacity_threshold=0.5)
        assert n == 0
        assert not path.is_file()


# ---------------------------------------------------------------------------
# TestMakeCameraPoseImage
# ---------------------------------------------------------------------------


class TestMakeCameraPoseImage:
    """Test the make_camera_pose_image function (input + target frustums)."""

    def _identity_cameras(self, v: int, offset: float = 0.0) -> torch.Tensor:
        c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, v, 1, 1)
        # spread the centers so the plot has extent
        for i in range(v):
            c2w[0, i, :3, 3] = torch.tensor([float(i) + offset, 0.0, 0.0])
        return c2w

    def test_input_and_target_returns_chw_float(self) -> None:
        c2w_in = self._identity_cameras(2)
        c2w_tg = self._identity_cameras(4, offset=0.5)
        img = make_camera_pose_image(c2w_in, c2w_tg)
        assert img.ndim == 3
        assert img.shape[0] == 3
        assert img.dtype == torch.float32
        assert img.min() >= 0.0 and img.max() <= 1.0

    def test_input_only(self) -> None:
        img = make_camera_pose_image(self._identity_cameras(4))
        assert img.shape[0] == 3

    def test_accepts_unbatched_cameras(self) -> None:
        c2w_in = self._identity_cameras(3)[0]  # [V, 4, 4]
        c2w_tg = self._identity_cameras(2)[0]
        img = make_camera_pose_image(c2w_in, c2w_tg)
        assert img.shape[0] == 3


# ---------------------------------------------------------------------------
# TestVizIsDue
# ---------------------------------------------------------------------------


class TestVizIsDue:
    """Test the viz_is_due cadence helper."""

    def test_first_batch_on_due_epoch(self) -> None:
        assert viz_is_due(batch_idx=0, current_epoch=0, every_n_epochs=1) is True

    def test_non_first_batch_is_skipped(self) -> None:
        assert viz_is_due(batch_idx=2, current_epoch=0, every_n_epochs=1) is False

    def test_off_epoch_is_skipped(self) -> None:
        assert viz_is_due(batch_idx=0, current_epoch=1, every_n_epochs=2) is False

    def test_on_epoch_cadence(self) -> None:
        assert viz_is_due(batch_idx=0, current_epoch=4, every_n_epochs=2) is True

    def test_disabled_when_zero(self) -> None:
        assert viz_is_due(batch_idx=0, current_epoch=0, every_n_epochs=0) is False


# ---------------------------------------------------------------------------
# TestCameraIntrinsicStats
# ---------------------------------------------------------------------------


class TestCameraIntrinsicStats:
    """Test the camera_intrinsic_stats summary function."""

    def test_normalizes_by_image_size(self) -> None:
        # fxfycxcy in pixels at image_size=256: fx=fy=256, cx=cy=128
        fxfycxcy = torch.tensor([[256.0, 256.0, 128.0, 128.0]])
        stats = camera_intrinsic_stats(fxfycxcy, image_size=256)
        assert stats['focal_fx_mean'] == 1.0
        assert stats['focal_fy_mean'] == 1.0
        assert stats['cx_mean'] == 0.5
        assert stats['cy_mean'] == 0.5

    def test_reports_focal_spread(self) -> None:
        fxfycxcy = torch.tensor([
            [256.0, 256.0, 128.0, 128.0],
            [512.0, 512.0, 128.0, 128.0],
        ])
        stats = camera_intrinsic_stats(fxfycxcy, image_size=256)
        assert stats['focal_fx_std'] > 0.0

    def test_returns_plain_floats(self) -> None:
        fxfycxcy = torch.tensor([[256.0, 256.0, 128.0, 128.0]])
        stats = camera_intrinsic_stats(fxfycxcy, image_size=256)
        assert all(isinstance(v, float) for v in stats.values())
