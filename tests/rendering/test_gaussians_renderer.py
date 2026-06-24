"""Tests for beast.rendering.gaussians_renderer utility functions."""

import numpy as np
import torch

from beast.rendering.gaussians_renderer import (
    RGB2SH,
    SH2RGB,
    build_rotation,
    build_scaling_rotation,
    camera_frustum_constraint,
    strip_lowerdiag,
    strip_symmetric,
)


class TestStripLowerdiag:
    """Test the strip_lowerdiag function."""

    def test_output_shape(self) -> None:
        L = torch.randn(5, 3, 3)
        result = strip_lowerdiag(L)
        assert result.shape == (5, 6)

    def test_extracts_correct_elements(self) -> None:
        L = torch.arange(9).float().reshape(1, 3, 3)
        result = strip_lowerdiag(L)
        expected = torch.tensor([[0., 1., 2., 4., 5., 8.]])
        assert torch.allclose(result, expected)

    def test_single_element_batch(self) -> None:
        L = torch.eye(3).unsqueeze(0)
        result = strip_lowerdiag(L)
        expected = torch.tensor([[1., 0., 0., 1., 0., 1.]])
        assert torch.allclose(result, expected)


class TestStripSymmetric:
    """Test the strip_symmetric function."""

    def test_delegates_to_strip_lowerdiag(self) -> None:
        sym = torch.randn(4, 3, 3)
        assert torch.allclose(strip_symmetric(sym), strip_lowerdiag(sym))

    def test_output_shape(self) -> None:
        sym = torch.randn(3, 3, 3)
        assert strip_symmetric(sym).shape == (3, 6)


class TestBuildRotation:
    """Test the build_rotation function."""

    def test_output_shape(self) -> None:
        quats = torch.randn(7, 4)
        R = build_rotation(quats)
        assert R.shape == (7, 3, 3)

    def test_identity_quaternion(self) -> None:
        quat = torch.tensor([[1., 0., 0., 0.]])
        R = build_rotation(quat)
        assert torch.allclose(R[0], torch.eye(3), atol=1e-6)

    def test_output_is_orthogonal(self) -> None:
        torch.manual_seed(42)
        quats = torch.randn(10, 4)
        R = build_rotation(quats)
        RRt = torch.bmm(R, R.transpose(1, 2))
        assert torch.allclose(RRt, torch.eye(3).unsqueeze(0).expand(10, -1, -1), atol=1e-5)

    def test_determinant_is_positive_one(self) -> None:
        torch.manual_seed(42)
        quats = torch.randn(10, 4)
        R = build_rotation(quats)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(10), atol=1e-5)

    def test_90_degree_z_rotation(self) -> None:
        quat = torch.tensor([[0.7071068, 0., 0., 0.7071068]])
        R = build_rotation(quat)
        expected = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        assert torch.allclose(R[0], expected, atol=1e-5)


class TestBuildScalingRotation:
    """Test the build_scaling_rotation function."""

    def test_output_shape(self) -> None:
        s = torch.ones(5, 3)
        r = torch.tensor([[1., 0., 0., 0.]]).expand(5, -1)
        result = build_scaling_rotation(s, r)
        assert result.shape == (5, 3, 3)

    def test_identity_rotation_uniform_scale(self) -> None:
        s = torch.tensor([[2., 2., 2.]])
        r = torch.tensor([[1., 0., 0., 0.]])
        result = build_scaling_rotation(s, r)
        expected = 2.0 * torch.eye(3)
        assert torch.allclose(result[0], expected, atol=1e-5)

    def test_identity_rotation_nonuniform_scale(self) -> None:
        s = torch.tensor([[1., 2., 3.]])
        r = torch.tensor([[1., 0., 0., 0.]])
        result = build_scaling_rotation(s, r)
        expected = torch.diag(torch.tensor([1., 2., 3.]))
        assert torch.allclose(result[0], expected, atol=1e-5)


class TestRGB2SH:
    """Test the RGB2SH function."""

    def test_midgray_maps_to_zero(self) -> None:
        rgb = torch.tensor([0.5, 0.5, 0.5])
        sh = RGB2SH(rgb)
        assert torch.allclose(sh, torch.zeros(3), atol=1e-7)

    def test_output_shape_preserved(self) -> None:
        rgb = torch.randn(8, 3)
        assert RGB2SH(rgb).shape == (8, 3)


class TestSH2RGB:
    """Test the SH2RGB function."""

    def test_zero_maps_to_midgray(self) -> None:
        sh = torch.tensor([0., 0., 0.])
        rgb = SH2RGB(sh)
        assert torch.allclose(rgb, torch.tensor([0.5, 0.5, 0.5]), atol=1e-7)

    def test_roundtrip_with_rgb2sh(self) -> None:
        torch.manual_seed(0)
        rgb = torch.rand(10, 3)
        assert torch.allclose(SH2RGB(RGB2SH(rgb)), rgb, atol=1e-6)

    def test_numpy_input(self) -> None:
        sh = np.zeros(3, dtype=np.float32)
        rgb = SH2RGB(sh)
        assert isinstance(rgb, np.ndarray)
        np.testing.assert_allclose(rgb, np.array([0.5, 0.5, 0.5]), atol=1e-7)

    def test_output_shape_preserved(self) -> None:
        sh = torch.randn(8, 3)
        assert SH2RGB(sh).shape == (8, 3)


class TestCameraFrustumConstraint:
    """Test the camera_frustum_constraint function (CPU-only, no rasterizer)."""

    @staticmethod
    def _identity_cam() -> tuple[torch.Tensor, torch.Tensor]:
        """A single camera at the origin looking down +z with a 64x64 image."""
        c2w = torch.eye(4).unsqueeze(0)  # [1, 4, 4]
        # fx = fy = 100, principal point at image centre (cx = cy = 32)
        fxfycxcy = torch.tensor([[100.0, 100.0, 32.0, 32.0]])
        return c2w, fxfycxcy

    def test_point_in_front_kept(self) -> None:
        c2w, fxfycxcy = self._identity_cam()
        xyz = torch.tensor([[0.0, 0.0, 2.0]])  # on-axis, in front
        opacity = torch.tensor([[0.8]])
        out = camera_frustum_constraint(xyz, opacity, c2w, fxfycxcy, 64, 64)
        assert torch.allclose(out, opacity)

    def test_point_behind_camera_zeroed(self) -> None:
        c2w, fxfycxcy = self._identity_cam()
        xyz = torch.tensor([[0.0, 0.0, -2.0]])  # behind the camera
        opacity = torch.tensor([[0.8]])
        out = camera_frustum_constraint(xyz, opacity, c2w, fxfycxcy, 64, 64)
        assert out.item() == 0.0

    def test_point_outside_fov_zeroed(self) -> None:
        c2w, fxfycxcy = self._identity_cam()
        xyz = torch.tensor([[10.0, 0.0, 2.0]])  # far off-axis, projects outside image
        opacity = torch.tensor([[0.8]])
        out = camera_frustum_constraint(xyz, opacity, c2w, fxfycxcy, 64, 64)
        assert out.item() == 0.0

    def test_intersection_of_two_cameras(self) -> None:
        # cam0 at origin sees (0,0,2); cam1 shifted far along +x does not.
        c2w0 = torch.eye(4)
        c2w1 = torch.eye(4)
        c2w1[0, 3] = 100.0  # translate camera centre by +100 in world x
        c2w = torch.stack([c2w0, c2w1])  # [2, 4, 4]
        fxfycxcy = torch.tensor([[100.0, 100.0, 32.0, 32.0]]).expand(2, -1)
        xyz = torch.tensor([[0.0, 0.0, 2.0]])
        opacity = torch.tensor([[0.8]])
        out = camera_frustum_constraint(xyz, opacity, c2w, fxfycxcy, 64, 64)
        assert out.item() == 0.0  # not seen by cam1 → excluded from the intersection

    def test_opacity_shape_preserved(self) -> None:
        c2w, fxfycxcy = self._identity_cam()
        xyz = torch.randn(50, 3)
        opacity = torch.rand(50, 1)
        out = camera_frustum_constraint(xyz, opacity, c2w, fxfycxcy, 64, 64)
        assert out.shape == opacity.shape
