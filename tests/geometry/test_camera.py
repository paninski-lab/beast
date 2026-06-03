"""Tests for beast.geometry.camera utility functions."""

import math

import numpy as np
import pytest
import torch

from beast.geometry.camera import (
    cam_info_to_plucker,
    get_interpolated_k,
    get_interpolated_poses,
    get_interpolated_poses_many,
    get_ordered_poses_and_k,
    intrinsics_to_fxfycxcy,
    normalize_camera_sequence,
    quaternion_from_matrix,
    quaternion_matrix,
    quaternion_slerp,
    scale_intrinsics,
    unit_vector,
    w2c_to_c2w,
)


def _make_w2c(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build a 4×4 w2c from a 3×3 rotation and 3-vector translation."""
    m = torch.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def _rot_z(theta: float) -> torch.Tensor:
    """3×3 rotation matrix around z-axis by theta radians."""
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class TestW2cToC2w:
    """Test the w2c_to_c2w function."""

    def test_identity_maps_to_identity(self) -> None:
        w2c = torch.eye(4)
        assert torch.allclose(w2c_to_c2w(w2c), torch.eye(4), atol=1e-6)

    def test_pure_translation_inverted(self) -> None:
        # w2c with t=[1,2,3] and R=I → c2w with t=[-1,-2,-3]
        w2c = torch.eye(4)
        w2c[:3, 3] = torch.tensor([1., 2., 3.])
        c2w = w2c_to_c2w(w2c)
        assert torch.allclose(c2w[:3, 3], torch.tensor([-1., -2., -3.]), atol=1e-6)
        assert torch.allclose(c2w[:3, :3], torch.eye(3), atol=1e-6)

    def test_rotation_transposed(self) -> None:
        R = _rot_z(torch.pi / 4)
        w2c = _make_w2c(R, torch.zeros(3))
        c2w = w2c_to_c2w(w2c)
        assert torch.allclose(c2w[:3, :3], R.T, atol=1e-6)

    def test_round_trip(self) -> None:
        R = _rot_z(1.2)
        w2c = _make_w2c(R, torch.tensor([3., -1., 2.]))
        assert torch.allclose(w2c_to_c2w(w2c_to_c2w(w2c)), w2c, atol=1e-6)

    def test_bottom_row_preserved(self) -> None:
        w2c = _make_w2c(_rot_z(0.5), torch.tensor([1., 2., 3.]))
        c2w = w2c_to_c2w(w2c)
        assert torch.allclose(c2w[3], torch.tensor([0., 0., 0., 1.]), atol=1e-6)

    def test_output_rotation_is_orthogonal(self) -> None:
        w2c = _make_w2c(_rot_z(0.7), torch.tensor([1., 0., -2.]))
        R = w2c_to_c2w(w2c)[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)

    def test_batched_input(self) -> None:
        # [B, 4, 4] input
        B = 5
        w2c = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
        w2c[:, :3, 3] = torch.randn(B, 3)
        c2w = w2c_to_c2w(w2c)
        assert c2w.shape == (B, 4, 4)
        # each should round-trip
        assert torch.allclose(w2c_to_c2w(c2w), w2c, atol=1e-6)

    def test_higher_batch_dims(self) -> None:
        # [B, V, 4, 4] input
        w2c = torch.eye(4).reshape(1, 1, 4, 4).expand(3, 4, -1, -1)
        c2w = w2c_to_c2w(w2c)
        assert c2w.shape == (3, 4, 4, 4)


class TestIntrinsicsToFxfycxcy:
    """Test the intrinsics_to_fxfycxcy function."""

    def test_known_values(self) -> None:
        K = torch.tensor([[500., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
        out = intrinsics_to_fxfycxcy(K)
        assert torch.allclose(out, torch.tensor([500., 400., 320., 240.]))

    def test_output_shape_unbatched(self) -> None:
        K = torch.eye(3)
        assert intrinsics_to_fxfycxcy(K).shape == (4,)

    def test_output_shape_batched(self) -> None:
        K = torch.eye(3).unsqueeze(0).expand(7, -1, -1)
        assert intrinsics_to_fxfycxcy(K).shape == (7, 4)

    def test_batched_values(self) -> None:
        fx, fy, cx, cy = 800., 600., 400., 300.
        K = torch.tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        K_batch = K.unsqueeze(0).expand(3, -1, -1)
        out = intrinsics_to_fxfycxcy(K_batch)
        expected = torch.tensor([fx, fy, cx, cy]).unsqueeze(0).expand(3, -1)
        assert torch.allclose(out, expected)


class TestScaleIntrinsics:
    """Test the scale_intrinsics function."""

    def test_unit_scale_unchanged(self) -> None:
        K = torch.tensor([[500., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
        assert torch.allclose(scale_intrinsics(K, 1.0, 1.0), K)

    def test_uniform_scale(self) -> None:
        K = torch.tensor([[500., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
        K2 = scale_intrinsics(K, 2.0, 2.0)
        assert torch.allclose(K2[0, 0], torch.tensor(1000.))
        assert torch.allclose(K2[1, 1], torch.tensor(800.))
        assert torch.allclose(K2[0, 2], torch.tensor(640.))
        assert torch.allclose(K2[1, 2], torch.tensor(480.))

    def test_anisotropic_scale(self) -> None:
        K = torch.tensor([[400., 0., 200.], [0., 400., 200.], [0., 0., 1.]])
        K2 = scale_intrinsics(K, 0.5, 2.0)
        # fx, cx scaled by 0.5
        assert torch.allclose(K2[0, 0], torch.tensor(200.))
        assert torch.allclose(K2[0, 2], torch.tensor(100.))
        # fy, cy scaled by 2.0
        assert torch.allclose(K2[1, 1], torch.tensor(800.))
        assert torch.allclose(K2[1, 2], torch.tensor(400.))

    def test_does_not_modify_input(self) -> None:
        K = torch.tensor([[500., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
        K_orig = K.clone()
        scale_intrinsics(K, 2.0, 2.0)
        assert torch.allclose(K, K_orig)

    def test_skew_and_bottom_row_untouched(self) -> None:
        K = torch.tensor([[500., 1., 320.], [0., 400., 240.], [0., 0., 1.]])
        K2 = scale_intrinsics(K, 2.0, 2.0)
        assert K2[0, 1].item() == 1.0  # skew unchanged
        assert torch.allclose(K2[2], torch.tensor([0., 0., 1.]))

    def test_batched_input(self) -> None:
        K = torch.tensor([[500., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
        K_batch = K.unsqueeze(0).expand(4, -1, -1).clone()
        K2 = scale_intrinsics(K_batch, 0.5, 0.5)
        assert K2.shape == (4, 3, 3)
        assert torch.allclose(K2[:, 0, 0], torch.full((4,), 250.))


class TestNormalizeCameraSequence:
    """Test the normalize_camera_sequence function."""

    def _simple_w2c(self, t: list[float]) -> torch.Tensor:
        """w2c with identity rotation and given translation."""
        m = torch.eye(4)
        m[:3, 3] = torch.tensor(t)
        return m

    def test_camera0_at_origin(self) -> None:
        # after normalization camera 0 should be at the world origin
        w2c = torch.stack([
            self._simple_w2c([0., 0., 5.]),
            self._simple_w2c([0., 0., 10.]),
            self._simple_w2c([0., 0., 15.]),
        ])
        c2w = normalize_camera_sequence(w2c)
        assert torch.allclose(c2w[0, :3, 3], torch.zeros(3), atol=1e-6)

    def test_mean_distance_is_one(self) -> None:
        w2c = torch.stack([
            self._simple_w2c([0., 0., 0.]),
            self._simple_w2c([0., 0., 10.]),
            self._simple_w2c([0., 0., 20.]),
        ])
        c2w = normalize_camera_sequence(w2c)
        mean_dist = c2w[:, :3, 3].norm(dim=-1).mean()
        assert torch.allclose(mean_dist, torch.tensor(1.0), atol=1e-5)

    def test_output_is_valid_se3(self) -> None:
        w2c = torch.stack([
            _make_w2c(_rot_z(0.0), torch.tensor([0., 0., 5.])),
            _make_w2c(_rot_z(1.0), torch.tensor([3., 0., 5.])),
            _make_w2c(_rot_z(2.0), torch.tensor([-3., 0., 5.])),
        ])
        c2w = normalize_camera_sequence(w2c)
        # bottom row
        expected_bottom = torch.tensor([0., 0., 0., 1.]).expand(3, -1)
        assert torch.allclose(c2w[:, 3, :], expected_bottom, atol=1e-6)
        # orthogonal rotation blocks
        R = c2w[:, :3, :3]
        RRt = torch.bmm(R, R.transpose(1, 2))
        assert torch.allclose(RRt, torch.eye(3).unsqueeze(0).expand(3, -1, -1), atol=1e-5)

    def test_single_camera_is_identity(self) -> None:
        # one camera at origin → c2w should be identity
        w2c = self._simple_w2c([0., 0., 0.]).unsqueeze(0)
        c2w = normalize_camera_sequence(w2c)
        assert torch.allclose(c2w[0], torch.eye(4), atol=1e-6)

    def test_output_shape(self) -> None:
        w2c = torch.eye(4).unsqueeze(0).expand(6, -1, -1)
        assert normalize_camera_sequence(w2c).shape == (6, 4, 4)


# ---------------------------------------------------------------------------
# numpy-based camera utilities
# ---------------------------------------------------------------------------


def _rot_z_np(theta: float) -> np.ndarray:
    """3×3 rotation matrix around z-axis by theta radians (numpy)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _pose_np(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 pose matrix from a 3×3 rotation and 3-vector translation."""
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


class TestUnitVector:
    """Test the unit_vector function."""

    def test_1d_unit_length(self) -> None:
        v = np.array([3., 4., 0.])
        uv = unit_vector(v)
        assert abs(np.linalg.norm(uv) - 1.0) < 1e-10

    def test_1d_known_values(self) -> None:
        v = np.array([1., 0., 0.])
        np.testing.assert_allclose(unit_vector(v), v, atol=1e-10)

    def test_2d_axis0(self) -> None:
        v = np.array([[3., 1.], [4., 0.]])
        uv = unit_vector(v, axis=0)
        assert uv.shape == (2, 2)
        norms = np.linalg.norm(uv, axis=0)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-10)

    def test_2d_axis1(self) -> None:
        v = np.array([[3., 4.], [0., 5.]])
        uv = unit_vector(v, axis=1)
        # each row should have norm 1 (or 0 if zero row)
        np.testing.assert_allclose(np.linalg.norm(uv[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(uv[1]), 1.0, atol=1e-10)

    def test_does_not_modify_original(self) -> None:
        v = np.array([3., 4., 0.])
        v_orig = v.copy()
        unit_vector(v)
        np.testing.assert_array_equal(v, v_orig)


class TestQuaternionFromMatrix:
    """Test the quaternion_from_matrix function."""

    def test_identity_matrix(self) -> None:
        q = quaternion_from_matrix(np.eye(3))
        # identity → (1, 0, 0, 0); sign may flip but |q|=1
        np.testing.assert_allclose(abs(q[0]), 1.0, atol=1e-6)
        np.testing.assert_allclose(q[1:], [0., 0., 0.], atol=1e-6)

    def test_output_is_unit_quaternion(self) -> None:
        R = _rot_z_np(1.0)
        q = quaternion_from_matrix(R)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-9)

    def test_nonnegative_scalar_part(self) -> None:
        # function normalises so q[0] >= 0
        q = quaternion_from_matrix(_rot_z_np(math.pi))
        assert q[0] >= 0.0

    def test_isprecise_matches_default(self) -> None:
        R = _pose_np(_rot_z_np(0.7), np.zeros(3))  # 4×4 required for isprecise=True
        q_default = quaternion_from_matrix(R)
        q_precise = quaternion_from_matrix(R, isprecise=True)
        # same rotation, so either q or -q; compare absolute values
        np.testing.assert_allclose(np.abs(q_default), np.abs(q_precise), atol=1e-6)

    def test_4x4_input_accepted(self) -> None:
        M = np.eye(4)
        q = quaternion_from_matrix(M)
        assert q.shape == (4,)


class TestQuaternionSlerp:
    """Test the quaternion_slerp function."""

    def _quat_z(self, theta: float) -> np.ndarray:
        """Quaternion for rotation around z by theta."""
        return np.array([math.cos(theta / 2), 0., 0., math.sin(theta / 2)])

    def test_fraction_zero_returns_q0(self) -> None:
        q0 = self._quat_z(0.0)
        q1 = self._quat_z(1.0)
        np.testing.assert_allclose(quaternion_slerp(q0, q1, 0.0), q0, atol=1e-9)

    def test_fraction_one_returns_q1(self) -> None:
        q0 = self._quat_z(0.0)
        q1 = self._quat_z(1.0)
        np.testing.assert_allclose(quaternion_slerp(q0, q1, 1.0), q1, atol=1e-9)

    def test_midpoint_is_unit_quaternion(self) -> None:
        q0 = self._quat_z(0.0)
        q1 = self._quat_z(1.0)
        qm = quaternion_slerp(q0, q1, 0.5)
        np.testing.assert_allclose(np.linalg.norm(qm), 1.0, atol=1e-9)

    def test_midpoint_angle(self) -> None:
        # midpoint between 0° and 90° rotation should be 45°
        q0 = self._quat_z(0.0)
        q1 = self._quat_z(math.pi / 2)
        qm = quaternion_slerp(q0, q1, 0.5)
        expected = self._quat_z(math.pi / 4)
        np.testing.assert_allclose(np.abs(qm), np.abs(expected), atol=1e-6)

    def test_same_quaternions_returns_input(self) -> None:
        q = self._quat_z(0.3)
        np.testing.assert_allclose(quaternion_slerp(q, q, 0.5), q, atol=1e-9)


class TestQuaternionMatrix:
    """Test the quaternion_matrix function."""

    def test_identity_quaternion(self) -> None:
        M = quaternion_matrix([1., 0., 0., 0.])
        np.testing.assert_allclose(M[:3, :3], np.eye(3), atol=1e-9)

    def test_output_shape(self) -> None:
        M = quaternion_matrix([1., 0., 0., 0.])
        assert M.shape == (4, 4)

    def test_bottom_row_and_column(self) -> None:
        M = quaternion_matrix([1., 0., 0., 0.])
        np.testing.assert_allclose(M[3], [0., 0., 0., 1.], atol=1e-9)
        np.testing.assert_allclose(M[:, 3], [0., 0., 0., 1.], atol=1e-9)

    def test_rotation_block_is_orthogonal(self) -> None:
        q = np.array([math.cos(0.3), 0., 0., math.sin(0.3)])
        M = quaternion_matrix(q)
        R = M[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_known_rot_z_90(self) -> None:
        # 90° around z: (w=cos45°, x=0, y=0, z=sin45°)
        half = math.pi / 4
        q = [math.cos(half), 0., 0., math.sin(half)]
        M = quaternion_matrix(q)
        expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(M[:3, :3], expected, atol=1e-7)

    def test_near_zero_quaternion_returns_identity(self) -> None:
        M = quaternion_matrix([0., 0., 0., 0.])
        np.testing.assert_allclose(M, np.eye(4), atol=1e-9)


class TestGetInterpolatedPoses:
    """Test the get_interpolated_poses function."""

    def _pose3(self, t: list[float]) -> np.ndarray:
        """4×4 pose with identity rotation and given translation."""
        m = np.eye(4)
        m[:3, 3] = t
        return m

    def test_output_length(self) -> None:
        a = self._pose3([0., 0., 0.])
        b = self._pose3([1., 0., 0.])
        out = get_interpolated_poses(a, b, steps=7)
        assert len(out) == 7

    def test_each_pose_is_3x4(self) -> None:
        a = self._pose3([0., 0., 0.])
        b = self._pose3([1., 0., 0.])
        for pose in get_interpolated_poses(a, b, steps=5):
            assert np.array(pose).shape == (3, 4)

    def test_endpoints_match(self) -> None:
        a = self._pose3([0., 0., 0.])
        b = self._pose3([3., 0., 0.])
        out = get_interpolated_poses(a, b, steps=5)
        np.testing.assert_allclose(out[0][:, 3], a[:3, 3], atol=1e-6)
        np.testing.assert_allclose(out[-1][:, 3], b[:3, 3], atol=1e-6)

    def test_default_steps(self) -> None:
        a = self._pose3([0., 0., 0.])
        b = self._pose3([1., 0., 0.])
        assert len(get_interpolated_poses(a, b)) == 10


class TestGetInterpolatedK:
    """Test the get_interpolated_k function."""

    def test_output_length(self) -> None:
        K = torch.eye(3)
        out = get_interpolated_k(K, K * 2, steps=6)
        assert len(out) == 6

    def test_endpoints_match(self) -> None:
        k_a = torch.tensor([[400., 0., 200.], [0., 400., 200.], [0., 0., 1.]])
        k_b = torch.tensor([[800., 0., 400.], [0., 800., 400.], [0., 0., 1.]])
        out = get_interpolated_k(k_a, k_b, steps=5)
        assert torch.allclose(out[0], k_a, atol=1e-6)
        assert torch.allclose(out[-1], k_b, atol=1e-6)

    def test_midpoint_is_average(self) -> None:
        k_a = torch.tensor([[400., 0., 200.], [0., 400., 200.], [0., 0., 1.]])
        k_b = torch.tensor([[800., 0., 400.], [0., 800., 400.], [0., 0., 1.]])
        out = get_interpolated_k(k_a, k_b, steps=3)
        expected_mid = (k_a + k_b) / 2
        assert torch.allclose(out[1], expected_mid, atol=1e-6)

    def test_default_steps(self) -> None:
        K = torch.eye(3)
        assert len(get_interpolated_k(K, K)) == 10


class TestGetOrderedPosesAndK:
    """Test the get_ordered_poses_and_k function."""

    def _make_poses(self, translations: list[list[float]]) -> torch.Tensor:
        """Build [N, 3, 4] poses with identity rotations."""
        poses = []
        for t in translations:
            p = torch.zeros(3, 4)
            p[:3, :3] = torch.eye(3)
            p[:, 3] = torch.tensor(t)
            poses.append(p)
        return torch.stack(poses)

    def test_output_shapes_preserved(self) -> None:
        poses = self._make_poses([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        Ks = torch.eye(3).unsqueeze(0).expand(3, -1, -1).clone()
        ordered_poses, ordered_Ks = get_ordered_poses_and_k(poses, Ks)
        assert ordered_poses.shape == poses.shape
        assert ordered_Ks.shape == Ks.shape

    def test_first_pose_unchanged(self) -> None:
        # greedy nearest-neighbor always starts from poses[0]
        poses = self._make_poses([[0., 0., 0.], [10., 0., 0.], [1., 0., 0.]])
        Ks = torch.eye(3).unsqueeze(0).expand(3, -1, -1).clone()
        ordered_poses, _ = get_ordered_poses_and_k(poses, Ks)
        assert torch.allclose(ordered_poses[0], poses[0])

    def test_nearest_neighbor_ordering(self) -> None:
        # poses[0]→[1]→[3]→[2] by distance; [1] is at x=1, [3] at x=2, [2] at x=10
        translations = [[0., 0., 0.], [1., 0., 0.], [10., 0., 0.], [2., 0., 0.]]
        poses = self._make_poses(translations)
        Ks = torch.eye(3).unsqueeze(0).expand(4, -1, -1).clone()
        ordered_poses, _ = get_ordered_poses_and_k(poses, Ks)
        # second should be the x=1 pose (nearest to x=0)
        assert torch.allclose(ordered_poses[1, :, 3], torch.tensor([1., 0., 0.]))


class TestGetInterpolatedPosesMany:
    """Test the get_interpolated_poses_many function."""

    def _make_poses(self, translations: list[list[float]]) -> torch.Tensor:
        """Build [N, 3, 4] poses with identity rotations."""
        poses = []
        for t in translations:
            p = torch.zeros(3, 4)
            p[:3, :3] = torch.eye(3)
            p[:, 3] = torch.tensor(t)
            poses.append(p)
        return torch.stack(poses)

    def test_empty_poses_raises(self) -> None:
        poses = torch.zeros(0, 3, 4)
        Ks = torch.zeros(0, 3, 3)
        with pytest.raises(ValueError):
            get_interpolated_poses_many(poses, Ks)

    def test_single_pose_returns_one_step(self) -> None:
        poses = self._make_poses([[0., 0., 0.]])
        Ks = torch.eye(3).unsqueeze(0)
        out_poses, out_Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=5)
        assert out_poses.shape[0] == 1
        assert out_Ks.shape[0] == 1

    def test_two_poses_output_length(self) -> None:
        poses = self._make_poses([[0., 0., 0.], [1., 0., 0.]])
        Ks = torch.eye(3).unsqueeze(0).expand(2, -1, -1).clone()
        out_poses, out_Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=4)
        # one transition → 4 steps
        assert out_poses.shape[0] == 4
        assert out_Ks.shape[0] == 4

    def test_output_tensor_types(self) -> None:
        poses = self._make_poses([[0., 0., 0.], [1., 0., 0.]])
        Ks = torch.eye(3).unsqueeze(0).expand(2, -1, -1).clone()
        out_poses, out_Ks = get_interpolated_poses_many(poses, Ks)
        assert isinstance(out_poses, torch.Tensor)
        assert isinstance(out_Ks, torch.Tensor)

    def test_output_pose_shape(self) -> None:
        poses = self._make_poses([[0., 0., 0.], [1., 0., 0.]])
        Ks = torch.eye(3).unsqueeze(0).expand(2, -1, -1).clone()
        out_poses, _ = get_interpolated_poses_many(poses, Ks, steps_per_transition=3)
        # each pose is 3×4
        assert out_poses.shape[1:] == (3, 4)


class TestCamInfoToPlucker:
    """Test the cam_info_to_plucker function."""

    def _identity_cam(self, b: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Identity c2w and simple fxfycxcy for b cameras at 32×32."""
        c2w = torch.eye(4).unsqueeze(0).expand(b, -1, -1).clone()
        fxfycxcy = torch.tensor([0.5, 0.5, 0.5, 0.5]).unsqueeze(0).expand(b, -1).clone()
        return c2w, fxfycxcy

    def test_output_shape_single_batch(self) -> None:
        c2w, fxfycxcy = self._identity_cam(b=2)
        imgs_info = {'height': 8, 'width': 8}
        out = cam_info_to_plucker(c2w, fxfycxcy, imgs_info, normalized=True)
        assert out.shape == (2, 6, 8, 8)

    def test_output_shape_bv_input(self) -> None:
        # [b, v, 4, 4] input → [b*v, 6, h, w]
        b, v = 2, 3
        c2w = torch.eye(4).reshape(1, 1, 4, 4).expand(b, v, -1, -1).clone()
        fxfycxcy = torch.tensor([0.5, 0.5, 0.5, 0.5]).reshape(1, 1, 4).expand(b, v, -1).clone()
        imgs_info = {'height': 4, 'width': 4}
        out = cam_info_to_plucker(c2w, fxfycxcy, imgs_info, normalized=True)
        assert out.shape == (b * v, 6, 4, 4)

    def test_direction_vectors_are_unit(self) -> None:
        c2w, fxfycxcy = self._identity_cam(b=1)
        imgs_info = {'height': 8, 'width': 8}
        out = cam_info_to_plucker(c2w, fxfycxcy, imgs_info, normalized=True)
        # last 3 channels are direction; each pixel should be unit
        directions = out[0, 3:]  # [3, h, w]
        norms = directions.norm(dim=0)  # [h, w]
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_return_moment_false_gives_origin_direction(self) -> None:
        c2w, fxfycxcy = self._identity_cam(b=1)
        imgs_info = {'height': 4, 'width': 4}
        out = cam_info_to_plucker(c2w, fxfycxcy, imgs_info, normalized=True, return_moment=False)
        # origin channels: identity camera at origin → all zeros
        origins = out[0, :3]  # [3, h, w]
        assert torch.allclose(origins, torch.zeros_like(origins), atol=1e-6)

    def test_normalized_false_uses_pixel_intrinsics(self) -> None:
        # when normalized=False, fxfycxcy is already in pixels
        b = 1
        h, w = 8, 8
        c2w = torch.eye(4).unsqueeze(0)
        # pixel-space intrinsics: fx=fy=4, cx=cy=4
        fxfycxcy = torch.tensor([[4., 4., 4., 4.]])
        imgs_info = {'height': h, 'width': w}
        out = cam_info_to_plucker(c2w, fxfycxcy, imgs_info, normalized=False)
        assert out.shape == (b, 6, h, w)
