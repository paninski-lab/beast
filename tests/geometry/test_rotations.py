"""Tests for beast.geometry.rotations utility functions."""

import torch

from beast.geometry.rotations import quat2mat, rot6d2mat


def _identity_6d() -> torch.Tensor:
    """6D representation of the identity rotation."""
    return torch.tensor([1., 0., 0., 0., 1., 0.])


def _quat_identity() -> torch.Tensor:
    """Unit quaternion (w,x,y,z) for the identity rotation."""
    return torch.tensor([1., 0., 0., 0.])


def _quat_rot_z_90() -> torch.Tensor:
    """Unit quaternion for 90° rotation around z-axis."""
    return torch.tensor([0.7071068, 0., 0., 0.7071068])


class TestRot6d2Mat:
    """Test the rot6d2mat function."""

    def test_identity_input(self) -> None:
        R = rot6d2mat(_identity_6d())
        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_output_shape_unbatched(self) -> None:
        assert rot6d2mat(_identity_6d()).shape == (3, 3)

    def test_output_shape_batched(self) -> None:
        inp = _identity_6d().unsqueeze(0).expand(5, -1)
        assert rot6d2mat(inp).shape == (5, 3, 3)

    def test_output_is_orthogonal(self) -> None:
        # random 6D input should still produce an orthogonal matrix
        torch.manual_seed(0)
        inp = torch.randn(8, 6)
        R = rot6d2mat(inp)
        RRt = torch.bmm(R, R.transpose(1, 2))
        assert torch.allclose(RRt, torch.eye(3).unsqueeze(0).expand(8, -1, -1), atol=1e-5)

    def test_determinant_is_positive_one(self) -> None:
        torch.manual_seed(1)
        inp = torch.randn(8, 6)
        R = rot6d2mat(inp)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(8), atol=1e-5)

    def test_non_orthogonal_input_is_projected(self) -> None:
        # even when the two input columns are not orthogonal, output must be orthogonal
        inp = torch.tensor([1., 1., 0., 1., 0., 0.])
        R = rot6d2mat(inp)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)

    def test_higher_batch_dims(self) -> None:
        inp = _identity_6d().reshape(1, 1, 6).expand(3, 4, -1)
        assert rot6d2mat(inp).shape == (3, 4, 3, 3)


class TestQuat2Mat:
    """Test the quat2mat function."""

    def test_identity_quaternion(self) -> None:
        R = quat2mat(_quat_identity())
        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_output_shape_unbatched(self) -> None:
        assert quat2mat(_quat_identity()).shape == (3, 3)

    def test_output_shape_batched(self) -> None:
        q = _quat_identity().unsqueeze(0).expand(5, -1)
        assert quat2mat(q).shape == (5, 3, 3)

    def test_rot_z_90(self) -> None:
        # 90° around z: x→y, y→-x, z→z
        R = quat2mat(_quat_rot_z_90())
        expected = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        assert torch.allclose(R, expected, atol=1e-5)

    def test_output_is_orthogonal(self) -> None:
        torch.manual_seed(2)
        q = torch.randn(8, 4)
        R = quat2mat(q)
        RRt = torch.bmm(R, R.transpose(1, 2))
        assert torch.allclose(RRt, torch.eye(3).unsqueeze(0).expand(8, -1, -1), atol=1e-5)

    def test_determinant_is_positive_one(self) -> None:
        torch.manual_seed(3)
        q = torch.randn(8, 4)
        R = quat2mat(q)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(8), atol=1e-5)

    def test_unnormalized_quaternion(self) -> None:
        # scaled quaternion should give the same rotation as the unit version
        q_unit = _quat_rot_z_90()
        q_scaled = q_unit * 3.7
        assert torch.allclose(quat2mat(q_unit), quat2mat(q_scaled), atol=1e-5)

    def test_antipodal_quaternions_give_same_rotation(self) -> None:
        q = _quat_rot_z_90()
        assert torch.allclose(quat2mat(q), quat2mat(-q), atol=1e-5)

    def test_higher_batch_dims(self) -> None:
        q = _quat_identity().reshape(1, 1, 4).expand(3, 4, -1)
        assert quat2mat(q).shape == (3, 4, 3, 3)
