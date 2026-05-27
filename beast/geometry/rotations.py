"""Rotation representation conversion utilities (6D, quaternion, SE3 matrix)."""

import torch
import torch.nn.functional as F


def rot6d2mat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks",
    CVPR 2019.

    Args:
        x: tensor of shape [B, 6] with 6D rotation representation.

    Returns:
        rotation matrices of shape [B, 3, 3].
    """
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    rotMat = torch.stack((b1, b2, b3), dim=-1)  # [B, 3, 3]
    return rotMat


def mat2rot6d(x: torch.Tensor) -> torch.Tensor:
    """Convert SE(3) matrices to 6D rotation + translation.

    Args:
        x: tensor of shape [B, 4, 4], batch of SE(3) matrices.

    Returns:
        tensor of shape [B, 9] = [6D rot, 3D trans].
    """
    if x.shape[-2:] != (4, 4):
        raise ValueError(f'Input must be of shape [B, 4, 4], got {x.shape}')

    trans = x[:, :3, 3]
    rot = x[:, :3, :3]
    rot6d = rot[:, :, :2].permute(0, 2, 1).reshape(rot.shape[0], 6)
    return torch.cat([rot6d, trans], dim=1)


def quat2mat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: tensor of shape [B, 4] with quaternion coefficients (w, x, y, z).

    Returns:
        rotation matrices of shape [B, 3, 3].
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2,
    ], dim=1).view(B, 3, 3)
    return rotMat


def mat2quat(x: torch.Tensor) -> torch.Tensor:
    """Convert SE(3) matrix to quaternion + translation.

    Args:
        x: tensor of shape [B, 4, 4], batch of SE(3) matrices.

    Returns:
        tensor of shape [B, 7] = [4D quaternion, 3D trans].
    """
    trans = x[:, :3, 3]
    rot = x[:, :3, :3]
    quat = mat2quat_transform(rot)
    return torch.cat([quat, trans], dim=1)


def mat2quat_transform(rotation_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 4D quaternion vector.

    Args:
        rotation_matrix: tensor of shape [B, 3, 3].
        eps: small value for numerical stability.

    Returns:
        quaternion tensor of shape [B, 4].

    Raises:
        TypeError: if input is not a torch.Tensor.
        ValueError: if input has wrong shape.
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(rotation_matrix)}')

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            f'Input size must be a three dimensional tensor. Got {rotation_matrix.shape}'
        )
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            f'Input size must be a N x 3 x 4  tensor. Got {rotation_matrix.shape}'
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([
        rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
        t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
        rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
    ], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([
        rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
        rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
        t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
    ], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([
        rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2,
    ], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([
        t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
        rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
        rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
    ], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3
    )
    q *= 0.5
    return q
