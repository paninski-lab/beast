"""Rotation representation conversion utilities."""

import torch
import torch.nn.functional as F


def rot6d2mat(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert a 6D rotation representation to a 3×3 rotation matrix.

    Uses the Gram-Schmidt orthonormalization procedure from Zhou et al.,
    "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019).
    The first three values encode the first column and the second three encode
    the second column of the rotation matrix; the third column is their cross product.

    Parameters
    ----------
    rot_6d: rotation tensor of shape (..., 6).

    Returns
    -------
    rotation matrices of shape (..., 3, 3) with columns [a1, a2, a3].

    """
    a1 = F.normalize(rot_6d[..., :3], dim=-1)
    a2 = rot_6d[..., 3:]
    a2 = F.normalize(a2 - (a1 * a2).sum(dim=-1, keepdim=True) * a1, dim=-1)
    a3 = torch.linalg.cross(a1, a2)
    return torch.stack([a1, a2, a3], dim=-1)


def quat2mat(quat: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to 3×3 rotation matrices.

    Parameters
    ----------
    quat: quaternion tensor of shape (..., 4) in (w, x, y, z) order.
        Need not be pre-normalized.

    Returns
    -------
    rotation matrices of shape (..., 3, 3).

    """
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)
    mat = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return mat.reshape(*quat.shape[:-1], 3, 3)
