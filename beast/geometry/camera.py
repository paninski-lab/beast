"""Camera transformation helpers, Plucker ray encoding, and pose utilities.

Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors.
Licensed under the Apache License, Version 2.0.

"""

import math

import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

_EPS = np.finfo(float).eps * 4.0


def unit_vector(data: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Return ndarray normalized by Euclidean norm along axis.

    Parameters
    ----------
    data: input array.
    axis: the axis along which to normalize into unit vector.

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


def quaternion_from_matrix(matrix: np.ndarray, isprecise: bool = False) -> np.ndarray:
    """Return quaternion from rotation matrix.

    Parameters
    ----------
    matrix: rotation matrix to obtain quaternion.
    isprecise: if True, input matrix is assumed to be a precise rotation matrix
        and a faster algorithm is used.

    """
    M = np.asarray(matrix, dtype=np.float64)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        K = [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
        K = np.array(K)
        K /= 3.0
        w, V = np.linalg.eigh(K)
        q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_slerp(
    quat0: np.ndarray,
    quat1: np.ndarray,
    fraction: float,
    spin: int = 0,
    shortestpath: bool = True,
) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.

    Parameters
    ----------
    quat0: first quaternion.
    quat1: second quaternion.
    fraction: interpolation parameter (0 → quat0, 1 → quat1).
    spin: additional spin to place on the interpolation.
    shortestpath: whether to return the short or long path to rotation.

    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError('Input quaternions invalid.')
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quaternion_matrix(quaternion: npt.ArrayLike) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.

    Parameters
    ----------
    quaternion: value to convert to matrix.

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_interpolated_poses(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    steps: int = 10,
) -> list[np.ndarray]:
    """Return interpolation of poses with the specified number of steps.

    Parameters
    ----------
    pose_a: first pose.
    pose_b: second pose.
    steps: number of steps the interpolated pose path should contain.

    """
    quat_a = quaternion_from_matrix(pose_a[:3, :3])
    quat_b = quaternion_from_matrix(pose_b[:3, :3])

    ts = np.linspace(0, 1, steps)
    quats = [quaternion_slerp(quat_a, quat_b, t) for t in ts]
    trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

    poses_ab = []
    for quat, tran in zip(quats, trans, strict=True):
        pose = np.identity(4)
        pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
        pose[:3, 3] = tran
        poses_ab.append(pose[:3])
    return poses_ab


def get_interpolated_k(
    k_a: Float[Tensor, '3 3'],
    k_b: Float[Tensor, '3 3'],
    steps: int = 10,
) -> list[Float[Tensor, '3 4']]:
    """Return interpolated path between two camera intrinsic matrices.

    Parameters
    ----------
    k_a: camera matrix 1.
    k_b: camera matrix 2.
    steps: number of steps the interpolated path should contain.

    Returns
    -------
    list of interpolated camera matrices.

    """
    Ks: list[Float[Tensor, '3 3']] = []
    ts = np.linspace(0, 1, steps)
    for t in ts:
        new_k = k_a * (1.0 - t) + k_b * t
        Ks.append(new_k)
    return Ks


def get_ordered_poses_and_k(
    poses: Float[Tensor, 'num_poses 3 4'],
    Ks: Float[Tensor, 'num_poses 3 3'],
) -> tuple[Float[Tensor, 'num_poses 3 4'], Float[Tensor, 'num_poses 3 3']]:
    """Return poses and intrinsics ordered by Euclidean distance between poses.

    Parameters
    ----------
    poses: list of camera poses.
    Ks: list of camera intrinsics.

    Returns
    -------
    tuple of ordered poses and intrinsics.

    """
    poses_num = len(poses)

    ordered_poses = torch.unsqueeze(poses[0], 0)
    ordered_ks = torch.unsqueeze(Ks[0], 0)

    poses = poses[1:]
    Ks = Ks[1:]

    for _ in range(poses_num - 1):
        distances = torch.norm(ordered_poses[-1][:, 3] - poses[:, :, 3], dim=1)
        idx = torch.argmin(distances)
        ordered_poses = torch.cat((ordered_poses, torch.unsqueeze(poses[idx], 0)), dim=0)
        ordered_ks = torch.cat((ordered_ks, torch.unsqueeze(Ks[idx], 0)), dim=0)
        poses = torch.cat((poses[0:idx], poses[idx + 1:]), dim=0)
        Ks = torch.cat((Ks[0:idx], Ks[idx + 1:]), dim=0)

    return ordered_poses, ordered_ks


def get_interpolated_poses_many(
    poses: Float[Tensor, 'num_poses 3 4'],
    Ks: Float[Tensor, 'num_poses 3 3'],
    steps_per_transition: int = 10,
    order_poses: bool = False,
) -> tuple[Float[Tensor, 'num_poses 3 4'], Float[Tensor, 'num_poses 3 3']]:
    """Return interpolated poses for many camera poses.

    Parameters
    ----------
    poses: list of camera poses.
    Ks: list of camera intrinsics.
    steps_per_transition: number of steps per transition.
    order_poses: whether to order poses by Euclidean distance.

    Returns
    -------
    tuple of (new poses, intrinsics).

    """
    traj = []
    k_interp = []

    if poses.shape[0] == 0:
        raise ValueError('get_interpolated_poses_many requires at least one pose')
    if poses.shape[0] == 1:
        traj = [poses[0].cpu().numpy()]
        k_interp = [Ks[0]]
    else:
        if order_poses:
            poses, Ks = get_ordered_poses_and_k(poses, Ks)
        for idx in range(poses.shape[0] - 1):
            pose_a = poses[idx].cpu().numpy()
            pose_b = poses[idx + 1].cpu().numpy()
            poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)
            traj += poses_ab
            k_interp += get_interpolated_k(Ks[idx], Ks[idx + 1], steps=steps_per_transition)

    traj = np.stack(traj, axis=0)
    k_interp = torch.stack(k_interp, dim=0)

    return torch.tensor(traj, dtype=torch.float32), torch.tensor(k_interp, dtype=torch.float32)


def w2c_to_c2w(w2c: torch.Tensor) -> torch.Tensor:
    """Convert world-to-camera matrices to camera-to-world using the analytical SE3 inverse.

    For a rigid body transform [R | t; 0 | 1], the inverse is [R^T | -R^T t; 0 | 1].
    Handles any leading batch dimensions.

    Parameters
    ----------
    w2c: world-to-camera matrices of shape (..., 4, 4).

    Returns
    -------
    camera-to-world matrices of shape (..., 4, 4).

    """
    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3:]
    R_T = R.transpose(-2, -1)
    t_new = -torch.matmul(R_T, t)
    c2w = torch.zeros_like(w2c)
    c2w[..., :3, :3] = R_T
    c2w[..., :3, 3:] = t_new
    c2w[..., 3, 3] = 1.0
    return c2w


def intrinsics_to_fxfycxcy(K: torch.Tensor) -> torch.Tensor:
    """Extract [fx, fy, cx, cy] from a 3x3 camera intrinsics matrix.

    Parameters
    ----------
    K: intrinsics matrix of shape (..., 3, 3).

    Returns
    -------
    tensor of shape (..., 4) containing [fx, fy, cx, cy].

    """
    return torch.stack(
        [K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]],
        dim=-1,
    )


def scale_intrinsics(K: torch.Tensor, scale_w: float, scale_h: float) -> torch.Tensor:
    """Scale camera intrinsics to account for an image resize.

    Parameters
    ----------
    K: intrinsics matrix of shape (..., 3, 3).
    scale_w: horizontal scale factor (new_width / orig_width).
    scale_h: vertical scale factor (new_height / orig_height).

    Returns
    -------
    scaled intrinsics matrix of shape (..., 3, 3).

    """
    K_scaled = K.clone()
    K_scaled[..., 0, 0] = K_scaled[..., 0, 0] * scale_w  # fx
    K_scaled[..., 0, 2] = K_scaled[..., 0, 2] * scale_w  # cx
    K_scaled[..., 1, 1] = K_scaled[..., 1, 1] * scale_h  # fy
    K_scaled[..., 1, 2] = K_scaled[..., 1, 2] * scale_h  # cy
    return K_scaled


def normalize_camera_sequence(extrinsics: torch.Tensor) -> torch.Tensor:
    """Normalize a sequence of w2c camera matrices and return c2w.

    Applies two normalizations in sequence:
    1. Re-center: transforms coordinates so camera 0 is at the world origin.
    2. Scale: divides all translations so the mean camera distance from origin is 1.

    Parameters
    ----------
    extrinsics: world-to-camera matrices of shape (V, 4, 4).

    Returns
    -------
    camera-to-world matrices of shape (V, 4, 4) in the normalized coordinate frame.

    """
    first_c2w = w2c_to_c2w(extrinsics[0:1]).squeeze(0)  # (4, 4)
    ex_norm = torch.matmul(extrinsics, first_c2w)  # (V, 4, 4)

    c2w = w2c_to_c2w(ex_norm)  # (V, 4, 4)
    scale = c2w[:, :3, 3].norm(dim=-1).mean()

    if scale > 1e-8:
        ex_norm = ex_norm.clone()
        ex_norm[:, :3, 3] = ex_norm[:, :3, 3] / scale
        c2w = w2c_to_c2w(ex_norm)

    return c2w


def cam_info_to_plucker(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    target_imgs_info: dict,
    normalized: bool = True,
    return_moment: bool = True,
) -> torch.Tensor:
    """Compute per-pixel Plucker ray embeddings from camera parameters.

    Parameters
    ----------
    c2w: camera-to-world matrices of shape [b, 4, 4] or [b, v, 4, 4].
    fxfycxcy: intrinsics of shape [b, 4] or [b, v, 4].
    target_imgs_info: dict with keys 'height' and 'width'.
    normalized: if True, scale fxfycxcy by image resolution before use.
    return_moment: if True, return moment+direction encoding; otherwise
        origin+direction.

    Returns
    -------
    Plucker ray tensor of shape [b, 6, h, w].

    """
    if len(c2w.shape) == 3:
        b = c2w.shape[0]
    elif len(c2w.shape) == 4:
        c2w = rearrange(c2w.clone(), 'b v n d -> (b v) n d')
        fxfycxcy = rearrange(fxfycxcy.clone(), 'b v d -> (b v) d')
        b = c2w.shape[0]

    h, w = target_imgs_info['height'], target_imgs_info['width']

    fxfycxcy = fxfycxcy.clone()
    if normalized:
        fxfycxcy[:, 0] *= w
        fxfycxcy[:, 1] *= h
        fxfycxcy[:, 2] *= w
        fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b, h*w, 3]

    ray_o = ray_o.reshape(b, h, w, 3).permute(0, 3, 1, 2)  # [b, 3, h, w]
    ray_d = ray_d.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    if return_moment:
        plucker = torch.cat(
            [
                torch.cross(ray_o, ray_d, dim=1),
                ray_d,
            ],
            dim=1,
        )
    else:
        plucker = torch.cat(
            [
                ray_o,
                ray_d,
            ],
            dim=1,
        )
    return plucker  # [b, 6, h, w]
