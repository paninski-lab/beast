"""Camera transformation helpers, Plucker ray encoding, and pose utilities.

Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors.
Licensed under the Apache License, Version 2.0.

"""

import math
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from beast.geometry.rotations import quat2mat, rot6d2mat

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


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
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
) -> list[float]:
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


def normalize(x: torch.Tensor) -> Float[Tensor, '*batch']:
    """Return a normalized vector."""
    return x / torch.linalg.norm(x)


def normalize_np(x: np.ndarray) -> np.ndarray:
    """Normalize a numpy array."""
    return x / np.linalg.norm(x)


def get_forward_facing_trajectory(
    c2w: torch.Tensor,
    Ks: torch.Tensor,
    N: int,
    N_rots: int = 2,
    zrate: float = 0.25,
    focal: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a forward-facing spiral trajectory around the central object.

    Parameters
    ----------
    c2w: 4x4 camera-to-world matrices in OpenCV format.
    Ks: 3x3 camera intrinsics.
    N: number of poses to generate.
    N_rots: number of rotations.
    zrate: z movement rate.
    focal: focal distance for look-at.

    Returns
    -------
    tuple of (poses [N, 4, 4], intrinsics [N, 3, 3]).

    """
    poses = []
    Ks_list = []
    rads = np.array([0.1, 0.1, 1.2, 1.])
    c2w = c2w.cpu().numpy()
    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([
                np.cos(theta),
                -np.sin(theta),
                (0.2 + theta / (2. * np.pi * N_rots)),
                1.,
            ]) * rads,
        )

        new_z = normalize_np(np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])) - c)
        new_x = normalize_np(np.cross(np.array([0, 1, 0]), new_z))
        new_y = normalize_np(np.cross(new_z, new_x))
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = np.stack([new_x, new_y, new_z], 1)
        new_c2w[:3, 3] = c
        poses.append(torch.tensor(new_c2w))
        Ks_list.append(Ks)
    return torch.stack(poses), torch.stack(Ks_list)


def normalize_with_norm(
    x: torch.Tensor,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize tensor along axis and return normalized value with norms.

    Parameters
    ----------
    x: tensor to normalize.
    dim: axis along which to normalize.

    Returns
    -------
    tuple of (normalized tensor, corresponding norms).

    """
    norm = torch.maximum(
        torch.linalg.vector_norm(x, dim=dim, keepdims=True),
        torch.tensor([_EPS]).to(x),
    )
    return x / norm, norm


def viewmatrix(
    lookat: torch.Tensor,
    up: torch.Tensor,
    pos: torch.Tensor,
) -> Float[Tensor, '*batch']:
    """Return a camera transformation matrix.

    Parameters
    ----------
    lookat: the direction the camera is looking.
    up: the upward direction of the camera.
    pos: the position of the camera.

    Returns
    -------
    camera transformation matrix.

    """
    vec2 = normalize(lookat)
    vec1_avg = normalize(up)
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m


def get_distortion_params(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Float[Tensor, '*batch']:
    """Return a distortion parameters tensor.

    Parameters
    ----------
    k1: first radial distortion parameter.
    k2: second radial distortion parameter.
    k3: third radial distortion parameter.
    k4: fourth radial distortion parameter.
    p1: first tangential distortion parameter.
    p2: second tangential distortion parameter.

    Returns
    -------
    distortion parameters tensor of shape (6,).

    """
    return torch.Tensor([k1, k2, k3, k4, p1, p2])


def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute residuals and jacobians for radial_and_tangential_undistort.

    Adapted from MultiNeRF.

    Parameters
    ----------
    x: updated x coordinates.
    y: updated y coordinates.
    xd: distorted x coordinates.
    yd: distorted y coordinates.
    distortion_params: distortion parameters [k1, k2, k3, k4, p1, p2].

    Returns
    -------
    tuple of (fx, fy, fx_x, fx_y, fy_x, fy_y).

    """
    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    p1 = distortion_params[..., 4]
    p2 = distortion_params[..., 5]

    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def radial_and_tangential_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
) -> torch.Tensor:
    """Compute undistorted coords given OpenCV distortion parameters.

    Adapted from MultiNeRF.

    Parameters
    ----------
    coords: distorted coordinates.
    distortion_params: distortion parameters [k1, k2, k3, k4, p1, p2].
    eps: convergence epsilon.
    max_iterations: maximum number of Newton iterations.

    Returns
    -------
    undistorted coordinates.

    """
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=coords[..., 0], yd=coords[..., 1], distortion_params=distortion_params,
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(
            torch.abs(denominator) > eps,
            x_numerator / denominator,
            torch.zeros_like(denominator),
        )
        step_y = torch.where(
            torch.abs(denominator) > eps,
            y_numerator / denominator,
            torch.zeros_like(denominator),
        )

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)


def rotation_matrix(
    a: Float[Tensor, '3'],
    b: Float[Tensor, '3'],
) -> Float[Tensor, '3 3']:
    """Compute the rotation matrix that rotates vector a to vector b.

    Parameters
    ----------
    a: the vector to rotate.
    b: the vector to rotate to.

    Returns
    -------
    rotation matrix of shape (3, 3).

    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))


def focus_of_attention(
    poses: Float[Tensor, '*num_poses 4 4'],
    initial_focus: Float[Tensor, '3'],
) -> Float[Tensor, '3']:
    """Compute the focus of attention of a set of cameras.

    Only cameras that have the focus of attention in front of them are considered.

    Parameters
    ----------
    poses: camera poses of shape (num_poses, 4, 4).
    initial_focus: initial 3D point to activate cameras.

    Returns
    -------
    3D position of the focus of attention.

    """
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    focus_pt = initial_focus
    active = torch.sum(
        active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1,
    ) > 0
    done = False
    while torch.sum(active.int()) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]
        m = torch.eye(3) - active_directions * torch.transpose(active_directions, -2, -1)
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
        active = torch.sum(
            active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1,
        ) > 0
        if active.all():
            done = True
    return focus_pt


def auto_orient_and_center_poses(
    poses: Float[Tensor, '*num_poses 4 4'],
    method: Literal['pca', 'up', 'vertical', 'none'] = 'up',
    center_method: Literal['poses', 'focus', 'none'] = 'poses',
) -> tuple[Float[Tensor, '*num_poses 3 4'], Float[Tensor, '3 4']]:
    """Orient and center camera poses.

    Parameters
    ----------
    poses: camera poses of shape (num_poses, 4, 4).
    method: orientation method — 'pca', 'up', 'vertical', or 'none'.
    center_method: centering method — 'poses', 'focus', or 'none'.

    Returns
    -------
    tuple of (oriented poses [num_poses, 3, 4], transform [3, 4]).

    """
    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == 'poses':
        translation = mean_origin
    elif center_method == 'focus':
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == 'none':
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f'Unknown value for center_method: {center_method}')

    if method == 'pca':
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method in ('up', 'vertical'):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == 'vertical':
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                up_vertical = Vh[2, :]
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                up = up / torch.linalg.norm(up)

        rot = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rot, rot @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == 'none':
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f'Unknown value for method: {method}')

    return oriented_poses, transform


@torch.jit.script
def fisheye624_project(xyz, params):
    """Batched FisheyeRadTanThinPrism (Fisheye624) projection.

    Parameters
    ----------
    xyz: BxNx3 tensor of 3D points to be projected.
    params: Bx16 or Bx15 tensor of Fisheye624 parameters.

    Returns
    -------
    uv: BxNx2 tensor of 2D projections.

    """
    assert xyz.ndim == 3
    assert params.ndim == 2
    assert params.shape[-1] == 16 or params.shape[-1] == 15, 'This model allows fx != fy'
    eps = 1e-9
    B, N = xyz.shape[0], xyz.shape[1]

    z = xyz[:, :, 2].reshape(B, N, 1)
    z = torch.where(torch.abs(z) < eps, eps * torch.sign(z), z)
    ab = xyz[:, :, :2] / z
    r = torch.norm(ab, dim=-1, p=2, keepdim=True)
    th = torch.atan(r)
    th_divr = torch.where(r < eps, torch.ones_like(ab), ab / r)
    th_k = th.reshape(B, N, 1).clone()
    for i in range(6):
        th_k = th_k + params[:, -12 + i].reshape(B, 1, 1) * torch.pow(th, 3 + i * 2)
    xr_yr = th_k * th_divr
    uv_dist = xr_yr

    p0 = params[:, -6].reshape(B, 1)
    p1 = params[:, -5].reshape(B, 1)
    xr = xr_yr[:, :, 0].reshape(B, N)
    yr = xr_yr[:, :, 1].reshape(B, N)
    xr_yr_sq = torch.square(xr_yr)
    xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
    yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
    rd_sq = xr_sq + yr_sq
    uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
    uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
    uv_dist = torch.stack([uv_dist_tu, uv_dist_tv], dim=-1)

    s0 = params[:, -4].reshape(B, 1)
    s1 = params[:, -3].reshape(B, 1)
    s2 = params[:, -2].reshape(B, 1)
    s3 = params[:, -1].reshape(B, 1)
    rd_4 = torch.square(rd_sq)
    uv_dist[:, :, 0] = uv_dist[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
    uv_dist[:, :, 1] = uv_dist[:, :, 1] + (s2 * rd_sq + s3 * rd_4)

    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)
    result = uv_dist * fx_fy + cx_cy

    return result


@torch.jit.script
def fisheye624_unproject_helper(uv, params, max_iters: int = 5):
    """Batched FisheyeRadTanThinPrism unprojection via Newton's method.

    Parameters
    ----------
    uv: BxNx2 tensor of 2D pixels to be unprojected.
    params: Bx16 or Bx15 tensor of Fisheye624 parameters.
    max_iters: number of Newton iterations.

    Returns
    -------
    xyz: BxNx3 tensor of 3D rays with z=1.

    """
    assert uv.ndim == 3, 'Expected batched input shaped BxNx3'
    assert params.ndim == 2
    assert params.shape[-1] == 16 or params.shape[-1] == 15, 'This model allows fx != fy'
    eps = 1e-6
    B, N = uv.shape[0], uv.shape[1]

    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    xr_yr = uv_dist.clone()
    for _ in range(max_iters):
        uv_dist_est = xr_yr.clone()
        p0 = params[:, -6].reshape(B, 1)
        p1 = params[:, -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (
            (2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1
        )
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (
            (2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0
        )
        s0 = params[:, -4].reshape(B, 1)
        s1 = params[:, -3].reshape(B, 1)
        s2 = params[:, -2].reshape(B, 1)
        s3 = params[:, -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
        duv_dist_dxr_yr = uv.new_ones(B, N, 2, 2)
        duv_dist_dxr_yr[:, :, 0, 0] = 1.0 + 6.0 * xr_yr[:, :, 0] * p0 + 2.0 * xr_yr[:, :, 1] * p1
        offdiag = 2.0 * (xr_yr[:, :, 0] * p1 + xr_yr[:, :, 1] * p0)
        duv_dist_dxr_yr[:, :, 0, 1] = offdiag
        duv_dist_dxr_yr[:, :, 1, 0] = offdiag
        duv_dist_dxr_yr[:, :, 1, 1] = 1.0 + 6.0 * xr_yr[:, :, 1] * p1 + 2.0 * xr_yr[:, :, 0] * p0
        xr_yr_sq_norm = xr_yr_sq[:, :, 0] + xr_yr_sq[:, :, 1]
        temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 0, 0] = duv_dist_dxr_yr[:, :, 0, 0] + (xr_yr[:, :, 0] * temp1)
        duv_dist_dxr_yr[:, :, 0, 1] = duv_dist_dxr_yr[:, :, 0, 1] + (xr_yr[:, :, 1] * temp1)
        temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 1, 0] = duv_dist_dxr_yr[:, :, 1, 0] + (xr_yr[:, :, 0] * temp2)
        duv_dist_dxr_yr[:, :, 1, 1] = duv_dist_dxr_yr[:, :, 1, 1] + (xr_yr[:, :, 1] * temp2)
        mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
        a = mat[:, 0, 0].reshape(-1, 1, 1)
        b = mat[:, 0, 1].reshape(-1, 1, 1)
        c = mat[:, 1, 0].reshape(-1, 1, 1)
        d = mat[:, 1, 1].reshape(-1, 1, 1)
        det = 1.0 / ((a * d) - (b * c))
        top = torch.cat([d, -b], dim=2)
        bot = torch.cat([-c, a], dim=2)
        inv = det * torch.cat([top, bot], dim=1)
        inv = inv.reshape(B, N, 2, 2)
        diff = uv_dist - uv_dist_est
        a = inv[:, :, 0, 0]
        b = inv[:, :, 0, 1]
        c = inv[:, :, 1, 0]
        d = inv[:, :, 1, 1]
        e = diff[:, :, 0]
        f = diff[:, :, 1]
        step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
        xr_yr = xr_yr + step

    xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
    th = xr_yr_norm.clone()
    for _ in range(max_iters):
        th_radial = uv.new_ones(B, N, 1)
        dthd_th = uv.new_ones(B, N, 1)
        for k in range(6):
            r_k = params[:, -12 + k].reshape(B, 1, 1)
            th_radial = th_radial + (r_k * torch.pow(th, 2 + k * 2))
            dthd_th = dthd_th + ((3.0 + 2.0 * k) * r_k * torch.pow(th, 2 + k * 2))
        th_radial = th_radial * th
        step = (xr_yr_norm - th_radial) / dthd_th
        step = torch.where(dthd_th.abs() > eps, step, torch.sign(step) * eps * 10.0)
        th = th + step
    close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
    ray_dir = torch.where(close_to_zero, xr_yr, torch.tan(th) / xr_yr_norm * xr_yr)
    ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)
    return ray


def fisheye624_unproject(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
) -> torch.Tensor:
    """Unproject 2D point to 3D with Fisheye624 model.

    Parameters
    ----------
    coords: 2D coordinates tensor.
    distortion_params: Fisheye624 distortion parameters.

    Returns
    -------
    3D ray directions.

    """
    dirs = fisheye624_unproject_helper(
        coords.unsqueeze(0), distortion_params[0].unsqueeze(0),
    )
    dirs[..., 1] = -dirs[..., 1]
    dirs[..., 2] = -dirs[..., 2]
    return dirs


def get_cam_se3(
    cam_info: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert encoded camera info to SE(3) matrix and intrinsics.

    Parameters
    ----------
    cam_info: tensor of shape [b, n] where n=13 uses 6D rotation or n=11
        uses quaternion. Layout is [rot, 3D trans, 4D fxfycxcy].

    Returns
    -------
    tuple of (c2w [b, 4, 4], fxfycxcy [b, 4]).

    Raises
    ------
    NotImplementedError
        if cam_info width is not 11 or 13.

    """
    b, n = cam_info.shape

    if n == 13:
        rot_6d = cam_info[:, :6]
        R = rot6d2mat(rot_6d)  # [b, 3, 3]
        t = cam_info[:, 6:9].unsqueeze(-1)  # [b, 3, 1]
        fxfycxcy = cam_info[:, 9:]  # [b, 4]
    elif n == 11:
        rot_quat = cam_info[:, :4]
        R = quat2mat(rot_quat)
        t = cam_info[:, 4:7].unsqueeze(-1)  # [b, 3, 1]
        fxfycxcy = cam_info[:, 7:]  # [b, 4]
    else:
        raise NotImplementedError

    Rt = torch.cat([R, t], dim=2)  # [b, 3, 4]
    bottom = (
        torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
        .view(1, 1, 4)
        .repeat(b, 1, 1)
    )
    c2w = torch.cat([Rt, bottom], dim=1)  # [b, 4, 4]
    return c2w, fxfycxcy


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
