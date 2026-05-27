"""ICP and Kabsch alignment utilities for point cloud registration."""

import contextlib
from typing import Any

import numpy as np
import torch

try:
    import open3d as o3d
    from open3d.pipelines.registration import RegistrationResult
except ImportError:
    o3d = None
    RegistrationResult = Any


def run_icp(
    source_pcd,
    target_pcd,
    src_idx: list[int],
    tgt_idx: list[int],
    max_correspondence_distance: float = 0.05,
) -> tuple[np.ndarray, Any]:
    """Run the full ICP pipeline: Kabsch init → ICP refinement.

    Args:
        source_pcd: Open3D PointCloud to be aligned.
        target_pcd: Open3D PointCloud used as reference.
        src_idx: source landmark indices for Kabsch initialisation.
        tgt_idx: target landmark indices for Kabsch initialisation.
        max_correspondence_distance: ICP correspondence threshold.

    Returns:
        tuple of (T_icp [4, 4] ndarray, RegistrationResult).
    """
    T_init = estimate_initial_transform(source_pcd, target_pcd, src_idx, tgt_idx)

    result, T_icp = refine_with_icp(source_pcd, target_pcd, T_init, max_correspondence_distance)

    return T_icp, result


def kabsch_transform(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
    """Estimate rigid-body transform from N≥3 point correspondences via Kabsch algorithm.

    Args:
        src_pts: source points of shape (N, 3).
        tgt_pts: corresponding target points of shape (N, 3).

    Returns:
        SE(3) transformation matrix of shape (4, 4) such that tgt ≈ T @ [src | 1]ᵀ.
    """
    src = np.asarray(src_pts, dtype=float)
    tgt = np.asarray(tgt_pts, dtype=float)

    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    src_demean = src - src_c
    tgt_demean = tgt - tgt_c

    H = src_demean.T @ tgt_demean

    U, _, Vt = np.linalg.svd(H)

    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)

    R = Vt.T @ D @ U.T
    t = tgt_c - R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def kabsch_rotation_batched(
    cross_cov: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Batched Wahba/Kabsch rotation from cross-covariance H = Xc^T Yc of shape [B, 3, 3].

    Runs SVD/det in float32. Returns proper SO(3) R of shape [B, 3, 3] cast to out_dtype.

    Args:
        cross_cov: cross-covariance matrices of shape [B, 3, 3].
        eps: regularisation term added to the diagonal before SVD.
        out_dtype: output dtype.

    Returns:
        rotation matrices of shape [B, 3, 3].
    """
    dev = cross_cov.device
    h32 = cross_cov.float()
    eye = torch.eye(3, device=dev, dtype=torch.float32).unsqueeze(0).expand_as(h32)
    h32 = h32 + eye * float(eps)

    u, _, vh = torch.linalg.svd(h32)
    v = vh.transpose(-2, -1)
    r = torch.bmm(v, u.transpose(-2, -1))

    neg = torch.linalg.det(r) < 0
    if neg.any():
        vf = v.clone()
        vf[neg, :, 2] *= -1.0
        r = torch.bmm(vf, u.transpose(-2, -1))

    return r.to(out_dtype)


def estimate_merge_kabsch_rt_torch(
    src_cloud: torch.Tensor,
    tgt_cloud: torch.Tensor,
    source_indices: list[int],
    target_indices: list[int],
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kabsch SE(3) from indexed correspondences on full (N, 3) point clouds.

    Row-vector convention: x' = x @ R.T + t maps source toward target.
    Differentiable w.r.t. src_cloud and tgt_cloud; indices are discrete.

    Args:
        src_cloud: source point cloud of shape (N, 3).
        tgt_cloud: target point cloud of shape (N, 3).
        source_indices: correspondence indices into src_cloud.
        target_indices: correspondence indices into tgt_cloud.
        eps: regularisation for SVD.

    Returns:
        tuple of (R [3, 3], t [3]) tensors.
    """
    if src_cloud.dim() != 2 or src_cloud.shape[-1] != 3:
        raise ValueError(f'Expected src_cloud (N, 3), got {src_cloud.shape}')
    if tgt_cloud.shape != src_cloud.shape:
        raise ValueError(f'tgt_cloud shape {tgt_cloud.shape} != src {src_cloud.shape}')

    ii = torch.tensor(source_indices, device=src_cloud.device, dtype=torch.long)
    jj = torch.tensor(target_indices, device=src_cloud.device, dtype=torch.long)
    src_corr = src_cloud.index_select(0, ii)
    tgt_corr = tgt_cloud.index_select(0, jj)
    out_dtype = src_cloud.dtype

    mu_s = src_corr.mean(dim=0)
    mu_t = tgt_corr.mean(dim=0)
    xc = src_corr - mu_s
    yc = tgt_corr - mu_t
    h = (xc.T @ yc).unsqueeze(0)

    amp_off = (
        torch.amp.autocast(device_type='cuda', enabled=False)
        if src_cloud.is_cuda
        else contextlib.nullcontext()
    )
    with amp_off:
        r = kabsch_rotation_batched(h, eps=eps, out_dtype=out_dtype).squeeze(0)

    t = mu_t - mu_s @ r.T
    return r, t


def run_point_to_point_icp(
    source,
    target,
    trans_init: np.ndarray,
    threshold: float = 0.02,
):
    """Run point-to-point ICP registration.

    Args:
        source: Open3D source PointCloud.
        target: Open3D target PointCloud.
        trans_init: initial transformation of shape (4, 4).
        threshold: maximum correspondence distance.

    Returns:
        Open3D RegistrationResult.
    """
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )


def estimate_initial_transform(
    source_pcd,
    target_pcd,
    source_indices: list[int],
    target_indices: list[int],
) -> np.ndarray:
    """Compute initial SE(3) transform from manual correspondences via Kabsch algorithm.

    Args:
        source_pcd: Open3D source PointCloud.
        target_pcd: Open3D target PointCloud.
        source_indices: indices of landmark points in source_pcd.
        target_indices: indices of corresponding landmark points in target_pcd.

    Returns:
        T_init: (4, 4) ndarray initial transformation.
    """
    src_pts = np.asarray(source_pcd.points)[source_indices]
    tgt_pts = np.asarray(target_pcd.points)[target_indices]

    T_init = kabsch_transform(src_pts, tgt_pts)

    src_transformed = (T_init[:3, :3] @ src_pts.T).T + T_init[:3, 3]
    residuals = np.linalg.norm(src_transformed - tgt_pts, axis=1)  # noqa: F841
    return T_init


def refine_with_icp(
    source_pcd,
    target_pcd,
    T_init: np.ndarray,
    max_correspondence_distance: float = 0.05,
) -> tuple[Any, np.ndarray]:
    """Refine the Kabsch initial transform with point-to-point ICP.

    Args:
        source_pcd: Open3D source PointCloud.
        target_pcd: Open3D target PointCloud.
        T_init: initial transformation of shape (4, 4).
        max_correspondence_distance: ICP correspondence threshold (same units as cloud).

    Returns:
        tuple of (RegistrationResult, T_icp [4, 4] ndarray).
    """
    result = run_point_to_point_icp(source_pcd, target_pcd, T_init, max_correspondence_distance)

    T_icp = np.asarray(result.transformation)
    return result, T_icp
