"""Validation-time visualization for ERayZer.

Builds TensorBoard-ready images so training quality can be inspected live:

- ``make_render_grid``: a 2-row GT-vs-render comparison over all views, with
  colored borders marking input (reference) vs target (novel) views.
- ``make_camera_pose_image``: a 3D plot of all predicted camera poses as
  frustums, colored by input vs target (frustum style ported from the
  multi-view fig_teaser_cam_erayzer figure).
- ``camera_intrinsic_stats``: scalar summaries of the predicted intrinsics.
- ``export_gaussian_glb``: write the predicted Gaussians as a GLB point cloud.
- ``viz_is_due``: cadence helper deciding when to log during validation.

Image builders return ``[3, H, W]`` float tensors in ``[0, 1]`` for
``SummaryWriter.add_image`` with the default ``CHW`` data format.
"""

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

# shared color code: input (reference) views vs target (novel) views
_INPUT_RGB = (0.20, 0.80, 0.36)   # green
_TARGET_RGB = (0.95, 0.35, 0.20)  # orange-red
_SH_C0 = 0.28209479177387814      # 0th-order SH coefficient (SH DC -> RGB)


def viz_is_due(batch_idx: int, current_epoch: int, every_n_epochs: int) -> bool:
    """Decide whether validation visuals should be logged for this batch.

    Visuals are logged only for the first validation batch of an epoch, and only
    every ``every_n_epochs`` epochs.

    Parameters
    ----------
    batch_idx: index of the current validation batch.
    current_epoch: current training epoch.
    every_n_epochs: log cadence in epochs; ``<= 0`` disables logging.

    Returns
    -------
    True if visuals should be logged now.

    """
    if every_n_epochs <= 0:
        return False
    if batch_idx != 0:
        return False
    return current_epoch % every_n_epochs == 0


# ---------------------------------------------------------------------------
# Render grid
# ---------------------------------------------------------------------------


def _select_sample(t: torch.Tensor, sample_idx: int) -> torch.Tensor:
    """Return a single sample [V, ...] from a [B, V, ...] or [V, ...] tensor."""
    if t.ndim >= 5:
        return t[sample_idx]
    return t


def _add_border(
    img: torch.Tensor,
    color: tuple[float, float, float],
    width: int = 4,
) -> torch.Tensor:
    """Paint a solid colored border of ``width`` pixels around a [3, H, W] image."""
    out = img.clone()
    c = torch.tensor(color, dtype=out.dtype).view(3, 1, 1)
    out[:, :width, :] = c
    out[:, -width:, :] = c
    out[:, :, :width] = c
    out[:, :, -width:] = c
    return out


def _row_from_views(
    views: torch.Tensor,
    sample_idx: int,
    border_color: tuple[float, float, float],
    pad: int = 2,
) -> torch.Tensor:
    """Tile a sample's views horizontally, each with a colored border."""
    v = _select_sample(views, sample_idx).detach().cpu().float().clamp(0.0, 1.0)
    h = v.shape[2]
    sep = torch.ones(3, h, pad)
    cols = []
    for i in range(v.shape[0]):
        cols.append(_add_border(v[i], border_color))
        if i < v.shape[0] - 1:
            cols.append(sep)
    return torch.cat(cols, dim=2)


def _build_row(
    groups: list[tuple[torch.Tensor, tuple[float, float, float]]],
    sample_idx: int,
    group_gap: int = 10,
) -> torch.Tensor:
    """Concatenate per-group bordered view strips into one [3, H, W] row."""
    parts = []
    for i, (views, color) in enumerate(groups):
        row = _row_from_views(views, sample_idx, color)
        parts.append(row)
        if i < len(groups) - 1:
            parts.append(torch.ones(3, row.shape[1], group_gap))  # gap between groups
    return torch.cat(parts, dim=2)


def _pad_width(row: torch.Tensor, width: int, fill: float = 1.0) -> torch.Tensor:
    """Right-pad a [3, H, W] row to ``width`` with a constant fill."""
    cur = row.shape[2]
    if cur >= width:
        return row
    pad = torch.full((3, row.shape[1], width - cur), fill)
    return torch.cat([row, pad], dim=2)


def _stack_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    """Right-pad rows to a common width and stack them with grey separators."""
    width = max(row.shape[2] for row in rows)
    rows = [_pad_width(row, width) for row in rows]
    sep = torch.full((3, 3, width), 0.5)  # grey separator between rows
    stacked = []
    for i, row in enumerate(rows):
        stacked.append(row)
        if i < len(rows) - 1:
            stacked.append(sep)
    return torch.cat(stacked, dim=1).clamp(0.0, 1.0).float()


def make_render_grid(
    input_image: torch.Tensor,
    target_image: torch.Tensor,
    render: torch.Tensor,
    render_input: torch.Tensor | None = None,
    sample_idx: int = 0,
) -> torch.Tensor:
    """Build a 2-row GT-vs-render comparison grid over all views.

    Row 0 is ground truth, row 1 is the model's render, with one column per view
    in the order input views then target views. Each image gets a colored
    border: green for input (reference) views, red for target (novel) views, so
    the two view types are distinguishable at a glance. When ``render_input`` is
    omitted only the target views are shown (the input views cannot be rendered).

    Inputs are stacks of per-view images shaped ``[B, V, 3, H, W]`` (or
    ``[V, 3, H, W]``).

    Parameters
    ----------
    input_image: reference views fed to the model.
    target_image: ground-truth novel (target) views.
    render: predicted renders of the target views.
    render_input: optional renders of the input views (input-view NVS).
    sample_idx: which batch element to visualize.

    Returns
    -------
    a ``[3, H, W]`` float image in ``[0, 1]``: row 0 GT, row 1 render.

    """
    if render_input is not None:
        gt_row = _build_row(
            [(input_image, _INPUT_RGB), (target_image, _TARGET_RGB)], sample_idx,
        )
        render_row = _build_row(
            [(render_input, _INPUT_RGB), (render, _TARGET_RGB)], sample_idx,
        )
    else:
        gt_row = _build_row([(target_image, _TARGET_RGB)], sample_idx)
        render_row = _build_row([(render, _TARGET_RGB)], sample_idx)
    return _stack_rows([gt_row, render_row])


# ---------------------------------------------------------------------------
# Camera-pose plot
# ---------------------------------------------------------------------------


def _select_cameras(c2w: torch.Tensor, sample_idx: int) -> np.ndarray:
    """Return a [V, 4, 4] numpy array of camera matrices for one sample."""
    if c2w.ndim == 4:
        c2w = c2w[sample_idx]
    return c2w.detach().cpu().to(torch.float32).numpy()


def _frustum_segments(c2w: np.ndarray, scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Line segments of one camera frustum in world coords (OpenCV convention).

    Apex at the camera center, base ahead at +z, plus a short stub on the top
    edge encoding the 'up' direction so orientation flips are visible.
    """
    s = scale
    pts_cam = np.array([
        [0.0, 0.0, 0.0],       # apex
        [-s, -s, 1.5 * s],     # base top-left
        [s, -s, 1.5 * s],      # base top-right
        [s, s, 1.5 * s],       # base bottom-right
        [-s, s, 1.5 * s],      # base bottom-left
    ])
    rot = c2w[:3, :3]
    trans = c2w[:3, 3]
    pts_w = (rot @ pts_cam.T).T + trans
    apex, base = pts_w[0], pts_w[1:]
    segs = [(apex, base[i]) for i in range(4)]
    segs += [(base[0], base[1]), (base[1], base[2]), (base[2], base[3]), (base[3], base[0])]
    top_mid = 0.5 * (base[0] + base[1])
    segs.append((top_mid, top_mid + 0.4 * s * (-rot[:, 1])))  # up stub
    return segs


def _frustum_scale_for(*c2w_arrays: np.ndarray) -> float:
    """Pick a frustum size from the spread of camera centers."""
    centers = np.concatenate([c[:, :3, 3] for c in c2w_arrays], axis=0)
    radii = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
    median_radius = float(np.median(radii)) if radii.size else 1.0
    return 0.15 * (median_radius if median_radius > 1e-6 else 1.0)


def _draw_camera_frustums(
    ax: Axes3D,
    c2w_np: np.ndarray,
    color: tuple[float, float, float],
    label: str,
    scale: float,
) -> None:
    """Plot one set of camera frustums plus center markers on ``ax``."""
    centers = c2w_np[:, :3, 3]
    # Axes3D.scatter accepts an array for zs at runtime; its stub types it as int
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],  # pyright: ignore[reportArgumentType]
        color=color, s=24, depthshade=False, label=label,
    )
    for v in range(c2w_np.shape[0]):
        for p0, p1 in _frustum_segments(c2w_np[v], scale):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=color, linewidth=1.3)


def _set_axes_equal(ax: Axes3D, centers: np.ndarray, pad: float = 0.3) -> None:
    """Tight equal-aspect bounding cube around all camera centers."""
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    mid = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    half = (1.0 + pad) * 0.5 * (extent if extent > 0 else 1.0)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))


def make_camera_pose_image(
    c2w_input: torch.Tensor,
    c2w_target: torch.Tensor | None = None,
    sample_idx: int = 0,
) -> torch.Tensor:
    """Render a 3D plot of all predicted camera poses to an image tensor.

    Input (reference) cameras are drawn as green frustums and target (novel)
    cameras as red frustums, so every camera in the batch is shown and the two
    types are color-coded. Uses the Agg backend so it works headless.

    Parameters
    ----------
    c2w_input: input camera-to-world matrices ``[B, V, 4, 4]`` or ``[V, 4, 4]``.
    c2w_target: optional target camera-to-world matrices, same layout.
    sample_idx: which batch element to visualize.

    Returns
    -------
    a ``[3, H, W]`` float image in ``[0, 1]``.

    """
    cams_input = _select_cameras(c2w_input, sample_idx)
    arrays = [cams_input]
    cams_target = None
    if c2w_target is not None:
        cams_target = _select_cameras(c2w_target, sample_idx)
        arrays.append(cams_target)
    scale = _frustum_scale_for(*arrays)

    fig = Figure(figsize=(6, 5))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15, azim=-70)
    _draw_camera_frustums(ax, cams_input, _INPUT_RGB, 'input', scale)
    if cams_target is not None:
        _draw_camera_frustums(ax, cams_target, _TARGET_RGB, 'target', scale)

    centers = np.concatenate([c[:, :3, 3] for c in arrays], axis=0)
    _set_axes_equal(ax, centers)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # set_zticklabels is wrapped by a descriptor the stub types as non-callable
    ax.set_zticklabels([])  # pyright: ignore[reportCallIssue]
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('predicted cameras (green=input, red=target)')

    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()  # [H, W, 3] uint8
    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return img.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Scalars + point cloud
# ---------------------------------------------------------------------------


def camera_intrinsic_stats(fxfycxcy: torch.Tensor, image_size: int) -> dict[str, float]:
    """Summarize predicted intrinsics as normalized scalar statistics.

    Parameters
    ----------
    fxfycxcy: intrinsics tensor of shape ``[..., 4]`` in pixels (fx, fy, cx, cy).
    image_size: image side length used to normalize pixel intrinsics.

    Returns
    -------
    dict of plain floats: fx/fy means, fx spread (std), and cx/cy means, all
    normalized to ``[0, 1]`` image fractions.

    """
    norm = fxfycxcy.detach().float() / image_size
    fx = norm[..., 0].flatten()
    fy = norm[..., 1].flatten()
    cx = norm[..., 2].flatten()
    cy = norm[..., 3].flatten()
    return {
        'focal_fx_mean': fx.mean().item(),
        'focal_fx_std': fx.std(unbiased=False).item(),
        'focal_fy_mean': fy.mean().item(),
        'cx_mean': cx.mean().item(),
        'cy_mean': cy.mean().item(),
    }


def export_gaussian_glb(gaussian_model, path, opacity_threshold: float = 0.0) -> int:
    """Export a predicted Gaussian field as a colored point-cloud GLB.

    Reads positions and SH-DC color from a ``GaussianModel`` (anything exposing
    ``get_xyz``, ``get_features``, ``get_opacity``), optionally drops near-empty
    Gaussians, and writes a GLB point cloud. ``trimesh`` is imported lazily so the
    core model import does not depend on it.

    Parameters
    ----------
    gaussian_model: a GaussianModel-like object with get_xyz/get_features/get_opacity.
    path: output ``.glb`` path.
    opacity_threshold: drop Gaussians with activated opacity at or below this.

    Returns
    -------
    number of points written (0 if nothing passed the filter; no file is written).

    """
    from pathlib import Path

    import trimesh  # lazy: keep trimesh optional for the core model import

    xyz = gaussian_model.get_xyz.detach().cpu().float()
    feat = gaussian_model.get_features.detach().cpu().float()  # [N, K, 3]
    opacity = gaussian_model.get_opacity.detach().cpu().float().reshape(-1)
    rgb = (feat[:, 0, :] * _SH_C0 + 0.5).clamp(0.0, 1.0)  # SH DC -> RGB

    if opacity_threshold > 0.0:
        mask = opacity > opacity_threshold
        xyz, rgb = xyz[mask], rgb[mask]

    n = int(xyz.shape[0])
    if n == 0:
        return 0

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = (rgb.numpy() * 255).astype(np.uint8)
    rgba = np.concatenate([colors, np.full((n, 1), 255, np.uint8)], axis=1)
    trimesh.PointCloud(vertices=xyz.numpy(), colors=rgba).export(str(path))
    return n
