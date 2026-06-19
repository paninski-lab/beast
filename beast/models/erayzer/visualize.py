"""Validation-time visualization for ERayZer.

Builds TensorBoard-ready images so training quality can be inspected live:

- ``make_render_grid``: an input / GT-target / predicted-render comparison grid.
- ``make_camera_pose_image``: a 3D plot of predicted camera poses (frustustum
  axes) so pose prediction can be sanity-checked.
- ``camera_intrinsic_stats``: scalar summaries of the predicted intrinsics.
- ``viz_is_due``: cadence helper deciding when to log during validation.

All image builders return ``[3, H, W]`` float tensors in ``[0, 1]`` suitable for
``SummaryWriter.add_image`` with the default ``CHW`` data format.
"""

from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

_AXIS_COLORS = ('r', 'g', 'b')  # camera x / y / z axes
_SH_C0 = 0.28209479177387814  # 0th-order SH coefficient (SH DC -> RGB)


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


def _select_sample(t: torch.Tensor, sample_idx: int) -> torch.Tensor:
    """Return a single sample [V, ...] from a [B, V, ...] or [V, ...] tensor."""
    if t.ndim >= 5:
        return t[sample_idx]
    return t


def _hstack_views(t: torch.Tensor, sample_idx: int, pad: int = 2) -> torch.Tensor:
    """Horizontally tile a sample's views into one [3, H, sum_W] image."""
    views = _select_sample(t, sample_idx).detach().cpu().float().clamp(0.0, 1.0)
    h = views.shape[2]
    sep = torch.ones(3, h, pad)
    cols = []
    for i in range(views.shape[0]):
        cols.append(views[i])
        if i < views.shape[0] - 1:
            cols.append(sep)
    return torch.cat(cols, dim=2)


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
    sep = torch.full((3, 2, width), 0.5)  # grey separator between rows
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
    sample_idx: int = 0,
) -> torch.Tensor:
    """Build an input / GT-target / predicted-render comparison grid.

    Each input is a stack of per-view images shaped ``[B, V, 3, H, W]`` (or
    ``[V, 3, H, W]``). The three rows are tiled horizontally over their views and
    stacked vertically; rows with fewer views are right-padded so widths match.

    Parameters
    ----------
    input_image: reference views fed to the model.
    target_image: ground-truth novel (target) views.
    render: predicted renders of the target views.
    sample_idx: which batch element to visualize.

    Returns
    -------
    a ``[3, H, W]`` float image in ``[0, 1]``: row 0 input, row 1 GT, row 2 pred.

    """
    return _stack_rows([
        _hstack_views(input_image, sample_idx),
        _hstack_views(target_image, sample_idx),
        _hstack_views(render, sample_idx),
    ])


def make_recon_grid(
    gt_views: torch.Tensor,
    pred_views: torch.Tensor,
    sample_idx: int = 0,
) -> torch.Tensor:
    """Build a GT-vs-render comparison grid for input-view reconstruction (NVS).

    Rendering the predicted Gaussians from the *input* camera poses and comparing
    against the input images is a self-reconstruction check on the geometry.

    Parameters
    ----------
    gt_views: ground-truth views ``[B, V, 3, H, W]`` (or ``[V, 3, H, W]``).
    pred_views: rendered views at the same cameras, matching shape.
    sample_idx: which batch element to visualize.

    Returns
    -------
    a ``[3, H, W]`` float image in ``[0, 1]``: row 0 GT, row 1 render.

    """
    return _stack_rows([
        _hstack_views(gt_views, sample_idx),
        _hstack_views(pred_views, sample_idx),
    ])


def _select_cameras(c2w: torch.Tensor, sample_idx: int) -> np.ndarray:
    """Return a [V, 4, 4] numpy array of camera matrices for one sample."""
    if c2w.ndim == 4:
        c2w = c2w[sample_idx]
    return c2w.detach().cpu().to(torch.float32).numpy()


def _set_equal_aspect(ax: Axes3D) -> None:
    """Force equal 3D aspect ratio so camera geometry is not warped."""
    lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    span = (lims[:, 1] - lims[:, 0]).max() / 2
    if span <= 0:
        span = 1.0
    mid = lims.mean(axis=1)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


def make_camera_pose_image(
    c2w: torch.Tensor,
    sample_idx: int = 0,
    scale: float = 0.5,
) -> torch.Tensor:
    """Render a 3D plot of predicted camera poses to an image tensor.

    Each camera center is a black dot; its orientation is drawn as RGB axes
    (red=x, green=y, blue=z). Uses the Agg backend so it works headless.

    Parameters
    ----------
    c2w: camera-to-world matrices ``[B, V, 4, 4]`` or ``[V, 4, 4]``.
    sample_idx: which batch element to visualize.
    scale: length of the drawn orientation axes.

    Returns
    -------
    a ``[3, H, W]`` float image in ``[0, 1]``.

    """
    cams = _select_cameras(c2w, sample_idx)

    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(cams.shape[0]):
        cam = cams[i]
        center = cam[:3, 3]
        ax.scatter(*center, color='black', s=20)
        ax.text(*center, f' {i}', verticalalignment='bottom')
        for j in range(3):
            direction = cam[:3, j]
            ax.quiver(
                center[0], center[1], center[2],
                direction[0], direction[1], direction[2],
                length=scale, color=_AXIS_COLORS[j], linewidth=2,
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('predicted cameras (R=X, G=Y, B=Z)')
    _set_equal_aspect(ax)

    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()  # [H, W, 3] uint8
    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return img.clamp(0.0, 1.0)


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


def export_gaussian_glb(gaussian_model, path: str | Path, opacity_threshold: float = 0.0) -> int:
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
