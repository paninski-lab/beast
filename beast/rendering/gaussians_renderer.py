"""Gaussian splatting renderer and 3D Gaussian model utilities."""

import logging
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import videoio
from einops import rearrange
from gsplat import rasterization
from plyfile import PlyData, PlyElement
from torch import nn

_logger = logging.getLogger(__name__)


@torch.no_grad()
def get_turntable_cameras(
    hfov: float = 50,
    num_views: int = 8,
    w: int = 384,
    h: int = 384,
    radius: float = 2.7,
    elevation: float = 20,
    up_vector: np.ndarray | None = None,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Generate a set of turntable camera poses around the origin.

    Parameters
    ----------
    hfov: horizontal field of view in degrees.
    num_views: number of cameras to generate.
    w: image width in pixels.
    h: image height in pixels.
    radius: distance from origin to camera.
    elevation: camera elevation angle in degrees.
    up_vector: world up vector; defaults to [0, 0, 1].

    Returns
    -------
    tuple of (w, h, num_views, fxfycxcy [V, 4], c2ws [V, 4, 4]).

    """
    if up_vector is None:
        up_vector = np.array([0, 0, 1])
    fx = w / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    fxfycxcy = np.array([fx, fy, cx, cy]).reshape(1, 4).repeat(num_views, axis=0)
    azimuths = np.linspace(0, 360, num_views, endpoint=False)
    elevations = np.ones_like(azimuths) * elevation
    c2ws = []
    for elev, azim in zip(elevations, azimuths, strict=True):
        elev, azim = np.deg2rad(elev), np.deg2rad(azim)
        z = radius * np.sin(elev)
        base = radius * np.cos(elev)
        x = base * np.cos(azim)
        y = base * np.sin(azim)
        cam_pos = np.array([x, y, z])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        R = np.stack((right, -up, forward), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)
    return w, h, num_views, fxfycxcy, c2ws


def imageseq2video(images: np.ndarray, filename: str | Path, fps: int = 24) -> None:
    """Save an image sequence to a video file.

    Parameters
    ----------
    images: image array of shape [T, H, W, 3], uint8 or float32.
    filename: output video path.
    fps: frames per second.

    """
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0

    videoio.videosave(filename, images, lossless=True, preset='veryfast', fps=fps)


def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    """Extract lower-diagonal elements of a batch of matrices.

    Parameters
    ----------
    L: input tensor of shape [N, 3, 3].

    Returns
    -------
    uncertainty tensor of shape [N, 6].

    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    """Extract symmetric lower-diagonal elements.

    Parameters
    ----------
    sym: symmetric matrix batch of shape [N, 3, 3].

    Returns
    -------
    lower-diagonal elements of shape [N, 6].

    """
    return strip_lowerdiag(sym)


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices from quaternions.

    Parameters
    ----------
    r: quaternion tensor of shape [N, 4].

    Returns
    -------
    rotation matrices of shape [N, 3, 3].

    """
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build scaling+rotation covariance matrix.

    Parameters
    ----------
    s: scale tensor of shape [N, 3].
    r: rotation quaternion tensor of shape [N, 4].

    Returns
    -------
    covariance factor tensor of shape [N, 3, 3].

    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics at unit directions.

    Uses hardcoded SH polynomials. Works with torch/np/jnp.

    Parameters
    ----------
    deg: int SH degree (0-4 supported).
    sh: SH coefficients of shape [..., C, (deg + 1) ** 2].
    dirs: unit directions of shape [..., 3].

    Returns
    -------
    SH-evaluated values of shape [..., C].

    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to SH coefficients.

    Parameters
    ----------
    rgb: RGB tensor in [0, 1].

    Returns
    -------
    SH coefficient tensor.

    """
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    """Convert SH coefficients to RGB.

    Parameters
    ----------
    sh: SH coefficient tensor.

    Returns
    -------
    RGB tensor.

    """
    return sh * C0 + 0.5


def create_video(
    image_folder: str | Path,
    output_video_file: str | Path,
    framerate: int = 30,
) -> None:
    """Create a video from a folder of PNG images.

    Parameters
    ----------
    image_folder: directory containing PNG images sorted by name.
    output_video_file: output video file path.
    framerate: frames per second.

    """
    image_folder = Path(image_folder)
    images = sorted(p.name for p in image_folder.iterdir() if p.suffix == '.png')

    frame = cv2.imread(str(image_folder / images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        str(output_video_file), cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height),
    )

    for image in images:
        video.write(cv2.imread(str(image_folder / image)))

    cv2.destroyAllWindows()
    video.release()


class Camera(nn.Module):
    """Camera model (OpenCV convention) for Gaussian splatting rendering."""

    def __init__(
        self,
        C2W: torch.Tensor,
        fxfycxcy: torch.Tensor,
        h: int,
        w: int,
    ) -> None:
        """Initialize camera.

        Parameters
        ----------
        C2W: 4x4 camera-to-world matrix (OpenCV convention).
        fxfycxcy: intrinsics vector of length 4.
        h: image height.
        w: image width.

        """
        super().__init__()
        self.C2W = C2W.clone().float()
        self.W2C = self.C2W.inverse()
        self.h = h
        self.w = w

        self.znear = 0.01
        self.zfar = 100.0

        fx, fy, cx, cy = fxfycxcy[0], fxfycxcy[1], fxfycxcy[2], fxfycxcy[3]
        self.tanfovX = w / (2 * fx)
        self.tanfovY = h / (2 * fy)

        def getProjectionMatrix(W, H, fx, fy, cx, cy, znear, zfar):
            """Build an OpenGL-style projection matrix from intrinsics."""
            P = torch.zeros(4, 4, device=fx.device)
            P[0, 0] = 2 * fx / W
            P[1, 1] = 2 * fy / H
            P[0, 2] = 2 * (cx / W) - 1
            P[1, 2] = 2 * (cy / H) - 1
            P[2, 2] = -(zfar + znear) / (zfar - znear)
            P[3, 2] = 1.0
            P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            self.w, self.h, fx, fy, cx, cy, self.znear, self.zfar,
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.C2W[:3, 3]


class GaussianModel:
    """3D Gaussian splat model with per-Gaussian attributes."""

    def setup_functions(self) -> None:
        """Set up activation functions for Gaussian parameters."""
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """Build covariance matrix from scaling and rotation."""
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.inv_scaling_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation

    def __init__(self, sh_degree: int, scaling_modifier: float | None = None) -> None:
        """Initialize with SH degree and optional scaling modifier.

        Parameters
        ----------
        sh_degree: spherical harmonics degree.
        scaling_modifier: optional global scale factor applied at render time.

        """
        self.sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        if self.sh_degree > 0:
            self._features_rest = torch.empty(0)
        else:
            self._features_rest = None
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()

        self.scaling_modifier = scaling_modifier

    def empty(self) -> None:
        """Reset to empty state."""
        self.__init__(self.sh_degree, self.scaling_modifier)

    def set_data(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
    ) -> 'GaussianModel':
        """Set Gaussian data from tensors.

        Parameters
        ----------
        xyz: positions of shape (N, 3).
        features: SH features of shape (N, (sh_degree + 1) ** 2, 3).
        scaling: log-scale values of shape (N, 3).
        rotation: quaternions of shape (N, 4).
        opacity: logit-opacity of shape (N, 1).

        Returns
        -------
        self (for chaining).

        """
        self._xyz = xyz
        self._features_dc = features[:, :1, :].contiguous()
        if self.sh_degree > 0:
            self._features_rest = features[:, 1:, :].contiguous()
        else:
            self._features_rest = None
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        return self

    def to(self, device: str | torch.device) -> 'GaussianModel':
        """Move all tensors to device.

        Parameters
        ----------
        device: target device.

        Returns
        -------
        self (for chaining).

        """
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        if self.sh_degree > 0:
            self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        return self

    def filter(self, valid_mask: torch.Tensor) -> 'GaussianModel':
        """Keep only the Gaussians indicated by valid_mask.

        Parameters
        ----------
        valid_mask: boolean tensor of shape (N,).

        Returns
        -------
        self (for chaining).

        """
        self._xyz = self._xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        if self.sh_degree > 0:
            self._features_rest = self._features_rest[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._rotation = self._rotation[valid_mask]
        self._opacity = self._opacity[valid_mask]
        return self

    def crop(self, crop_bbx: list[float] | None = None) -> 'GaussianModel':
        """Remove Gaussians outside the given bounding box.

        Parameters
        ----------
        crop_bbx: [x_min, x_max, y_min, y_max, z_min, z_max]; defaults to unit cube.

        Returns
        -------
        self (for chaining).

        """
        if crop_bbx is None:
            crop_bbx = [-1, 1, -1, 1, -1, 1]
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbx
        xyz = self._xyz
        invalid_mask = (
            (xyz[:, 0] < x_min)
            | (xyz[:, 0] > x_max)
            | (xyz[:, 1] < y_min)
            | (xyz[:, 1] > y_max)
            | (xyz[:, 2] < z_min)
            | (xyz[:, 2] > z_max)
        )
        valid_mask = ~invalid_mask

        return self.filter(valid_mask)

    def prune(self, opacity_thres: float = 0.05) -> 'GaussianModel':
        """Remove low-opacity Gaussians.

        Parameters
        ----------
        opacity_thres: opacity threshold below which Gaussians are removed.

        Returns
        -------
        self (for chaining).

        """
        opacity = self.get_opacity.squeeze(1)
        valid_mask = opacity > opacity_thres

        return self.filter(valid_mask)

    def prune_by_nearfar(
        self,
        cam_origins: torch.Tensor,
        nearfar_percent: tuple[float, float] = (0.01, 0.99),
    ) -> 'GaussianModel':
        """Remove Gaussians that are too close or too far from any camera.

        Parameters
        ----------
        cam_origins: camera origin positions of shape [V, 3].
        nearfar_percent: (near_pct, far_pct) quantile thresholds in [0, 1].

        Returns
        -------
        self (for chaining).

        """
        assert len(nearfar_percent) == 2
        assert nearfar_percent[0] < nearfar_percent[1]
        assert nearfar_percent[0] >= 0 and nearfar_percent[1] <= 1

        device = self._xyz.device
        dists = torch.cdist(self._xyz[None], cam_origins[None].to(device))[0]
        dists_percentile = torch.quantile(
            dists, torch.tensor(nearfar_percent).to(device), dim=0,
        )
        reject_mask = (dists < dists_percentile[0:1, :]) | (
            dists > dists_percentile[1:2, :]
        )
        reject_mask = reject_mask.any(dim=1)
        valid_mask = ~reject_mask

        return self.filter(valid_mask)

    def apply_all_filters(
        self,
        opacity_thres: float = 0.05,
        crop_bbx: list[float] | None = None,
        cam_origins: torch.Tensor | None = None,
        nearfar_percent: tuple[float, float] = (0.005, 1.0),
    ) -> 'GaussianModel':
        """Apply opacity pruning, bounding-box crop, and near/far pruning.

        Parameters
        ----------
        opacity_thres: opacity threshold for pruning.
        crop_bbx: optional bounding box [x_min, x_max, y_min, y_max, z_min, z_max].
        cam_origins: optional camera origins for near/far pruning.
        nearfar_percent: (near_pct, far_pct) quantile thresholds.

        Returns
        -------
        self (for chaining).

        """
        if crop_bbx is None:
            crop_bbx = [-1, 1, -1, 1, -1, 1]
        self.prune(opacity_thres)
        if crop_bbx is not None:
            self.crop(crop_bbx)
        if cam_origins is not None:
            self.prune_by_nearfar(cam_origins, nearfar_percent)
        return self

    def shrink_bbx(self, drop_ratio: float = 0.05) -> 'GaussianModel':
        """Shrink the bounding box by dropping outlier Gaussians.

        Parameters
        ----------
        drop_ratio: fraction of extreme values to drop from each side.

        Returns
        -------
        self (for chaining).

        """
        xyz = self._xyz
        xyz_min, xyz_max = torch.quantile(
            xyz,
            torch.tensor([drop_ratio, 1 - drop_ratio]).float().to(xyz.device),
            dim=0,
        )
        xyz_min = xyz_min.detach().cpu().numpy()
        xyz_max = xyz_max.detach().cpu().numpy()
        crop_bbx = [
            xyz_min[0],
            xyz_max[0],
            xyz_min[1],
            xyz_max[1],
            xyz_min[2],
            xyz_max[2],
        ]
        _logger.info(f'Shrinking bbx to {crop_bbx}')
        return self.crop(crop_bbx)

    def report_stats(self) -> None:
        """Log statistics for all Gaussian attributes."""
        _logger.info(
            f'xyz: {self._xyz.shape}, {self._xyz.min().item()}, {self._xyz.max().item()}'
        )
        _logger.info(
            f'features_dc: {self._features_dc.shape}, '
            f'{self._features_dc.min().item()}, {self._features_dc.max().item()}'
        )
        if self.sh_degree > 0:
            _logger.info(
                f'features_rest: {self._features_rest.shape}, '
                f'{self._features_rest.min().item()}, {self._features_rest.max().item()}'
            )
        _logger.info(
            f'scaling: {self._scaling.shape}, '
            f'{self._scaling.min().item()}, {self._scaling.max().item()}'
        )
        _logger.info(
            f'rotation: {self._rotation.shape}, '
            f'{self._rotation.min().item()}, {self._rotation.max().item()}'
        )
        _logger.info(
            f'opacity: {self._opacity.shape}, '
            f'{self._opacity.min().item()}, {self._opacity.max().item()}'
        )
        _logger.info(
            f'after activation, xyz: {self.get_xyz.shape}, '
            f'{self.get_xyz.min().item()}, {self.get_xyz.max().item()}'
        )
        _logger.info(
            f'after activation, features: {self.get_features.shape}, '
            f'{self.get_features.min().item()}, {self.get_features.max().item()}'
        )
        _logger.info(
            f'after activation, scaling: {self.get_scaling.shape}, '
            f'{self.get_scaling.min().item()}, {self.get_scaling.max().item()}'
        )
        _logger.info(
            f'after activation, rotation: {self.get_rotation.shape}, '
            f'{self.get_rotation.min().item()}, {self.get_rotation.max().item()}'
        )
        _logger.info(
            f'after activation, opacity: {self.get_opacity.shape}, '
            f'{self.get_opacity.min().item()}, {self.get_opacity.max().item()}'
        )
        _logger.info(
            f'after activation, covariance: {self.get_covariance().shape}, '
            f'{self.get_covariance().min().item()}, {self.get_covariance().max().item()}'
        )

    @property
    def get_scaling(self) -> torch.Tensor:
        """Return activated scaling values."""
        if self.scaling_modifier is not None:
            return self.scaling_activation(self._scaling) * self.scaling_modifier
        else:
            return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        """Return normalised rotation quaternions."""
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self) -> torch.Tensor:
        """Return xyz positions."""
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        """Return all SH feature coefficients."""
        if self.sh_degree > 0:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            return self._features_dc

    @property
    def get_opacity(self) -> torch.Tensor:
        """Return activated opacity values."""
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier: float = 1) -> torch.Tensor:
        """Return covariance matrices from scaling and rotation.

        Parameters
        ----------
        scaling_modifier: global scale multiplier.

        Returns
        -------
        covariance tensor of shape [N, 6].

        """
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation,
        )

    def construct_dtypes(
        self,
        use_fp16: bool = False,
        enable_gs_viewer: bool = True,
    ) -> list[tuple[str, str]]:
        """Build PLY element dtype list for save_ply.

        Parameters
        ----------
        use_fp16: use float16 instead of float32 for PLY storage.
        enable_gs_viewer: pad SH to degree 3 for GS viewer compatibility.

        Returns
        -------
        list of (name, dtype_str) pairs.

        """
        if not use_fp16:
            dtype_list = [
                ('x', 'f4'),
                ('y', 'f4'),
                ('z', 'f4'),
                ('red', 'u1'),
                ('green', 'u1'),
                ('blue', 'u1'),
            ]
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                dtype_list.append((f'f_dc_{i}', 'f4'))

            if enable_gs_viewer:
                assert self.sh_degree <= 3, 'GS viewer only supports SH up to degree 3'
                sh_degree = 3
                for i in range(((sh_degree + 1) ** 2 - 1) * 3):
                    dtype_list.append((f'f_rest_{i}', 'f4'))
            else:
                if self.sh_degree > 0:
                    for i in range(
                        self._features_rest.shape[1] * self._features_rest.shape[2]
                    ):
                        dtype_list.append((f'f_rest_{i}', 'f4'))

            dtype_list.append(('opacity', 'f4'))
            for i in range(self._scaling.shape[1]):
                dtype_list.append((f'scale_{i}', 'f4'))
            for i in range(self._rotation.shape[1]):
                dtype_list.append((f'rot_{i}', 'f4'))
        else:
            dtype_list = [
                ('x', 'f2'),
                ('y', 'f2'),
                ('z', 'f2'),
                ('red', 'u1'),
                ('green', 'u1'),
                ('blue', 'u1'),
            ]
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                dtype_list.append((f'f_dc_{i}', 'f2'))

            if self.sh_degree > 0:
                for i in range(
                    self._features_rest.shape[1] * self._features_rest.shape[2]
                ):
                    dtype_list.append((f'f_rest_{i}', 'f2'))
            dtype_list.append(('opacity', 'f2'))
            for i in range(self._scaling.shape[1]):
                dtype_list.append((f'scale_{i}', 'f2'))
            for i in range(self._rotation.shape[1]):
                dtype_list.append((f'rot_{i}', 'f2'))
        return dtype_list

    def save_ply(
        self,
        path: str | Path,
        use_fp16: bool = False,
        enable_gs_viewer: bool = True,
        color_code: bool = False,
        filter_mask: np.ndarray | None = None,
    ) -> None:
        """Save Gaussians to a PLY file.

        Parameters
        ----------
        path: output file path.
        use_fp16: use float16 storage.
        enable_gs_viewer: pad SH to degree 3 for viewer compatibility.
        color_code: replace RGB with viridis colormap based on index.
        filter_mask: optional boolean mask to select a subset.

        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        if not color_code:
            rgb = (SH2RGB(f_dc) * 255.0).clip(0.0, 255.0).astype(np.uint8)
        else:
            index = np.linspace(0, 1, xyz.shape[0])
            rgb = matplotlib.colormaps['viridis'](index)[..., :3]
            rgb = (rgb * 255.0).clip(0.0, 255.0).astype(np.uint8)

        opacities = self._opacity.detach().cpu().numpy()
        if self.scaling_modifier is not None:
            scale = self.inv_scaling_activation(self.get_scaling).detach().cpu().numpy()
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = self.construct_dtypes(use_fp16, enable_gs_viewer)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        f_rest = None
        if self.sh_degree > 0:
            f_rest = (
                self._features_rest.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )

        if enable_gs_viewer:
            sh_degree = 3
            if f_rest is None:
                f_rest = np.zeros(
                    (xyz.shape[0], 3 * ((sh_degree + 1) ** 2 - 1)), dtype=np.float32,
                )
            elif f_rest.shape[1] < 3 * ((sh_degree + 1) ** 2 - 1):
                f_rest_pad = np.zeros(
                    (xyz.shape[0], 3 * ((sh_degree + 1) ** 2 - 1)), dtype=np.float32,
                )
                f_rest_pad[:, : f_rest.shape[1]] = f_rest
                f_rest = f_rest_pad

        if f_rest is not None:
            attributes = np.concatenate(
                (xyz, rgb, f_dc, f_rest, opacities, scale, rotation), axis=1,
            )
        else:
            attributes = np.concatenate(
                (xyz, rgb, f_dc, opacities, scale, rotation), axis=1,
            )

        if filter_mask is not None:
            attributes = attributes[filter_mask]
            elements = elements[filter_mask]

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path: str | Path) -> None:
        """Load Gaussians from a PLY file.

        Parameters
        ----------
        path: path to the PLY file.

        """
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]['x']),
                np.asarray(plydata.elements[0]['y']),
                np.asarray(plydata.elements[0]['z']),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])

        if self.sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith('f_rest_')
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1)
            )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith('scale_')
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith('rot')
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.from_numpy(xyz.astype(np.float32))
        self._features_dc = (
            torch.from_numpy(features_dc.astype(np.float32))
            .transpose(1, 2)
            .contiguous()
        )
        if self.sh_degree > 0:
            self._features_rest = (
                torch.from_numpy(features_extra.astype(np.float32))
                .transpose(1, 2)
                .contiguous()
            )
        self._opacity = torch.from_numpy(
            np.copy(opacities).astype(np.float32)
        ).contiguous()
        self._scaling = torch.from_numpy(scales.astype(np.float32)).contiguous()
        self._rotation = torch.from_numpy(rots.astype(np.float32)).contiguous()


def render_opencv_cam_gsplat(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: torch.Tensor,
    fxfycxcy: torch.Tensor,
    sh_degree: int | None = None,
    near_plane: float = 0.2,
    bg_color: tuple[float, float, float] | torch.Tensor = (1.0, 1.0, 1.0),
    render_depth: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Render a batch of views using gsplat rasterisation.

    Parameters
    ----------
    pc: Gaussian point cloud model.
    height: render height in pixels.
    width: render width in pixels.
    C2W: camera-to-world matrices of shape [V, 4, 4].
    fxfycxcy: intrinsics of shape [V, 4].
    sh_degree: SH degree to evaluate; defaults to pc.sh_degree.
    near_plane: near clipping plane distance.
    bg_color: background colour (1-D or 2-D tensor or tuple).
    render_depth: whether to also render a depth map.

    Returns
    -------
    dict with keys 'render' ([V, 3, H, W]) and 'depth' ([V, 1, H, W] or None).

    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    for name, t in [('xyz', means3D), ('scales', scales), ('rotation', rotations)]:
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError(
                f'Gaussian {name} contains NaN or Inf. '
                'Check model outputs and consider using float32 or clamping scales.'
            )
    num_cams = C2W.size(0)
    if torch.is_tensor(bg_color):
        bg_color = bg_color.to(device=C2W.device, dtype=torch.float32)
    else:
        bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)
    if bg_color.ndim == 1:
        bg_color = bg_color.unsqueeze(0).expand(num_cams, -1)
    elif bg_color.ndim == 2 and bg_color.size(0) == 1:
        bg_color = bg_color.expand(num_cams, -1)
    elif bg_color.ndim == 2 and bg_color.size(0) == num_cams:
        pass
    else:
        raise ValueError(
            f'Invalid bg_color shape {tuple(bg_color.shape)}; '
            f'expected [3], [1,3], or [{num_cams},3].'
        )
    W2C = C2W.inverse()

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(
            f'Invalid render dimensions: width={width}, height={height}. '
            'Check that input images have valid (H, W) from data.'
        )

    intr = torch.zeros(fxfycxcy.size(0), 3, 3, device=fxfycxcy.device, dtype=fxfycxcy.dtype)
    intr[:, 0, 0] = fxfycxcy[:, 0]
    intr[:, 1, 1] = fxfycxcy[:, 1]
    intr[:, 0, 2] = fxfycxcy[:, 2]
    intr[:, 1, 2] = fxfycxcy[:, 3]
    intr[:, 2, 2] = 1.0

    render_mode = 'RGB+ED' if render_depth else 'RGB'
    render_colors, _, _ = rasterization(
        means3D, rotations, scales, opacity.squeeze(),
        shs, W2C, intr, width, height,
        near_plane=near_plane,
        sh_degree=sh_degree,
        backgrounds=bg_color,
        render_mode=render_mode,
        packed=False,
    )
    if render_mode != 'RGB':
        render_colors, render_depth_map = render_colors[..., :3], render_colors[..., 3:]
    else:
        render_depth_map = None
    return {
        'render': render_colors.permute(0, 3, 1, 2),
        'depth': (
            render_depth_map.permute(0, 3, 1, 2) if torch.is_tensor(render_depth_map) else None
        ),
    }


def render_opencv_cam(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: torch.Tensor,
    fxfycxcy: torch.Tensor,
    sh_degree: int | None = None,
    near_plane: float = 0.2,
    bg_color: tuple[float, float, float] | torch.Tensor = (1.0, 1.0, 1.0),
    render_depth: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Compatibility wrapper for legacy single-camera render paths.

    Routes calls through the batched gsplat implementation.

    Parameters
    ----------
    pc: Gaussian point cloud model.
    height: render height in pixels.
    width: render width in pixels.
    C2W: camera-to-world matrix of shape [4, 4] or [1, 4, 4].
    fxfycxcy: intrinsics of shape [4] or [1, 4].
    sh_degree: SH degree to evaluate; defaults to pc.sh_degree.
    near_plane: near clipping plane distance.
    bg_color: background colour.
    render_depth: whether to also render a depth map.

    Returns
    -------
    dict with keys 'render', 'depth', 'alpha'.

    """
    squeeze_view = C2W.ndim == 2
    if squeeze_view:
        C2W = C2W.unsqueeze(0)
    if fxfycxcy.ndim == 1:
        fxfycxcy = fxfycxcy.unsqueeze(0)

    if sh_degree is None:
        sh_degree = pc.sh_degree

    buffers = render_opencv_cam_gsplat(
        pc,
        height,
        width,
        C2W,
        fxfycxcy,
        sh_degree=sh_degree,
        near_plane=near_plane,
        bg_color=bg_color,
        render_depth=render_depth,
    )
    if squeeze_view:
        buffers['render'] = buffers['render'][0]
        if torch.is_tensor(buffers.get('depth')):
            buffers['depth'] = buffers['depth'][0]
    buffers.setdefault('alpha', None)
    return buffers


class DeferredGaussianRender(torch.autograd.Function):
    """Custom autograd function for deferred Gaussian rendering."""

    @staticmethod
    def forward(
        ctx,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        scaling_modifier=None,
    ):
        """Forward pass: render all views and return image tensor.

        Parameters
        ----------
        ctx: autograd context.
        xyz: Gaussian positions of shape [b, n_gaussians, 3].
        features: SH features of shape [b, n_gaussians, (sh_degree+1)^2, 3].
        scaling: log-scale of shape [b, n_gaussians, 3].
        rotation: quaternions of shape [b, n_gaussians, 4].
        opacity: logit-opacity of shape [b, n_gaussians, 1].
        height: render height.
        width: render width.
        C2W: camera-to-world matrices of shape [b, v, 4, 4].
        fxfycxcy: intrinsics of shape [b, v, 4].
        scaling_modifier: optional global scaling factor.

        Returns
        -------
        renders of shape [b, v, 3, height, width].

        """
        ctx.scaling_modifier = scaling_modifier

        sh_degree = int(math.sqrt(features.shape[-2])) - 1

        gaussians_model = GaussianModel(sh_degree, scaling_modifier)

        with torch.no_grad():
            b, v = C2W.size(0), C2W.size(1)
            renders = []
            for i in range(b):
                pc = gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i],
                )
                for j in range(v):
                    renders.append(
                        render_opencv_cam(pc, height, width, C2W[i, j], fxfycxcy[i, j])[
                            'render'
                        ]
                    )
            renders = torch.stack(renders, dim=0)
            renders = renders.reshape(b, v, 3, height, width)

        renders = renders.requires_grad_()

        ctx.save_for_backward(xyz, features, scaling, rotation, opacity, C2W, fxfycxcy)
        ctx.rendering_size = (height, width)
        ctx.sh_degree = sh_degree

        del gaussians_model

        return renders

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: compute gradients w.r.t. Gaussian parameters.

        Parameters
        ----------
        ctx: autograd context.
        grad_output: upstream gradient tensor.

        Returns
        -------
        gradients w.r.t. xyz, features, scaling, rotation, opacity, and None for
        height, width, C2W, fxfycxcy, scaling_modifier.

        """
        xyz, features, scaling, rotation, opacity, C2W, fxfycxcy = ctx.saved_tensors
        height, width = ctx.rendering_size
        sh_degree = ctx.sh_degree

        input_dict = OrderedDict(
            [
                ('xyz', xyz),
                ('features', features),
                ('scaling', scaling),
                ('rotation', rotation),
                ('opacity', opacity),
            ]
        )
        input_dict = {k: v.detach().requires_grad_() for k, v in input_dict.items()}

        gaussians_model = GaussianModel(sh_degree, ctx.scaling_modifier)

        with torch.enable_grad():
            b, v = C2W.size(0), C2W.size(1)
            for i in range(b):
                for j in range(v):
                    pc = gaussians_model.set_data(
                        **{k: v[i] for k, v in input_dict.items()}
                    )

                    render = render_opencv_cam(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j],
                    )['render']

                    render.backward(grad_output[i, j])

        del gaussians_model

        return *[var.grad for var in input_dict.values()], None, None, None, None, None


deferred_gaussian_render = DeferredGaussianRender.apply


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def render_turntable(
    pc: GaussianModel,
    rendering_resolution: int = 384,
    num_views: int = 8,
) -> np.ndarray:
    """Render a turntable sequence around the Gaussian cloud.

    Parameters
    ----------
    pc: Gaussian point cloud model.
    rendering_resolution: square render resolution in pixels.
    num_views: number of equally-spaced viewpoints.

    Returns
    -------
    uint8 numpy array of shape [H, V*W, 3].

    """
    w, h, v, fxfycxcy, c2w = get_turntable_cameras(
        h=rendering_resolution, w=rendering_resolution, num_views=num_views,
    )

    device = pc._xyz.device
    fxfycxcy = torch.from_numpy(fxfycxcy).float().to(device)
    c2w = torch.from_numpy(c2w).float().to(device)

    renderings = torch.zeros(v, 3, h, w, dtype=torch.float32, device=device)
    for j in range(v):
        renderings[j] = render_opencv_cam(pc, h, w, c2w[j], fxfycxcy[j])['render']
    torch.cuda.empty_cache()
    renderings = renderings.detach().cpu().numpy()
    renderings = (renderings * 255).clip(0, 255).astype(np.uint8)
    renderings = rearrange(renderings, 'v c h w -> h (v w) c')
    return renderings


class GaussianRenderer(nn.Module):
    """Batched Gaussian splat renderer with optional deferred rendering."""

    def __init__(
        self,
        sh_degree: int,
        scaling_modifier: float | None = None,
    ) -> None:
        """Initialize with SH degree and optional scaling modifier.

        Parameters
        ----------
        sh_degree: spherical harmonics degree.
        scaling_modifier: optional global scale factor applied at render time.

        """
        super().__init__()

        self.sh_degree = sh_degree
        self.scaling_modifier = scaling_modifier

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        deferred=True,
    ):
        """Render Gaussians for all batch items and views.

        Parameters
        ----------
        xyz: [b, n_gaussians, 3].
        features: [b, n_gaussians, (sh_degree+1)^2, 3].
        scaling: [b, n_gaussians, 3].
        rotation: [b, n_gaussians, 4].
        opacity: [b, n_gaussians, 1].
        height: render height.
        width: render width.
        C2W: [b, v, 4, 4].
        fxfycxcy: [b, v, 4].
        deferred: use deferred rendering (default True).

        Returns
        -------
        tuple of (renderings [b, v, 3, H, W], list of GaussianModel per batch item).

        """
        if deferred:
            renderings = deferred_gaussian_render(
                xyz,
                features,
                scaling,
                rotation,
                opacity,
                height,
                width,
                C2W,
                fxfycxcy,
                self.scaling_modifier,
            )

            b, v = C2W.size(0), C2W.size(1)
            gaussians_models = []
            for i in range(b):
                pc = GaussianModel(self.sh_degree, self.scaling_modifier)
                pc = pc.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i],
                )
                gaussians_models.append(pc)
        else:
            b, v = C2W.size(0), C2W.size(1)
            renderings = torch.zeros(
                b, v, 3, height, width, dtype=torch.float32, device=xyz.device,
            )

            gaussians_models = []
            for i in range(b):
                pc = GaussianModel(self.sh_degree, self.scaling_modifier)
                pc = pc.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i],
                )
                gaussians_models.append(pc)
                for j in range(v):
                    renderings[i, j] = render_opencv_cam(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j],
                    )['render']
        return renderings, gaussians_models


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def render_generic(
    pc: GaussianModel,
    c2ws: torch.Tensor,
    fxfycxcy: torch.Tensor,
    h: int = 512,
    w: int = 512,
) -> np.ndarray:
    """Render a sequence of views and return uint8 numpy images.

    Parameters
    ----------
    pc: Gaussian point cloud model.
    c2ws: camera-to-world matrices of shape [v, 4, 4].
    fxfycxcy: intrinsics of shape [v, 4].
    h: render height.
    w: render width.

    Returns
    -------
    uint8 numpy array of shape [v, H, W, 3].

    """
    v = c2ws.shape[0]
    device = pc._xyz.device
    fxfycxcy = fxfycxcy.float().to(device)
    renderings = torch.zeros(v, 3, h, w, dtype=torch.float32, device=device)
    for j in range(v):
        renderings[j] = render_opencv_cam(pc, h, w, c2ws[j], fxfycxcy[j])['render']
    torch.cuda.empty_cache()
    renderings = renderings.detach().cpu().numpy()
    renderings = (renderings * 255).clip(0, 255).astype(np.uint8)
    renderings = rearrange(renderings, 'v c h w -> v h w c')
    return renderings
