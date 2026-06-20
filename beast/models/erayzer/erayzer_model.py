"""ERayZer model: 3DGS renderer, loss computer, and Lightning model.

ERayZer is the base class that includes a camera-prediction branch
(transformer_encoder + PoseEstimator).  Subclasses can override
``_resolve_cameras`` to substitute a different camera source:

- ``BEAST3D``: drops the prediction modules and reads GT cameras from the
  batch dict.
- ``SABLE`` (future): inherits prediction and adds depth/pointcloud
  initialization.
"""

import copy
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.modules.module import _IncompatibleKeys

from beast.geometry.camera import cam_info_to_plucker, get_interpolated_poses_many
from beast.geometry.positional_encoding import get_2d_sincos_pos_embed
from beast.geometry.rotations import quat2mat, rot6d2mat
from beast.losses import masked_mse_loss
from beast.models.base import BaseLightningModel
from beast.models.erayzer.visualize import (
    camera_intrinsic_stats,
    export_gaussian_glb,
    make_camera_pose_image,
    make_render_grid,
    viz_is_due,
)
from beast.nn.perceptual import VGGPerceptual
from beast.nn.transformer import (
    QK_Norm_TransformerBlock,
    _init_weights,
    _init_weights_layerwise,
)
from beast.rendering.gaussians_renderer import (
    GaussianModel,
    deferred_gaussian_render,
    render_opencv_cam_gsplat,
)

_logger = logging.getLogger(__name__)

_INIT_CKPT_CONTAINER_KEYS = ('state_dict', 'model', 'model_state_dict')
_INIT_CKPT_DROP_PREFIXES = ('loss_computer.',)


def _filter_init_state_dict(
    raw: dict,
    drop_prefixes: tuple[str, ...] = _INIT_CKPT_DROP_PREFIXES,
) -> dict:
    """Extract a model weight dict from a raw checkpoint and drop ignored keys.

    Handles both bare state dicts and checkpoints that nest the weights under a
    container key (``state_dict``, ``model``, or ``model_state_dict``). Keys
    beginning with any of ``drop_prefixes`` are removed; the default drops the
    perceptual loss network (``loss_computer.*``), which is not part of the
    renderable model and is rebuilt from torchvision on demand.

    Parameters
    ----------
    raw: object returned by ``torch.load`` on a checkpoint file.
    drop_prefixes: key prefixes to exclude from the returned dict.

    Returns
    -------
    Flat ``{name: tensor}`` state dict ready for ``load_state_dict``.

    """
    state_dict = raw
    for key in _INIT_CKPT_CONTAINER_KEYS:
        if isinstance(raw, dict) and isinstance(raw.get(key), dict):
            state_dict = raw[key]
            break
    return {
        name: tensor
        for name, tensor in state_dict.items()
        if not any(name.startswith(prefix) for prefix in drop_prefixes)
    }


def sanitize(t: torch.Tensor) -> torch.Tensor:
    """Replace non-finite entries so downstream losses stay valid.

    Parameters
    ----------
    t: input tensor.

    Returns
    -------
    tensor with NaN → 0, ±Inf → ±1e6, clamped to [-1e6, 1e6].

    """
    _safe = 1e6
    return torch.nan_to_num(t, nan=0.0, posinf=_safe, neginf=-_safe).clamp(-_safe, _safe)


def build_transformer_blocks(
    num_layers: int,
    d: int,
    d_head: int,
    use_qk_norm: bool,
    special_init: bool = False,
    depth_init: bool = False,
) -> nn.ModuleList:
    """Build a ModuleList of QK_Norm_TransformerBlock layers.

    Parameters
    ----------
    num_layers: number of transformer layers.
    d: embedding dimension.
    d_head: per-head dimension.
    use_qk_norm: whether to apply QK normalization.
    special_init: if True, apply layerwise weight scaling.
    depth_init: if True, scale std by layer depth rather than total layers.

    Returns
    -------
    initialized ModuleList of transformer blocks.

    """
    layers = [
        QK_Norm_TransformerBlock(d, d_head, use_qk_norm=use_qk_norm)
        for _ in range(num_layers)
    ]
    if special_init:
        for idx, layer in enumerate(layers):
            std = 0.02 / (2 * (idx + 1)) ** 0.5 if depth_init else 0.02 / (2 * num_layers) ** 0.5
            layer.apply(lambda module, s=std: _init_weights_layerwise(module, s))
    return nn.ModuleList(layers)


# ---------------------------------------------------------------------------
# Camera-prediction helpers
# ---------------------------------------------------------------------------

def get_cam_se3(
    cam_info: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode camera tokens into c2w matrices and fxfycxcy intrinsics.

    Parameters
    ----------
    cam_info: predicted camera parameters of shape [B, D] where D is either
        13 (6D rotation + 3D translation + 4 intrinsics) or
        11 (quaternion rotation + 3D translation + 4 intrinsics).

    Returns
    -------
    tuple of (c2w [B, 4, 4], fxfycxcy [B, 4]).

    Raises
    ------
    NotImplementedError
        if D is neither 11 nor 13.

    """
    b, n = cam_info.shape
    if n == 13:
        R = rot6d2mat(cam_info[:, :6])
        t = cam_info[:, 6:9].unsqueeze(-1)
        fxfycxcy = cam_info[:, 9:]
    elif n == 11:
        R = quat2mat(cam_info[:, :4])
        t = cam_info[:, 4:7].unsqueeze(-1)
        fxfycxcy = cam_info[:, 7:]
    else:
        raise NotImplementedError(f'cam_info width {n} is neither 11 nor 13')

    bottom = torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
    bottom = bottom.view(1, 1, 4).expand(b, -1, -1)
    c2w = torch.cat([torch.cat([R, t], dim=2), bottom], dim=1)
    return c2w, fxfycxcy


class PoseEstimator(nn.Module):
    """Predict per-view SE3 poses and focal lengths from camera tokens.

    Supports three canonical modes (``first``, ``middle``, ``unordered``) and
    two pose parameterizations (``6d`` continuous rotation, ``quat``ernion).
    """

    _FOCAL_BIAS = 1.25  # default predicted focal length offset (normalized)

    def __init__(self, config: dict) -> None:
        """Initialize PoseEstimator.

        Parameters
        ----------
        config: full model config dict; reads config['model']['pose_latent']
            for canonical, mode, and representation settings.

        """
        super().__init__()
        self.config = config
        pose_cfg = config['model']['pose_latent']
        self.canonical = pose_cfg.get('canonical', 'first')
        if self.canonical not in ('first', 'middle', 'unordered'):
            raise ValueError(f'Unknown canonical mode: {self.canonical!r}')
        self.is_pairwise = pose_cfg.get('mode', 'pairwise') == 'pairwise'
        # per_view_focal=False (default) predicts one focal from the canonical
        # view and broadcasts it (matches the pretrained ERayZer checkpoint);
        # True applies the focal head independently to every view
        self.per_view_focal = pose_cfg.get('per_view_focal', False)
        self.pose_consistency_reg_weight = config['training'].get(
            'pose_consistency_reg_weight', 0.0,
        )
        self.pose_rep = pose_cfg.get('representation', '6d')
        if self.pose_rep == '6d':
            self.num_pose_element = 6
        elif self.pose_rep == 'quat':
            self.num_pose_element = 4
        else:
            raise NotImplementedError(f'Unknown pose representation: {self.pose_rep!r}')

        d = config['model']['transformer']['d']
        rel_head_in = d * 2 if self.is_pairwise else d
        self.rel_head = nn.Sequential(
            nn.Linear(rel_head_in, d, bias=True),
            nn.SiLU(),
            nn.Linear(d, self.num_pose_element + 3, bias=True),
        )
        self.rel_head.apply(_init_weights)

        self.canonical_k_head = nn.Sequential(
            nn.Linear(d, d, bias=True),
            nn.SiLU(),
            nn.Linear(d, 1, bias=False),
        )
        self.canonical_k_head.apply(_init_weights)

    def forward(
        self,
        x: torch.Tensor,
        v: int,
    ) -> torch.Tensor:
        """Predict camera parameters for all views.

        Parameters
        ----------
        x: per-view camera tokens of shape [B*V, D] or [B, V, D].
        v: number of views V.

        Returns
        -------
        camera parameter tensor of shape [B*V, pose_dim] where pose_dim is
        either 13 (6d) or 11 (quat).

        """
        return_flat = x.ndim == 2
        if return_flat:
            x = rearrange(x, '(b v) d -> b v d', v=v)

        if self.is_pairwise:
            if v == 1:
                out = self._forward_single_view(x, v)
            else:
                out = self._forward_canonical(x, v, self.canonical)
        else:
            out = self._forward_global(x, v, self.canonical)

        return rearrange(out, 'b v d -> (b v) d') if return_flat else out

    # ------------------------------------------------------------------
    # Private forward variants
    # ------------------------------------------------------------------

    def _identity_rt(
        self,
        b: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return an identity rotation+translation canonical tensor [B, 1, num_pose_element+3]."""
        if self.pose_rep == '6d':
            # first two columns of I₃ then zero translation
            vals = [1, 0, 0, 0, 1, 0, 0, 0, 0]
        else:
            # w=1, x=y=z=0 quaternion, then zero translation
            vals = [1, 0, 0, 0, 0, 0, 0]
        return torch.tensor(vals, device=device, dtype=dtype).reshape(1, 1, -1).expand(b, 1, -1)

    def _predict_fxfy(
        self,
        x: torch.Tensor,
        v: int,
        cano_idx: int = 0,
    ) -> torch.Tensor:
        """Predict focal lengths from camera tokens.

        With ``per_view_focal`` the focal head is applied independently to every
        view, so each camera gets its own focal length. Otherwise the head is
        applied to the canonical view only and broadcast to all views, matching
        how the pretrained ERayZer checkpoint was trained.

        Parameters
        ----------
        x: per-view camera tokens of shape [B, V, D].
        v: number of views to broadcast to.
        cano_idx: index of the canonical view (used when not per-view).

        Returns
        -------
        fxfy tensor of shape [B, V, 2] (fx == fy per view).

        """
        if self.per_view_focal:
            fxfy = self.canonical_k_head(x) + self._FOCAL_BIAS    # [B, V', 1]
            if fxfy.shape[1] == 1 and v > 1:
                fxfy = fxfy.expand(-1, v, -1)                     # broadcast single view
        else:
            x_cano = x[:, cano_idx:cano_idx + 1]                  # [B, 1, D]
            fxfy = self.canonical_k_head(x_cano) + self._FOCAL_BIAS  # [B, 1, 1]
            fxfy = fxfy.expand(-1, v, -1)                         # [B, V, 1]
        return fxfy.expand(-1, -1, 2)                            # [B, V, 2]

    def _append_cxcy(self, info: torch.Tensor) -> torch.Tensor:
        """Append fixed cx=0.5, cy=0.5 principal point to all views."""
        b, v = info.shape[:2]
        cxcy = info.new_tensor([0.5, 0.5]).reshape(1, 1, 2).expand(b, v, -1)
        return torch.cat([info, cxcy], dim=-1)

    def _forward_canonical(
        self,
        x: torch.Tensor,
        v: int,
        canonical: str,
    ) -> torch.Tensor:
        """Canonical-based pairwise prediction (first or middle view as anchor)."""
        b = x.shape[0]
        device, dtype = x.device, x.dtype

        if canonical == 'first':
            cano_idx = 0
            rel_indices = torch.arange(1, v, device=device)
        elif canonical == 'middle':
            cano_idx = v // 2
            rel_indices = torch.cat([
                torch.arange(cano_idx, device=device),
                torch.arange(cano_idx + 1, v, device=device),
            ])
        else:
            raise NotImplementedError

        x_canonical = x[:, cano_idx:cano_idx + 1]  # [B, 1, D]
        x_rel = x[:, rel_indices]                    # [B, V-1, D]

        # identity canonical pose + fxfy (per-view or canonical-broadcast)
        rt_cano = self._identity_rt(b, device, dtype)                    # [B, 1, num_pose+3]
        fxfy = self._predict_fxfy(x, v, cano_idx)                        # [B, V, 2]
        info = torch.cat([rt_cano.expand(b, v, -1), fxfy], dim=-1)       # [B, V, num_pose+3+2]

        # relative pose residuals
        feat_rel = torch.cat([x_canonical.expand(-1, v - 1, -1), x_rel], dim=-1)
        residuals = self.rel_head(feat_rel)                               # [B, V-1, num_pose+3]
        info[:, rel_indices, :self.num_pose_element + 3] = (
            info[:, rel_indices, :self.num_pose_element + 3] + residuals
        )

        return self._append_cxcy(info)

    def _forward_single_view(self, x: torch.Tensor, v: int) -> torch.Tensor:
        """Degenerate single-view case: identity pose broadcast to all views."""
        b = x.shape[0]
        rt_cano = self._identity_rt(b, x.device, x.dtype)
        fxfy = self._predict_fxfy(x, v, cano_idx=0)
        info = torch.cat([rt_cano.expand(b, v, -1), fxfy], dim=-1)
        return self._append_cxcy(info)

    def _forward_global(
        self,
        x: torch.Tensor,
        v: int,
        canonical: str,
    ) -> torch.Tensor:
        """Global (non-pairwise) prediction: each view gets its own pose offset from identity."""
        b = x.shape[0]
        device, dtype = x.device, x.dtype

        rt_canonical = self._identity_rt(b, device, dtype).expand(b, v, -1)
        residuals = self.rel_head(x)  # [B, V, num_pose+3]

        if canonical == 'unordered':
            cano_idx = 0
            rt_final = rt_canonical + residuals
        else:
            if canonical == 'first':
                cano_idx = 0
            elif canonical == 'middle':
                cano_idx = v // 2
            else:
                raise ValueError(f'Unknown canonical mode: {canonical!r}')
            mask = torch.ones(1, v, 1, device=device)
            mask[:, cano_idx, :] = 0.0
            rt_final = rt_canonical + residuals * mask

        fxfy = self._predict_fxfy(x, v, cano_idx)                        # [B, V, 2]
        info = torch.cat([rt_final, fxfy], dim=-1)
        return self._append_cxcy(info)


# ---------------------------------------------------------------------------
# Gaussian splatting helpers
# ---------------------------------------------------------------------------

class GaussiansUpsampler(nn.Module):
    """Project token features to per-Gaussian attributes."""

    def __init__(self, config: dict) -> None:
        """Initialize.

        Parameters
        ----------
        config: full model config dict.

        """
        super().__init__()
        self.config = config
        self.scaling_bias = config['model'].get('scaling_bias', -2.3)
        self.scaling_max = config['model'].get('scaling_max', -1.2)
        self.opacity_bias = config['model'].get('opacity_bias', -2.0)

    def to_gs(
        self,
        gaussians: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split and activate raw Gaussian attribute predictions.

        Parameters
        ----------
        gaussians: raw feature tensor of shape [B, N, d_out].

        Returns
        -------
        tuple of (xyz, features, scaling, rotation, opacity) tensors.

        """
        sh_degree = self.config['model']['gaussians']['sh_degree']
        n_sh = (sh_degree + 1) ** 2 * 3
        xyz, features, scaling, rotation, opacity = gaussians.split(
            [3, n_sh, 3, 4, 1], dim=2,
        )

        if not self.config['model']['hard_pixelalign']:
            xyz = xyz.clamp(-500.0, 500.0)

        features = features.reshape(
            features.size(0),
            features.size(1),
            (sh_degree + 1) ** 2,
            3,
        )

        scaling = (scaling + self.scaling_bias).clamp(max=self.scaling_max).clamp(min=-10.0)
        opacity = (opacity + self.opacity_bias).clamp(min=-10.0)
        return xyz, features, scaling, rotation, opacity


class Renderer(nn.Module):
    """Batched Gaussian splat renderer using config-driven dispatch."""

    def __init__(self, config: dict) -> None:
        """Initialize renderer from model config.

        Parameters
        ----------
        config: full model config dict; reads config['model']['gaussians'].

        """
        super().__init__()
        self.config = config
        self.sh_degree = config['model']['gaussians']['sh_degree']
        self.gaussians_model = GaussianModel(config['model']['gaussians']['sh_degree'])

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
    ):
        """Render Gaussians for all batch items and views.

        Parameters
        ----------
        xyz: [b, n_gaussians, 3].
        features: [b, n_gaussians, (sh_degree+1)^2, 3].
        scaling: [b, n_gaussians, 3].
        rotation: [b, n_gaussians, 4].
        opacity: [b, n_gaussians, 1].
        height: render height in pixels.
        width: render width in pixels.
        C2W: camera-to-world matrices of shape [b, v, 4, 4].
        fxfycxcy: intrinsics of shape [b, v, 4].

        Returns
        -------
        SimpleNamespace with fields render, depth, alpha.

        """
        if self.config['model'].get('use_deferred_rendering', False):
            renderings = deferred_gaussian_render(
                xyz, features, scaling, rotation, opacity, height, width, C2W, fxfycxcy,
            )
            b, v = C2W.size(0), C2W.size(1)
            depth = torch.zeros(b, v, 1, height, width, dtype=torch.float32, device=xyz.device)
            alpha = torch.zeros(b, v, 1, height, width, dtype=torch.float32, device=xyz.device)
        else:
            b, v = C2W.size(0), C2W.size(1)
            kw = {'dtype': torch.float32, 'device': xyz.device}
            renderings = torch.zeros(b, v, 3, height, width, **kw)
            depth = torch.zeros(b, v, 1, height, width, **kw)
            alpha = torch.zeros(b, v, 1, height, width, **kw)

            for i in range(b):
                pc = self.gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i],
                )
                near_plane = self.config['model'].get('near_plane', 0.2)
                buffers = render_opencv_cam_gsplat(
                    pc, height, width, C2W[i], fxfycxcy[i], self.sh_degree,
                    near_plane=near_plane,
                )
                renderings[i] = buffers['render']
                if 'depth' in buffers and buffers['depth'] is not None:
                    depth[i] = buffers['depth']
                if 'alpha' in buffers and buffers['alpha'] is not None:
                    alpha[i] = buffers['alpha']

        return SimpleNamespace(render=renderings, depth=depth, alpha=alpha)


def get_point_range_func(gaussians_config: dict):
    """Return a depth range mapping function based on the gaussians config.

    Parameters
    ----------
    gaussians_config: config dict for the gaussians sub-section.

    Returns
    -------
    callable mapping raw network output to depth values.

    Raises
    ------
    NotImplementedError
        if range_setting type is unrecognised.

    """
    range_setting = gaussians_config.get('range_setting', {'type': 'object_centric_depth'})
    if range_setting['type'] == 'object_centric_depth':
        def rangefunc(t):
            return (2.0 * torch.sigmoid(t) - 1.0) * 1.5 + 2.7
        return rangefunc
    elif range_setting['type'] == 'linear_depth':
        near = range_setting.get('near', 0.0)
        far = range_setting.get('far', 500.0)
        def rangefunc(t):
            return torch.sigmoid(t) * (far - near) + near
        return rangefunc
    elif range_setting['type'] == 'log_depth':
        near = range_setting.get('near', -6.2)
        far = range_setting.get('far', 6.2)
        def rangefunc(t):
            return torch.exp(torch.sigmoid(t) * (far - near) + near)
        return rangefunc
    elif range_setting['type'] == 'disparity':
        near = range_setting.get('near', 0.1)
        far = range_setting.get('far', 500.0)
        def rangefunc(t):
            return 1.0 / (torch.sigmoid(t) * (1.0 / near - 1.0 / far) + 1.0 / far)
        return rangefunc
    else:
        raise NotImplementedError(f'Unknown range_setting type: {range_setting["type"]}')


class LossComputer(nn.Module):
    """Composite render loss: L2 + gs_reg + optional LPIPS + optional perceptual."""

    def __init__(self, config: dict, device: str = 'cpu') -> None:
        """Initialize.

        Parameters
        ----------
        config: full training config dict; reads config['training'] for loss weights.
        device: device to place loss modules on.

        """
        super().__init__()
        self.config = config
        self.device = device

        self.perceptual_loss_module = None
        if self.config['training'].get('perceptual_loss_weight', 0.0) > 0.0:
            self.perceptual_loss_module = VGGPerceptual(device).eval()

    def forward(
        self,
        rendering: torch.Tensor,
        target: torch.Tensor,
        xyz_norm: torch.Tensor | None,
        xyz_init_norm: torch.Tensor | None,
        pixel_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        """Compute composite render loss.

        Parameters
        ----------
        rendering: predicted images of shape [B, V, 3, H, W].
        target: ground-truth images of shape [B, V, 3 or 4, H, W].
        xyz_norm: normalized xyz coordinates of shape [B, N, 3] or None.
        xyz_init_norm: normalized initial xyz coordinates of shape [B, N, 3] or None.
        pixel_mask: optional foreground mask of shape [B, V, H, W] or [B*V, 1, H, W].

        Returns
        -------
        SimpleNamespace with fields: loss, l2_loss, psnr, gs_reg_loss, perceptual_loss.

        """
        batch_size, num_views, _, height, width = rendering.shape
        rendering = rendering.reshape(batch_size * num_views, 3, height, width)
        target = target.reshape(batch_size * num_views, target.shape[2], height, width)

        if target.shape[1] == 4:
            target = target[:, :3]

        if pixel_mask is not None:
            pixel_mask = pixel_mask.reshape(batch_size * num_views, 1, height, width)
            l2_loss = masked_mse_loss(rendering, target, pixel_mask)
        else:
            l2_loss = F.mse_loss(rendering, target)

        gs_reg_loss = (
            F.mse_loss(xyz_norm, xyz_init_norm)
            if xyz_norm is not None and xyz_init_norm is not None
            else rendering.new_zeros(())
        )

        psnr = -10.0 * torch.log10(l2_loss.clamp_min(1e-8))

        perceptual_loss = rendering.new_zeros(())
        if self.perceptual_loss_module is not None:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        total_loss = (
            self.config['training'].get('l2_loss_weight', 1.0) * l2_loss
            + self.config['training'].get('gs_reg_loss_weight', 0.0) * gs_reg_loss
            + self.config['training'].get('perceptual_loss_weight', 0.0) * perceptual_loss
        )

        return SimpleNamespace(
            loss=total_loss,
            l2_loss=l2_loss,
            gs_reg_loss=gs_reg_loss,
            psnr=psnr,
            perceptual_loss=perceptual_loss,
        )


# ---------------------------------------------------------------------------
# ERayZer base model
# ---------------------------------------------------------------------------

class ERayZer(BaseLightningModel):
    """ERayZer: encode multi-view images into 3D Gaussians and render.

    Camera parameters are obtained via ``_resolve_cameras``, which by default
    runs a learned pose-prediction branch (transformer_encoder + PoseEstimator).
    Subclasses override ``_resolve_cameras`` to use a different source (e.g.
    ground-truth cameras in ``BEAST3D``).

    The camera prediction branch is only constructed when
    ``config['model']['transformer']['encoder_n_layer'] > 0``.
    """

    _NUM_REGISTER_TOKENS = 4

    def __init__(self, config: dict) -> None:
        """Initialize ERayZer model architecture.

        Parameters
        ----------
        config: full training/model config dict.

        """
        super().__init__(config)

        self.d = config['model']['transformer']['d']
        self.d_head = config['model']['transformer']['d_head']
        img_size = config['model']['image_tokenizer']['image_size']
        patch_size = config['model']['image_tokenizer']['patch_size']
        self.hh = self.ww = img_size // patch_size
        self.ph = self.pw = patch_size

        # patch tokenizer: images → per-patch embedding vectors
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                'b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)',
                ph=self.ph,
                pw=self.pw,
            ),
            nn.Linear(
                config['model']['image_tokenizer']['in_channels'] * self.ph * self.pw,
                self.d,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)

        # spatial positional embedding
        self.use_pe_embedding_layer = config['model'].get('input_with_pe', True)
        if self.use_pe_embedding_layer:
            self.pe_embedder = nn.Sequential(
                nn.Linear(self.d, self.d),
                nn.SiLU(),
                nn.Linear(self.d, self.d),
            )
            self.pe_embedder.apply(_init_weights)

            self.pe_embedder_plucker = nn.Sequential(
                nn.Linear(self.d, self.d),
                nn.SiLU(),
                nn.Linear(self.d, self.d),
            )
            self.pe_embedder_plucker.apply(_init_weights)

        use_qk_norm = config['model']['transformer'].get('use_qk_norm', False)
        special_init = config['model']['transformer'].get('special_init', False)
        depth_init = config['model']['transformer'].get('depth_init', False)

        # camera prediction branch (only when encoder_n_layer is set)
        encoder_n_layer = config['model']['transformer'].get('encoder_n_layer', 0)
        if encoder_n_layer > 0:
            if encoder_n_layer % 2 != 0:
                raise ValueError(
                    f'encoder_n_layer must be even (alternating frame/global attention), '
                    f'got {encoder_n_layer}'
                )
            self.num_register_tokens = self._NUM_REGISTER_TOKENS
            self.camera_token = nn.Parameter(torch.empty(1, 1, self.d))
            self.register_token = nn.Parameter(torch.empty(1, self.num_register_tokens, self.d))
            nn.init.normal_(self.camera_token, std=1e-6)
            nn.init.normal_(self.register_token, std=1e-6)
            self.transformer_encoder = build_transformer_blocks(
                num_layers=encoder_n_layer,
                d=self.d,
                d_head=self.d_head,
                use_qk_norm=use_qk_norm,
                special_init=special_init,
                depth_init=depth_init,
            )
            self.pose_predictor = PoseEstimator(config)

        # geometry encoder (alternating frame / global attention)
        encoder_geom_n_layer = config['model']['transformer']['encoder_geom_n_layer']
        if encoder_geom_n_layer % 2 != 0:
            raise ValueError(
                f'encoder_geom_n_layer must be even (alternating frame/global attention), '
                f'got {encoder_geom_n_layer}'
            )
        self.transformer_encoder_geom = build_transformer_blocks(
            num_layers=encoder_geom_n_layer,
            d=self.d,
            d_head=self.d_head,
            use_qk_norm=use_qk_norm,
            special_init=special_init,
            depth_init=depth_init,
        )

        # Plucker ray tokenizer: 6-channel ray map → per-patch d-dim embedding
        self.input_pose_tokenizer = nn.Sequential(
            Rearrange(
                'b v (hh ph) (ww pw) c -> (b v) (hh ww) (ph pw c)',
                ph=self.ph,
                pw=self.pw,
            ),
            nn.Linear(6 * self.ph * self.pw, self.d, bias=False),
        )
        self.input_pose_tokenizer.apply(_init_weights)

        # fusion MLP: concatenated (image + plucker) tokens → d-dim tokens
        self.mlp_fuse = nn.Sequential(
            nn.LayerNorm(self.d * 2, bias=False),
            nn.Linear(self.d * 2, self.d, bias=True),
            nn.SiLU(),
            nn.Linear(self.d, self.d, bias=True),
        )
        self.mlp_fuse.apply(_init_weights)

        self.num_views = config['training']['num_views']
        self.mask_ratio = float(config['model'].get('mask_ratio', 0.0))

        # Gaussian attribute decoder
        sh_degree = config['model']['gaussians']['sh_degree']
        n_gs_channels = 3 + (sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.d, bias=False),
            nn.Linear(self.d, self.ph * self.pw * n_gs_channels, bias=False),
        )
        self.image_token_decoder.apply(_init_weights)

        self.upsampler = GaussiansUpsampler(config)
        self.range_func = get_point_range_func(config['model']['gaussians'])
        self.renderer = Renderer(config)
        self.loss_computer = LossComputer(config)

        self.config_bk = copy.deepcopy(config)
        self.render_interpolate = config['training'].get('render_interpolate', False)

        if config['model']['transformer'].get('fix_decoder', False):
            self.freeze_weights()

        if config.get('inference', False):
            self.random_index = config['training'].get('random_inputs', False)
        else:
            self.random_index = config['training'].get('random_split', False)

        # warm-start fine-tuning from a pretrained ERayZer checkpoint, if given
        init_checkpoint = config['model'].get('init_checkpoint')
        if init_checkpoint:
            self._load_init_checkpoint(init_checkpoint)

    def _load_init_checkpoint(self, ckpt_path: str) -> _IncompatibleKeys:
        """Load pretrained model weights (only) from a checkpoint file.

        Reads the weights, drops non-model keys (e.g. the perceptual loss
        network), and loads with ``strict=False`` so a partially-overlapping
        checkpoint (different layer count, extra heads) still warm-starts the
        shared parameters.

        Parameters
        ----------
        ckpt_path: path to a ``.pt``/``.bin``/``.ckpt`` weight file.

        Returns
        -------
        the ``load_state_dict`` result with ``missing_keys``/``unexpected_keys``.

        Raises
        ------
        FileNotFoundError: if ``ckpt_path`` does not point to a file.

        """
        path = Path(ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(f'init_checkpoint not found: {path}')
        raw = torch.load(str(path), map_location='cpu', weights_only=False)
        state_dict = _filter_init_state_dict(raw)
        result = self.load_state_dict(state_dict, strict=False)
        _logger.info(
            f'loaded init_checkpoint {path.name}: {len(state_dict)} tensors '
            f'({len(result.missing_keys)} missing, '
            f'{len(result.unexpected_keys)} unexpected)'
        )
        return result

    # ------------------------------------------------------------------
    # Camera resolution hook
    # ------------------------------------------------------------------

    def _resolve_cameras(
        self,
        img_tokens: torch.Tensor,
        b: int,
        v_all: int,
        n: int,
        data: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Predict camera parameters from image tokens.

        Override this in subclasses to substitute a different camera source.
        The base implementation runs the learned camera-prediction branch.

        Parameters
        ----------
        img_tokens: spatial-PE-added image tokens of shape [B*V, n, d].
        b: batch size.
        v_all: total number of views (may include padding).
        n: number of patch tokens per view.
        data: batch dict (unused by the prediction branch but available to overrides).
        device: compute device.

        Returns
        -------
        tuple of:
            c2w_all [B, V, 4, 4] camera-to-world matrices,
            fxfycxcy_all [B, V, 4] intrinsics,
            normalized (bool) — True means fxfycxcy values are divided by image dims.

        Raises
        ------
        RuntimeError
            if the camera prediction modules were not initialized (encoder_n_layer == 0).

        """
        if not hasattr(self, 'transformer_encoder'):
            raise RuntimeError(
                'ERayZer camera prediction modules are not initialized '
                '(encoder_n_layer was 0 or not set). '
                'Either set encoder_n_layer > 0 in config, or use BEAST3D for GT cameras.'
            )

        # build per-view token sequences: [cam_token, register_tokens, img_tokens]
        cam_tokens = repeat(self.camera_token, '1 one d -> bv one d', bv=b * v_all)
        reg_tokens = repeat(self.register_token, '1 r d -> bv r d', bv=b * v_all)
        all_tokens = torch.cat([cam_tokens, reg_tokens, img_tokens], dim=1)
        all_tokens = rearrange(all_tokens, '(b v) n d -> b (v n) d', b=b)

        # camera encoder (alternating frame / global attention)
        all_tokens = self.run_vggt_encoder(all_tokens, b, v_all)
        all_tokens = rearrange(all_tokens, 'b (v n) d -> (b v) n d', v=v_all)

        # extract camera token (first token of each view)
        cam_out = all_tokens[:, 0]  # [B*V, d]

        # predict poses
        cam_info = self.pose_predictor(cam_out, v_all)  # [B*V, pose_dim]
        pred_c2w, pred_fxfycxcy = get_cam_se3(cam_info)
        pred_c2w = rearrange(pred_c2w, '(b v) h w -> b v h w', b=b)
        pred_fxfycxcy = rearrange(pred_fxfycxcy, '(b v) d -> b v d', b=b)

        # learn intrinsics end-to-end through the render loss (requires the
        # gsplat fork's v_Ks); freeze them for an initial warmup so geometry can
        # stabilize first (freeze_focal_steps=0 disables the freeze)
        freeze_steps = self.config['training'].get('freeze_focal_steps', 0)
        if self.global_step < freeze_steps:
            pred_fxfycxcy = pred_fxfycxcy.detach()

        return pred_c2w, pred_fxfycxcy, True

    # ------------------------------------------------------------------
    # BaseLightningModel interface
    # ------------------------------------------------------------------

    def get_model_outputs(self, batch_dict: dict) -> dict:
        """Run ERayZer forward pass and return a plain dict.

        Parameters
        ----------
        batch_dict: batch from the dataloader.

        Returns
        -------
        dict of model outputs converted from SimpleNamespace.

        """
        return vars(self(batch_dict))

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Compute composite render + regularization loss.

        Parameters
        ----------
        stage: one of 'train', 'val', 'test'.
        **kwargs: model output fields from get_model_outputs.

        Returns
        -------
        tuple of (total_loss, log_list).

        """
        rendering = kwargs['render']
        target = kwargs['target_image']
        xyz_norm = kwargs.get('xyz_norm')
        xyz_init_norm = kwargs.get('xyz_init_norm')
        pixel_mask = kwargs.get('pixel_mask')
        result = self.loss_computer(rendering, target, xyz_norm, xyz_init_norm, pixel_mask)
        log_list = [
            {'name': f'{stage}_l2', 'value': result.l2_loss},
            {'name': f'{stage}_psnr', 'value': result.psnr, 'prog_bar': True},
            {'name': f'{stage}_gs_reg', 'value': result.gs_reg_loss},
        ]
        if result.perceptual_loss.item() != 0.0:
            log_list.append({'name': f'{stage}_perceptual', 'value': result.perceptual_loss})
        return result.loss, log_list

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Run inference and return model outputs.

        Parameters
        ----------
        batch_dict: batch from the dataloader.
        batch_idx: batch index (unused).

        Returns
        -------
        dict of model outputs.

        """
        return self.get_model_outputs(batch_dict)

    def validation_step(self, batch_dict: dict, batch_idx: int) -> None:
        """Run validation and, on cadence, log render/camera/intrinsic visuals.

        Extends the base validation (loss logging) by logging an input/GT/pred
        render grid, a 3D predicted-camera plot, and intrinsic/Gaussian scalars
        to TensorBoard for the first validation batch every ``viz_every_n_epochs``.

        Parameters
        ----------
        batch_dict: batch from the validation dataloader.
        batch_idx: index of the current validation batch.

        """
        self.evaluate_batch(batch_dict, 'val')
        every = self.config['training'].get('viz_every_n_epochs', 1)
        if viz_is_due(batch_idx, self.current_epoch, every):
            self._log_validation_visuals(batch_dict)

    @torch.no_grad()
    def _log_validation_visuals(self, batch_dict: dict) -> None:
        """Log render grid, camera-pose plot, and scalar stats to TensorBoard.

        Best-effort: any failure is logged as a warning and never interrupts
        training. No-ops when no image-capable logger is attached.

        Parameters
        ----------
        batch_dict: batch from the validation dataloader.

        """
        experiment = getattr(self.logger, 'experiment', None)
        if experiment is None or not hasattr(experiment, 'add_image'):
            return
        try:
            out = self.get_model_outputs(batch_dict)
            step = self.global_step
            # one merged grid: input GT / input render / GT target / pred target
            experiment.add_image(
                'val/render_grid',
                make_render_grid(
                    out['input_image'],
                    out['target_image'],
                    out['render'],
                    render_input=out.get('render_input'),
                ),
                step,
            )
            experiment.add_image(
                'val/pred_cameras',
                make_camera_pose_image(out['c2w_target']),
                step,
            )
            image_size = self.config['model']['image_tokenizer']['image_size']
            stats = camera_intrinsic_stats(out['fxfycxcy_target'], image_size)
            for name, value in stats.items():
                experiment.add_scalar(f'val/{name}', value, step)

            gaussians = out.get('gaussians')
            if gaussians:
                opacity = torch.cat([g.get_opacity.flatten() for g in gaussians]).mean()
                scaling = torch.cat([g.get_scaling.flatten() for g in gaussians]).mean()
                experiment.add_scalar('val/opacity_mean', opacity.item(), step)
                experiment.add_scalar('val/scale_mean', scaling.item(), step)
                self._export_validation_pointcloud(gaussians[0], step)
        except Exception as exc:  # noqa: BLE001 — viz must never break training
            _logger.warning(f'validation visualization failed: {exc}')

    def _export_validation_pointcloud(self, gaussian_model, step: int) -> None:
        """Export the predicted Gaussian point cloud as a GLB under the log dir.

        Writes ``<log_dir>/pointclouds/val_step<step>.glb``. Best-effort: a
        missing log dir or export failure is logged and otherwise ignored.

        Parameters
        ----------
        gaussian_model: GaussianModel for the sample to export.
        step: current global step (used in the filename).

        """
        log_dir = getattr(self.logger, 'log_dir', None)
        if log_dir is None:
            return
        path = Path(log_dir) / 'pointclouds' / f'val_step{step:07d}.glb'
        # drop near-transparent Gaussians (opacity <= 0.05) from the export
        n = export_gaussian_glb(gaussian_model, path, opacity_threshold=0.05)
        _logger.info(f'saved validation point cloud ({n} pts): {path}')

    def configure_optimizers(self) -> dict:
        """Configure AdamW optimizer with OneCycleLR scheduler.

        Returns
        -------
        Lightning optimizer/scheduler config dict.

        """
        cfg = self.config['optimizer']
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=cfg['lr'],
            betas=(cfg['beta1'], cfg['beta2']),
            weight_decay=cfg['wd'],
        )
        total_steps = self.config['training']['max_fwdbwd_passes']
        warmup_steps = cfg['warmup']
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg['lr'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            div_factor=cfg.get('div_factor', 1.0),
            final_div_factor=cfg.get('final_div_factor', 1.0),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
            'monitor': 'val_loss',
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        data: dict,
        render_video: bool = False,
    ) -> SimpleNamespace:
        """Encode posed multi-view images into 3D Gaussians and render.

        Parameters
        ----------
        data: batch dict with keys:
            - image: [B, V, 3, H, W] in [0, 1]
            - c2w / fxfycxcy: only required when using a subclass that reads
              GT cameras (e.g. BEAST3D).
            - input_indices (optional): [B, n_in]
            - target_indices (optional): [B, n_tgt]
        render_video: whether to render an interpolated video trajectory.

        Returns
        -------
        SimpleNamespace with rendered images and auxiliary fields.

        """
        image_all = data['image'] * 2.0 - 1.0  # [B, V, 3, H, W], rescale [0,1] → [-1,1]

        b, v_real, c, h, w = image_all.shape
        device = image_all.device
        batch_idx = torch.arange(b, device=device).unsqueeze(1)

        input_idx, target_idx = self.resolve_view_indices(data, v_real, device)
        if self.training:
            input_idx, target_idx = self.maybe_randomize_view_indices(
                input_idx, target_idx, device,
            )

        v_input = input_idx.shape[1]

        # pad to 10 views (repeat last view) if fewer are available
        pad_views = max(0, 10 - v_real)
        if pad_views:
            last_view = image_all[:, -1:, ...].expand(-1, pad_views, -1, -1, -1)
            image_all = torch.cat([image_all, last_view], dim=1)
            v_all = 10
        else:
            v_all = v_real

        # tokenize all views and add spatial PE
        img_tokens = self.image_tokenizer(image_all)  # [B*V, n, d]
        _, n, d = img_tokens.shape
        if self.use_pe_embedding_layer:
            img_tokens = self.add_spatial_pe(
                img_tokens, b, v_all, self.hh, self.ww, embedder=self.pe_embedder,
            )

        # resolve cameras (predicted or GT, depending on subclass)
        c2w_all, fxfycxcy_all, normalized = self._resolve_cameras(
            img_tokens, b, v_all, n, data, device,
        )

        # select input and target cameras
        c2w_input = c2w_all[batch_idx, input_idx, ...]          # [B, v_input, 4, 4]
        fxfycxcy_input = fxfycxcy_all[batch_idx, input_idx, ...]  # [B, v_input, 4]
        c2w_target = c2w_all[batch_idx, target_idx, ...]          # [B, v_target, 4, 4]
        fxfycxcy_target = fxfycxcy_all[batch_idx, target_idx, ...].clone()

        # Plucker ray encoding for input views
        plucker_rays_input = cam_info_to_plucker(
            c2w_input,
            fxfycxcy_input,
            self.config['model']['target_image'],
            normalized=normalized,
            return_moment=True,
        )
        plucker_rays_input = rearrange(
            plucker_rays_input, '(b v) c h w -> b v h w c', b=b, v=v_input,
        )
        plucker_emb_input = self.input_pose_tokenizer(plucker_rays_input)
        if self.use_pe_embedding_layer:
            plucker_emb_input = self.add_spatial_pe(
                plucker_emb_input, b, v_input, self.hh, self.ww,
                embedder=self.pe_embedder_plucker,
            )
        plucker_emb_input = rearrange(plucker_emb_input, '(b v) n d -> b (v n) d', v=v_input)

        # select and optionally mask input image tokens
        img_tokens_all = rearrange(img_tokens, '(b v) n d -> b v n d', b=b, v=v_all)
        img_tokens_input = img_tokens_all[batch_idx, input_idx, ...]
        if self.training and self.mask_ratio > 0:
            keep = torch.rand(img_tokens_input.shape[:-1], device=device) >= self.mask_ratio
            img_tokens_input = img_tokens_input * keep.unsqueeze(-1).to(img_tokens_input.dtype)

        # fuse image tokens with Plucker embeddings
        img_tokens_input = rearrange(img_tokens_input, 'b v n d -> b (v n) d')
        img_tokens_input = torch.cat([img_tokens_input, plucker_emb_input], dim=-1)
        all_tokens = self.mlp_fuse(img_tokens_input)

        all_tokens = self.run_vggt_encoder_geom(all_tokens, b, v_input)

        # decode tokens to per-pixel Gaussian attributes
        img_aligned_gaussians = self.image_token_decoder(all_tokens)
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians, 'b (v n) d -> b v n d', v=v_input,
        )[:, :v_real]
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians,
            'b v n (ph pw c) -> b (v n ph pw) c',
            ph=self.ph,
            pw=self.pw,
        )

        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(img_aligned_gaussians)

        img_aligned_xyz = rearrange(
            xyz,
            'b (v hh ww ph pw) c -> b v c (hh ph) (ww pw)',
            v=v_input,
            hh=self.hh,
            ww=self.ww,
            ph=self.ph,
            pw=self.pw,
        )

        ray_o = None
        if self.config['model']['hard_pixelalign']:
            img_aligned_xyz = img_aligned_xyz.mean(dim=2, keepdim=True)
            img_aligned_xyz = self.range_func(img_aligned_xyz)
            plucker_rays = cam_info_to_plucker(
                c2w_input,
                fxfycxcy_input,
                self.config['model']['target_image'],
                normalized=normalized,
                return_moment=False,
            )
            plucker_rays = rearrange(plucker_rays, '(b v) c h w -> b v c h w', b=b)
            ray_o, ray_d = plucker_rays.split([3, 3], dim=2)
            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d

        xyz = rearrange(
            img_aligned_xyz,
            'b v c (hh ph) (ww pw) -> b (v hh ww ph pw) c',
            ph=self.ph,
            pw=self.pw,
        )

        xyz, features, scaling, rotation, opacity = map(
            sanitize, (xyz, features, scaling, rotation, opacity),
        )

        gaussian_attrs = SimpleNamespace(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )

        height, width = int(h), int(w)
        if height <= 0 or width <= 0:
            raise ValueError(
                f'Invalid image dimensions from batch: h={h}, w={w}. '
                'Check dataset preprocessing and target_image config.'
            )
        if normalized:
            scale = fxfycxcy_target.new_tensor([width, height, width, height])
            fxfycxcy_target = fxfycxcy_target * scale

        render = self.renderer(
            xyz, features, scaling, rotation, opacity,
            height, width, C2W=c2w_target, fxfycxcy=fxfycxcy_target,
        )

        # input-view reconstruction: render the Gaussians back at the reference
        # cameras as a self-consistency check; computed at val/inference only
        render_input = None
        if not self.training or self.config.get('inference', False):
            fxfycxcy_input_px = fxfycxcy_input
            if normalized:
                fxfycxcy_input_px = fxfycxcy_input * fxfycxcy_input.new_tensor(
                    [width, height, width, height],
                )
            with torch.no_grad():
                render_input = self.renderer(
                    xyz, features, scaling, rotation, opacity,
                    height, width, C2W=c2w_input, fxfycxcy=fxfycxcy_input_px,
                ).render

        vis_only_results = None
        should_render_video = (
            render_video
            or not self.training
            or self.config.get('inference', False)
            or data.get('return_render_video', False)
        )
        if should_render_video:
            with torch.no_grad():
                vis_only_results = self.render_images_video(
                    gaussian_attrs, c2w_target, fxfycxcy_target, normalized=False,
                )

        gaussians = []
        pixelalign_xyz = []
        gaussians_usage = []
        gaussians_scale = []
        gaussians_opacity = []
        for b_i in range(xyz.size(0)):
            self.renderer.gaussians_model.empty()
            gaussians_model = copy.deepcopy(self.renderer.gaussians_model)
            gaussians.append(
                gaussians_model.set_data(
                    xyz[b_i].detach().float(),
                    features[b_i].detach().float(),
                    scaling[b_i].detach().float(),
                    rotation[b_i].detach().float(),
                    opacity[b_i].detach().float(),
                )
            )

            usage_mask = gaussians[-1].get_opacity > 0.05
            usage = usage_mask.sum() / usage_mask.numel()
            gaussians_usage.append(usage.item() if torch.is_tensor(usage) else usage)

            mean_scale = gaussians[-1].get_scaling.mean()
            gaussians_scale.append(
                mean_scale.item() if torch.is_tensor(mean_scale) else mean_scale,
            )

            mean_opacity = gaussians[-1].get_opacity.mean()
            gaussians_opacity.append(
                mean_opacity.item() if torch.is_tensor(mean_opacity) else mean_opacity,
            )

            pa_xyz = gaussians[-1].get_xyz
            pa_xyz = rearrange(
                pa_xyz,
                '(v hh ww ph pw) c -> v c (hh ph) (ww pw)',
                v=v_input,
                hh=self.hh,
                ww=self.ww,
                ph=self.ph,
                pw=self.pw,
            )
            pixelalign_xyz.append(pa_xyz)
        pixelalign_xyz = torch.stack(pixelalign_xyz, dim=0)

        return SimpleNamespace(
            ray_o=ray_o,
            gaussians=gaussians,
            pixelalign_xyz=pixelalign_xyz,
            image=data['image'],
            input_image=data['image'][batch_idx, input_idx.clamp(max=v_real - 1), ...],
            target_image=data['image'][batch_idx, target_idx, ...],
            render=render.render,
            render_input=render_input,
            c2w_input=c2w_input,
            fxfycxcy_input=fxfycxcy_input,
            c2w_target=c2w_target,
            fxfycxcy_target=fxfycxcy_target,
            input_indices=input_idx,
            target_indices=target_idx,
            xyz_norm=None,
            xyz_init_norm=None,
            render_video=(
                vis_only_results.rendered_images_video.detach().clamp(0, 1)
                if vis_only_results is not None
                else None
            ),
        )

    def predict_frame_from_all_tokens(
        self,
        all_tokens: torch.Tensor,
        c2w_input: torch.Tensor,
        fxfycxcy_input: torch.Tensor,
        c2w_target: torch.Tensor,
        fxfycxcy_target: torch.Tensor,
        data: dict,
    ) -> SimpleNamespace:
        """Decode Gaussians and render from pre-computed tokens.

        Parameters
        ----------
        all_tokens: per-view patch tokens of shape [B, v_input, n, d].
        c2w_input: input camera-to-world matrices [B, v_input, 4, 4].
        fxfycxcy_input: input intrinsics [B, v_input, 4].
        c2w_target: target camera-to-world matrices [B, v_target, 4, 4].
        fxfycxcy_target: target intrinsics [B, v_target, 4].
        data: batch dict (used for image shape metadata).

        Returns
        -------
        SimpleNamespace with render, gaussians, pixelalign_xyz, and image fields.

        """
        b, v_input, n, d = all_tokens.shape
        h = w = int(self.config['model']['image_tokenizer']['image_size'])

        all_tokens = rearrange(all_tokens, 'b v n d -> b (v n) d', v=v_input)

        img_aligned_gaussians = self.image_token_decoder(all_tokens)
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians, 'b (v n) d -> b v n d', v=v_input,
        )
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians,
            'b v n (ph pw c) -> b (v n ph pw) c',
            ph=self.ph,
            pw=self.pw,
        )

        if img_aligned_gaussians.shape[0] != b:
            img_aligned_gaussians = rearrange(
                img_aligned_gaussians, '(b v) n c -> b (v n) c', b=b, v=v_input,
            )

        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(img_aligned_gaussians)

        img_aligned_xyz = rearrange(
            xyz,
            'b (v hh ww ph pw) c -> b v c (hh ph) (ww pw)',
            v=v_input,
            hh=self.hh,
            ww=self.ww,
            ph=self.ph,
            pw=self.pw,
        )

        # this method receives pre-computed cameras so normalized=False
        normalized = False
        ray_o = None
        if self.config['model']['hard_pixelalign']:
            img_aligned_xyz = img_aligned_xyz.mean(dim=2, keepdim=True)
            img_aligned_xyz = self.range_func(img_aligned_xyz)
            plucker_rays = cam_info_to_plucker(
                c2w_input,
                fxfycxcy_input,
                self.config['model']['target_image'],
                normalized=normalized,
                return_moment=False,
            )
            plucker_rays = rearrange(plucker_rays, '(b v) c h w -> b v c h w', b=b)
            ray_o, ray_d = plucker_rays.split([3, 3], dim=2)
            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d

        xyz = rearrange(
            img_aligned_xyz,
            'b v c (hh ph) (ww pw) -> b (v hh ww ph pw) c',
            ph=self.ph,
            pw=self.pw,
        )

        xyz, features, scaling, rotation, opacity = map(
            sanitize, (xyz, features, scaling, rotation, opacity),
        )

        height, width = int(h), int(w)
        if height <= 0 or width <= 0:
            raise ValueError(
                f'Invalid image dimensions: h={h}, w={w}. '
                'Check target_image config.'
            )

        render = self.renderer(
            xyz, features, scaling, rotation, opacity,
            height, width, C2W=c2w_target, fxfycxcy=fxfycxcy_target,
        )

        gaussians = []
        pixelalign_xyz = []
        for b_i in range(xyz.size(0)):
            self.renderer.gaussians_model.empty()
            gaussians_model = copy.deepcopy(self.renderer.gaussians_model)
            gaussians.append(
                gaussians_model.set_data(
                    xyz[b_i].detach().float(),
                    features[b_i].detach().float(),
                    scaling[b_i].detach().float(),
                    rotation[b_i].detach().float(),
                    opacity[b_i].detach().float(),
                )
            )
            pa_xyz = gaussians[-1].get_xyz
            pa_xyz = rearrange(
                pa_xyz,
                '(v hh ww ph pw) c -> v c (hh ph) (ww pw)',
                v=v_input,
                hh=self.hh,
                ww=self.ww,
                ph=self.ph,
                pw=self.pw,
            )
            pixelalign_xyz.append(pa_xyz)
        pixelalign_xyz = torch.stack(pixelalign_xyz, dim=0)

        return SimpleNamespace(
            ray_o=ray_o,
            gaussians=gaussians,
            image=data['image'],
            pixelalign_xyz=pixelalign_xyz,
            render=render.render,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def resolve_view_indices(
        self,
        data: dict[str, torch.Tensor],
        num_real_views: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve input and target view indices from the batch or config.

        Parameters
        ----------
        data: batch dict potentially containing explicit view index tensors.
        num_real_views: total number of views in the current batch.
        device: device for output tensors.

        Returns
        -------
        tuple of (input_indices [B, n_in], target_indices [B, n_tgt]).

        """
        batch_size = data['image'].shape[0]

        for input_key in ('input_indices', 'context_indices'):
            if input_key in data and 'target_indices' in data:
                input_idx = data[input_key].to(device=device, dtype=torch.long)
                target_idx = data['target_indices'].to(device=device, dtype=torch.long)
                return input_idx, target_idx

        # multi-view-style sampling (matches the multi-view sampler): each batch
        # draws a random TOTAL view count in [2, num_views], holds out a small
        # number of target views (target_view_range), and feeds the rest as
        # input. The per-sample camera shuffle in the dataset makes which
        # physical cameras land in each slot differ across samples.
        target_view_range = self.config['training'].get('target_view_range')
        if self.training and target_view_range:
            total = int(torch.randint(2, num_real_views + 1, (1,), device=device).item())
            lo = int(target_view_range[0])
            hi = min(int(target_view_range[1]), total - 1)
            hi = max(hi, lo)
            num_target = int(torch.randint(lo, hi + 1, (1,), device=device).item())
            num_input = total - num_target
            perm = torch.randperm(num_real_views, device=device)
            input_idx = perm[:num_input].unsqueeze(0).repeat(batch_size, 1)
            target_idx = perm[num_input:total].unsqueeze(0).repeat(batch_size, 1)
            return input_idx, target_idx

        # variable reference-view-count sampling: pick a random number of input
        # views in [min, max] (per batch, so all samples share a shape) and use
        # the remaining views as targets. The dataset shuffles which physical
        # cameras occupy each slot per sample, so this is random-N-random-which.
        if self.training and self.config['training'].get('random_num_input_views', False):
            lo = self.config['training'].get('min_input_views', 2)
            hi = min(self.config['training'].get('max_input_views', 5), num_real_views - 1)
            hi = max(hi, lo)
            num_input = int(torch.randint(lo, hi + 1, (1,), device=device).item())
            perm = torch.randperm(num_real_views, device=device)
            input_idx = perm[:num_input].unsqueeze(0).repeat(batch_size, 1)
            target_idx = perm[num_input:].unsqueeze(0).repeat(batch_size, 1)
            return input_idx, target_idx

        num_input = self.config['training'].get('num_input_views', num_real_views)
        num_target = self.config['training'].get('num_target_views', num_real_views)

        can_split = (
            not self.config.get('inference', False)
            and num_real_views == self.config['training']['num_views']
            and num_input + num_target <= num_real_views
        )

        if can_split and self.random_index:
            perm = torch.randperm(num_real_views, device=device)
            input_idx = perm[:num_input].unsqueeze(0).repeat(batch_size, 1)
            target_idx = perm[num_input:num_input + num_target].unsqueeze(0).repeat(batch_size, 1)
            return input_idx, target_idx

        if can_split:
            input_idx = torch.arange(num_input, device=device).unsqueeze(0).repeat(batch_size, 1)
            target_idx = torch.arange(
                num_input, num_input + num_target, device=device,
            ).unsqueeze(0).repeat(batch_size, 1)
            return input_idx, target_idx

        input_count = min(num_input, num_real_views)
        input_idx = torch.arange(
            input_count, device=device, dtype=torch.long,
        ).unsqueeze(0).repeat(batch_size, 1)
        target_idx = torch.arange(
            num_real_views, device=device, dtype=torch.long,
        ).unsqueeze(0).repeat(batch_size, 1)
        return input_idx, target_idx

    def maybe_randomize_view_indices(
        self,
        input_idx: torch.Tensor,
        target_idx: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly swap input and target view indices for two-view training.

        Parameters
        ----------
        input_idx: input view indices [B, n_in].
        target_idx: target view indices [B, n_tgt].
        device: device for the swap mask.

        Returns
        -------
        tuple of (input_idx, target_idx) possibly swapped per batch item.

        """
        b = input_idx.shape[0]
        swap = (torch.rand(b, device=device) < 0.5).unsqueeze(1)
        if input_idx.shape[1] == 1:
            return (
                torch.where(swap, target_idx, input_idx),
                torch.where(swap, input_idx, target_idx),
            )
        return input_idx, target_idx

    def add_spatial_pe(
        self,
        tokens: torch.Tensor,
        b: int,
        v: int,
        h_tokens: int,
        w_tokens: int,
        embedder: nn.Module,
    ) -> torch.Tensor:
        """Add 2D sinusoidal positional encoding to patch tokens.

        Parameters
        ----------
        tokens: patch token tensor of shape [B*V, n, d].
        b: batch size.
        v: number of views.
        h_tokens: number of token rows (H / patch_size).
        w_tokens: number of token columns (W / patch_size).
        embedder: learnable projection applied to the positional encoding.

        Returns
        -------
        tokens with spatial positional encoding added, shape [B*V, n, d].

        """
        bv, n, d = tokens.shape
        if h_tokens * w_tokens != n:
            raise ValueError(f'Token count {n} != h_tokens*w_tokens {h_tokens}*{w_tokens}')

        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=d,
            grid_size=(h_tokens, w_tokens),
            device=tokens.device,
        ).to(tokens.dtype)  # [n, d]

        spatial_pe = spatial_pe.reshape(1, 1, n, d).expand(b, v, -1, -1).reshape(bv, n, d)
        return tokens + embedder(spatial_pe)

    def run_vggt_encoder(
        self,
        all_tokens: torch.Tensor,
        b: int,
        v: int,
    ) -> torch.Tensor:
        """Run the camera-prediction encoder with alternating frame/global attention.

        Parameters
        ----------
        all_tokens: token tensor of shape [B, V*n, d].
        b: batch size.
        v: number of views.

        Returns
        -------
        updated token tensor of shape [B, V*n, d].

        """
        checkpoint_every = self.config['training']['grad_checkpoint_every']
        for i in range(0, len(self.transformer_encoder), checkpoint_every):
            if i % 2 == 0:
                all_tokens = rearrange(all_tokens, 'b (v n) d -> (b v) n d', v=v)
            else:
                all_tokens = rearrange(all_tokens, '(b v) n d -> b (v n) d', b=b)
            all_tokens = torch.utils.checkpoint.checkpoint(
                self._run_layers_encoder(i, i + 1),
                all_tokens,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens = self._run_layers_encoder(i + 1, i + checkpoint_every)(all_tokens)
        return all_tokens

    def _run_layers_encoder(self, start: int, end: int):
        """Return a closure applying camera-encoder layers [start, end).

        Parameters
        ----------
        start: index of the first layer to apply.
        end: one past the index of the last layer to apply.

        Returns
        -------
        callable that applies the specified layer range.

        """
        def custom_forward(tokens):
            for i in range(start, min(end, len(self.transformer_encoder))):
                tokens = self.transformer_encoder[i](tokens)
            return tokens
        return custom_forward

    def run_vggt_encoder_geom(
        self,
        all_tokens: torch.Tensor,
        b: int,
        v: int,
    ) -> torch.Tensor:
        """Run the geometry encoder with alternating frame/global attention.

        Parameters
        ----------
        all_tokens: token tensor of shape [B, V*n, d].
        b: batch size.
        v: number of views.

        Returns
        -------
        updated token tensor of shape [B, V*n, d].

        """
        checkpoint_every = self.config['training']['grad_checkpoint_every']
        for i in range(0, len(self.transformer_encoder_geom), checkpoint_every):
            if i % 2 == 0:
                all_tokens = rearrange(all_tokens, 'b (v n) d -> (b v) n d', v=v)
            else:
                all_tokens = rearrange(all_tokens, '(b v) n d -> b (v n) d', b=b)
            all_tokens = torch.utils.checkpoint.checkpoint(
                self._run_layers_geom(i, i + 1),
                all_tokens,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens = self._run_layers_geom(i + 1, i + checkpoint_every)(all_tokens)
        return all_tokens

    def _run_layers_geom(self, start: int, end: int):
        """Return a closure applying geometry-encoder layers [start, end).

        Parameters
        ----------
        start: index of the first layer to apply.
        end: one past the index of the last layer to apply.

        Returns
        -------
        callable that applies the specified layer range.

        """
        def custom_forward(tokens):
            for i in range(start, min(end, len(self.transformer_encoder_geom))):
                tokens = self.transformer_encoder_geom[i](tokens)
            return tokens
        return custom_forward

    def render_images_video(
        self,
        gaussian_attrs: SimpleNamespace,
        c2w_all: torch.Tensor,
        fxfycxcy_all: torch.Tensor,
        normalized: bool = False,
    ) -> SimpleNamespace:
        """Render an interpolated video trajectory from a set of Gaussians.

        Parameters
        ----------
        gaussian_attrs: SimpleNamespace with xyz, features, scaling, rotation, opacity.
        c2w_all: camera-to-world matrices [B, V, 4, 4].
        fxfycxcy_all: intrinsics [B, V, 4].
        normalized: whether intrinsics are normalized (will be scaled by image size).

        Returns
        -------
        SimpleNamespace with field rendered_images_video of shape [B, V', 3, H, W].

        """
        with torch.no_grad():
            xyz = gaussian_attrs.xyz.detach()
            features = gaussian_attrs.features.detach()
            scaling = gaussian_attrs.scaling.detach()
            rotation = gaussian_attrs.rotation.detach()
            opacity = gaussian_attrs.opacity.detach()
            c2w_all = c2w_all.detach()
            fxfycxcy_all = fxfycxcy_all.detach()

            b, v, _, _ = c2w_all.shape
            device = xyz.device
            num_frames = 30

            all_renderings = []
            for i in range(b):
                c2ws = c2w_all[i]
                fxfycxcy = fxfycxcy_all[i]
                Ks = torch.zeros((c2ws.shape[0], 3, 3), device=device)
                Ks[:, 0, 0] = fxfycxcy[:, 0]
                Ks[:, 1, 1] = fxfycxcy[:, 1]
                Ks[:, 0, 2] = fxfycxcy[:, 2]
                Ks[:, 1, 2] = fxfycxcy[:, 3]
                c2ws_interp, Ks_interp = get_interpolated_poses_many(
                    c2ws[:, :3, :4], Ks, num_frames,
                )
                frame_c2ws = torch.cat(
                    [
                        c2ws_interp.to(device),
                        torch.tensor([[[0, 0, 0, 1]]], device=device).expand(
                            c2ws_interp.shape[0], -1, -1,
                        ),
                    ],
                    dim=1,
                )
                frame_fxfycxcy = torch.zeros((c2ws_interp.shape[0], 4), device=device)
                frame_fxfycxcy[:, 0] = Ks_interp[:, 0, 0]
                frame_fxfycxcy[:, 1] = Ks_interp[:, 1, 1]
                frame_fxfycxcy[:, 2] = Ks_interp[:, 0, 2]
                frame_fxfycxcy[:, 3] = Ks_interp[:, 1, 2]

                img_size = self.config['model']['image_tokenizer']['image_size']
                num_views_traj = frame_c2ws.shape[0]
                renderings = []
                for start in range(0, num_views_traj, 5):
                    end = min(start + 5, num_views_traj)
                    rendered_batch = self.renderer(
                        xyz, features, scaling, rotation, opacity,
                        img_size, img_size,
                        C2W=frame_c2ws[start:end].unsqueeze(0),
                        fxfycxcy=frame_fxfycxcy[start:end].unsqueeze(0),
                    ).render.squeeze(0)
                    renderings.append(rendered_batch)

                all_renderings.append(torch.cat(renderings, dim=0))

            all_renderings = torch.stack(all_renderings)  # [B, V', 3, H, W]

        return SimpleNamespace(rendered_images_video=all_renderings)
