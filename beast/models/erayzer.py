"""ERayZer model components: renderer, loss computer, and depth range utilities."""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from beast.rendering.gaussians_renderer import (
    GaussianModel,
    deferred_gaussian_render,
    render_opencv_cam,
    render_opencv_cam_gsplat,
)
from beast.rendering.losses import PerceptualLoss, masked_mse_loss

try:
    import lpips
except ImportError:
    lpips = None


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
        height: render height in pixels.
        width: render width in pixels.
        C2W: camera-to-world matrices of shape [b, v, 4, 4].
        fxfycxcy: intrinsics of shape [b, v, 4].
        deferred: use deferred rendering (default True).

        Returns
        -------
        SimpleNamespace with fields render, depth, alpha.

        """
        if self.config['model'].get('use_deferred_rendering', False):
            renderings = deferred_gaussian_render(
                xyz, features, scaling, rotation, opacity, height, width, C2W, fxfycxcy,
            )
            b, v = C2W.size(0), C2W.size(1)
            depth = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device,
            )
            alpha = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device,
            )
        else:
            b, v = C2W.size(0), C2W.size(1)
            renderings = torch.zeros(
                b, v, 3, height, width, dtype=torch.float32, device=xyz.device,
            )

            depth = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device,
            )
            alpha = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device,
            )

            for i in range(b):
                pc = self.gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i],
                )
                if self.config['model'].get('use_gsplat', True):
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
                else:
                    for j in range(v):
                        buffers = render_opencv_cam(
                            pc, height, width, C2W[i, j], fxfycxcy[i, j],
                        )
                        renderings[i, j] = buffers['render']
                        if 'depth' in buffers and buffers['depth'] is not None:
                            depth[i, j] = buffers['depth']
                        if 'alpha' in buffers and buffers['alpha'] is not None:
                            alpha[i, j] = buffers['alpha']

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

        self.lpips_loss_module = None
        if self.config['training'].get('lpips_loss_weight', 0.0) > 0.0:
            if lpips is None:
                raise ImportError('lpips is not installed but lpips_loss_weight > 0')
            self.lpips_loss_module = lpips.LPIPS(net='vgg').to(device).eval()
            for param in self.lpips_loss_module.parameters():
                param.requires_grad = False

        self.perceptual_loss_module = None
        if self.config['training'].get('perceptual_loss_weight', 0.0) > 0.0:
            self.perceptual_loss_module = PerceptualLoss(device).eval()

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
        SimpleNamespace with fields: loss, l2_loss, psnr, gs_reg_loss, lpips_loss,
        perceptual_loss.

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

        lpips_loss = rendering.new_zeros(())
        if self.lpips_loss_module is not None:
            lpips_loss = self.lpips_loss_module(
                rendering * 2.0 - 1.0,
                target * 2.0 - 1.0,
            ).mean()

        perceptual_loss = rendering.new_zeros(())
        if self.perceptual_loss_module is not None:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        total_loss = (
            self.config['training'].get('l2_loss_weight', 1.0) * l2_loss
            + self.config['training'].get('gs_reg_loss_weight', 0.0) * gs_reg_loss
            + self.config['training'].get('lpips_loss_weight', 0.0) * lpips_loss
            + self.config['training'].get('perceptual_loss_weight', 0.0) * perceptual_loss
        )

        return SimpleNamespace(
            loss=total_loss,
            l2_loss=l2_loss,
            gs_reg_loss=gs_reg_loss,
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
        )
