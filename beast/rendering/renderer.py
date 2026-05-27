"""Gaussian splatting renderer wrapper and depth range function factory."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from beast.rendering.gaussians_renderer import (
    GaussianModel,
    deferred_gaussian_render,
    render_opencv_cam,
    render_opencv_cam_gsplat,
)


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
