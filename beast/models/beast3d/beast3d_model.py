"""BEAST3D model: ERayZer with a DINOv3 tokenizer, GT cameras, and frustum 3DGS.

BEAST3D specializes ERayZer for posed multi-view data: it reads ground-truth
cameras from the batch (so the pose-prediction branch is disabled via
``transformer.encoder_n_layer = 0``), tokenizes images with a frozen DINOv3
backbone, restricts Gaussians to the intersection of the input view frustums,
and supervises rendering with masked photometric + alpha losses against the GT
foreground mask (with random-background compositing during training).

Only the methods that differ from ERayZer are overridden; the encoder, Gaussian
decoder, renderer, and forward pass are inherited unchanged.
"""

import logging

import torch
import torch.nn.functional as F
from einops import rearrange

from beast.models.erayzer.erayzer_model import ERayZer
from beast.nn.dino import DinoV3

_logger = logging.getLogger(__name__)

# ImageNet statistics: DINOv3 expects [0, 1] images normalized by these.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Beast3D(ERayZer):
    """ERayZer variant with a DINOv3 tokenizer, GT cameras, and frustum 3DGS."""

    def __init__(self, config: dict) -> None:
        """Initialize BEAST3D from a full config dict.

        Parameters
        ----------
        config: full training/model config dict; reads ``config['model']`` for
            the BEAST3D-specific flags (use_dinov3, freeze_dinov3,
            dino_model_name, mask_loss_weight, random_background).

        """
        super().__init__(config)

        model_cfg = config['model']
        self.use_dinov3 = model_cfg.get('use_dinov3', True)
        self.freeze_dinov3 = model_cfg.get('freeze_dinov3', True)
        self.mask_loss_weight = model_cfg.get('mask_loss_weight', 0.1)
        self.random_background = model_cfg.get('random_background', True)

        if self.use_dinov3:
            # the learned patch tokenizer is unused; swap in a DINOv3 backbone
            del self.image_tokenizer
            self.dinov3 = DinoV3(
                model_name=model_cfg.get(
                    'dino_model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                ),
                freeze=self.freeze_dinov3,
            )
            if self.dinov3.embed_dim != self.d:
                raise ValueError(
                    f'DINOv3 embed_dim {self.dinov3.embed_dim} != transformer.d {self.d}; '
                    'set model.transformer.d to match the DINOv3 backbone.'
                )
            mode = 'frozen' if self.freeze_dinov3 else 'trainable'
            _logger.info(f'BEAST3D: using DINOv3 ViT-base tokenizer ({mode})')

    def train(self, mode: bool = True) -> 'Beast3D':
        """Pin a frozen DINOv3 to eval mode even while the rest of the model trains.

        Parameters
        ----------
        mode: whether to set training mode (True) or eval mode (False).

        Returns
        -------
        self.

        """
        super().train(mode)
        if self.use_dinov3 and self.freeze_dinov3:
            self.dinov3.eval()
        return self

    def _tokenize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Tokenize images with DINOv3 (ImageNet-normalized) or the base tokenizer.

        Parameters
        ----------
        images: image tensor of shape [B, V, 3, H, W] in [0, 1].

        Returns
        -------
        patch tokens of shape [B*V, n, d].

        """
        if not self.use_dinov3:
            return super()._tokenize_images(images)
        # DINOv3 expects ImageNet-normalized [0, 1] inputs
        mean = images.new_tensor(_IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = images.new_tensor(_IMAGENET_STD).view(1, 1, 3, 1, 1)
        images = (images - mean) / std
        patch_tokens, _ = self.dinov3(images)  # [B, V, n, d]
        return rearrange(patch_tokens, 'b v n d -> (b v) n d')

    def _resolve_cameras(
        self,
        img_tokens: torch.Tensor,
        b: int,
        v_all: int,
        n: int,
        data: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Return ground-truth cameras from the batch (no pose prediction).

        The dataset supplies ``c2w`` and ``fxfycxcy`` (intrinsics in pixels at the
        render resolution). They are padded to ``v_all`` to line up with the
        padded image tokens; padded views are never indexed by the input/target
        selection so their values are irrelevant.

        Parameters
        ----------
        img_tokens: image tokens (unused; cameras are read from ``data``).
        b: batch size.
        v_all: total number of (possibly padded) views.
        n: patch tokens per view (unused).
        data: batch dict with ``c2w`` [B, V, 4, 4] and ``fxfycxcy`` [B, V, 4].
        device: compute device.

        Returns
        -------
        tuple of (c2w [B, v_all, 4, 4], fxfycxcy [B, v_all, 4], normalized=False).

        """
        c2w = data['c2w'].to(device=device, dtype=torch.float32)
        fxfycxcy = data['fxfycxcy'].to(device=device, dtype=torch.float32)
        pad = v_all - c2w.shape[1]
        if pad > 0:
            c2w = torch.cat([c2w, c2w[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            fxfycxcy = torch.cat([fxfycxcy, fxfycxcy[:, -1:].expand(-1, pad, -1)], dim=1)
        # intrinsics are already in pixels at the render resolution
        return c2w, fxfycxcy, False

    def _sample_background(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Sample a random background colour during training, else None.

        Parameters
        ----------
        device: device for the colour tensor.
        dtype: dtype for the colour tensor.

        Returns
        -------
        a random [3] colour during training (when enabled), otherwise None.

        """
        if self.training and self.random_background:
            return torch.rand(3, device=device, dtype=dtype)
        return None

    def _prepare_target(
        self,
        target_image: torch.Tensor,
        data: dict,
        batch_idx: torch.Tensor,
        target_idx: torch.Tensor,
        bg_color: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Composite the background into the masked target and return the mask.

        When the batch carries no foreground mask, falls back to the base
        behaviour (raw target, no mask).

        Parameters
        ----------
        target_image: target views of shape [B, v_target, 3, H, W].
        data: batch dict; ``input_mask`` is [B, V, 1, H, W] when present.
        batch_idx: batch index helper of shape [B, 1].
        target_idx: target view indices of shape [B, v_target].
        bg_color: background colour [3] used by the renderer, or None.

        Returns
        -------
        tuple of (composited target [B, v_target, 3, H, W], mask [B, v_target, 1, H, W]).

        """
        input_mask = data.get('input_mask')
        if input_mask is None:
            return target_image, None
        mask = input_mask[batch_idx, target_idx, ...].to(
            device=target_image.device, dtype=target_image.dtype,
        )
        mask_fg = mask[:, :, :1, ...]  # first channel is the binary foreground
        target = target_image * mask_fg
        if bg_color is not None:
            target = target + bg_color.view(1, 1, 3, 1, 1) * (1.0 - mask_fg)
        return target, mask_fg

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Add a masked alpha-vs-foreground term to the base render loss.

        The base loss already applies the foreground mask to the photometric
        term (via ``pixel_mask``); this adds an MSE between the rendered alpha
        and the GT mask, weighted by ``mask_loss_weight``.

        Parameters
        ----------
        stage: one of 'train', 'val', 'test'.
        **kwargs: model output fields from get_model_outputs.

        Returns
        -------
        tuple of (total_loss, log_list).

        """
        loss, log_list = super().compute_loss(stage, **kwargs)
        render_alphas = kwargs.get('render_alphas')
        pixel_mask = kwargs.get('pixel_mask')
        if render_alphas is not None and pixel_mask is not None and self.mask_loss_weight > 0:
            # MSE on alpha (not BCE) avoids extreme gradients in the large bg region
            alpha = render_alphas.clamp(1e-5, 1.0 - 1e-5)
            mask_loss = F.mse_loss(alpha, pixel_mask.to(alpha.dtype))
            loss = loss + self.mask_loss_weight * mask_loss
            log_list.append({'name': f'{stage}_mask', 'value': mask_loss})
        return loss, log_list
