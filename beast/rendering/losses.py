"""Loss functions for novel view synthesis: perceptual, LPIPS, and composite render loss."""

import urllib.request
from pathlib import Path
from types import SimpleNamespace

import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19

try:
    import lpips
except ImportError:
    lpips = None


def subspace_overlap(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute inner product between subspaces defined by matrices A and B.

    Reference: https://github.com/themattinthehatt/behavenet

    Parameters
    ----------
    A: matrix of shape (a, d).
    B: matrix of shape (b, d).
    C: optional background subspace projection matrix of shape (c, d).

    Returns
    -------
    scalar Frobenius norm of UU^T divided by number of entries.

    """
    if C is None:
        U = torch.cat([A, B], dim=0)
    else:
        U = torch.cat([A, B, C], dim=0)
    d = U.shape[0]
    eye = torch.eye(d, device=U.device)
    return torch.mean((torch.matmul(U, torch.transpose(U, 1, 0)) - eye).pow(2))


class PerceptualLoss(nn.Module):
    """RayZer-style perceptual loss with a torchvision VGG19 fallback."""

    def __init__(self, device: str = 'cpu') -> None:
        """Initialize.

        Parameters
        ----------
        device: device to place the VGG model on.

        """
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._setup_feature_blocks()

    def _build_vgg(self) -> nn.Module:
        """Build the VGG19 model, loading matconvnet weights if available."""
        model = vgg19(weights=None)
        for idx, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[idx] = nn.AvgPool2d(kernel_size=2, stride=2)

        if not self._load_matconvnet_weights(model):
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            for idx, layer in enumerate(model.features):
                if isinstance(layer, nn.MaxPool2d):
                    model.features[idx] = nn.AvgPool2d(kernel_size=2, stride=2)

        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _load_matconvnet_weights(self, model: nn.Module) -> bool:
        """Attempt to load matconvnet VGG19 weights, downloading if needed.

        Parameters
        ----------
        model: the VGG19 model to load weights into.

        Returns
        -------
        True if weights were loaded successfully, False otherwise.

        """
        weight_file = Path('metric_checkpoint/imagenet-vgg-verydeep-19.mat')
        weight_file.parent.mkdir(parents=True, exist_ok=True)
        if not weight_file.exists():
            try:
                urllib.request.urlretrieve(
                    'https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat',
                    weight_file,
                )
            except Exception:
                return False

        try:
            vgg_data = scipy.io.loadmat(weight_file)
        except Exception:
            return False

        vgg_layers = vgg_data['layers'][0]
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [
            64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512,
        ]
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                weights = torch.from_numpy(
                    vgg_layers[layer_idx][0][0][2][0][0]
                ).permute(3, 2, 0, 1)
                biases = torch.from_numpy(
                    vgg_layers[layer_idx][0][0][2][0][1]
                ).view(filter_sizes[i])
                model.features[layer_idx].weight.copy_(weights)
                model.features[layer_idx].bias.copy_(biases)
        return True

    def _setup_feature_blocks(self) -> None:
        """Build sequential feature extraction blocks from VGG layers."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(
                *list(self.vgg.features[output_indices[i]: output_indices[i + 1]])
            )
            block = block.to(self.device).eval()
            for param in block.parameters():
                param.requires_grad = False
            self.blocks.append(block)

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run input through all feature blocks and collect intermediate outputs.

        Parameters
        ----------
        x: input image tensor.

        Returns
        -------
        list of feature tensors, one per block.

        """
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Subtract ImageNet channel means from images scaled to [0, 255].

        Parameters
        ----------
        images: image tensor in [0, 1] range.

        Returns
        -------
        preprocessed image tensor.

        """
        mean = torch.tensor([123.68, 116.779, 103.939], device=images.device)
        mean = mean.view(1, 3, 1, 1)
        return images * 255.0 - mean

    def forward(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between predicted and target images.

        Parameters
        ----------
        pred_img: predicted image tensor in [0, 1].
        target_img: target image tensor in [0, 1].

        Returns
        -------
        scalar perceptual loss.

        """
        pred_img = self._preprocess_images(pred_img)
        target_img = self._preprocess_images(target_img)

        pred_features = self._extract_features(pred_img)
        target_features = self._extract_features(target_img)

        e0 = torch.mean(torch.abs(target_img - pred_img))
        e1 = torch.mean(torch.abs(target_features[0] - pred_features[0])) / 2.6
        e2 = torch.mean(torch.abs(target_features[1] - pred_features[1])) / 4.8
        e3 = torch.mean(torch.abs(target_features[2] - pred_features[2])) / 3.7
        e4 = torch.mean(torch.abs(target_features[3] - pred_features[3])) / 5.6
        e5 = torch.mean(torch.abs(target_features[4] - pred_features[4])) * 10.0 / 1.5
        return (e0 + e1 + e2 + e3 + e4 + e5) / 255.0


def masked_mse_loss(
    rendering: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted MSE over RGB pixels using a foreground mask.

    Parameters
    ----------
    rendering: predicted image tensor.
    target: target image tensor.
    pixel_mask: binary mask tensor; sum(w*(p-t)^2) / sum(w), w repeated across channels.

    Returns
    -------
    scalar masked MSE loss.

    """
    m = pixel_mask.to(dtype=rendering.dtype, device=rendering.device)
    if m.ndim == 3:
        m = m.unsqueeze(1)
    valid_mask = m.expand_as(rendering)
    loss = F.mse_loss(rendering, target, reduction='none') * valid_mask
    normalizer = valid_mask.sum()
    return loss.sum() / normalizer


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
