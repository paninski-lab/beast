"""Perceptual loss modules using pretrained feature extractors."""

import urllib.request
from pathlib import Path

import scipy.io
import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG19_Weights, vgg19

# https://github.com/MLReproHub/SMAE/blob/main/src/loss/perceptual.py


class Perceptual(nn.Module):
    """Base class for perceptual loss modules.

    Subclasses must implement ``forward(x_hat, x) -> Tensor`` where ``x_hat``
    is the reconstruction and ``x`` is the target, both in [0, 1].
    """


class AlexPerceptual(Perceptual):
    """Perceptual loss using the first five layers of a pretrained AlexNet."""

    def __init__(self, *, device: str | torch.device, criterion: nn.Module) -> None:
        """Perceptual loss using pretrained AlexNet features [Pihlgren et al. 2020].

        Extracts features from the first five layers of AlexNet (pretrained on ImageNet)
        and computes loss between reconstructed and target feature maps.

        Parameters
        ----------
        device: device to run the feature extractor on (e.g. 'cuda', 'cpu')
        criterion: loss function applied to sigmoid-normalised features (e.g. nn.MSELoss())

        """
        super().__init__()
        alex_net = torchvision.models.alexnet(weights='IMAGENET1K_V1')
        perceptual_net = alex_net.features[:5].to(device)
        for parameter in perceptual_net.parameters():
            parameter.requires_grad = False
        self.net = perceptual_net
        self.criterion = criterion
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between reconstructed and target images.

        Parameters
        ----------
        x_hat: reconstructed image batch
        x: target image batch

        Returns
        -------
        scalar loss tensor

        """
        x_hat_features = self.sigmoid(self.net(x_hat))
        x_features = self.sigmoid(self.net(x))
        return self.criterion(x_hat_features, x_features)


class VGGPerceptual(Perceptual):
    """RayZer-style multi-scale perceptual loss using VGG19 features.

    Extracts features at five spatial scales and combines weighted L1 losses,
    matching the RayZer perceptual loss formulation.  Attempts to load
    matconvnet weights; falls back to torchvision ImageNet weights.
    """

    def __init__(self, device: str | torch.device = 'cpu') -> None:
        """Initialize VGG19 feature extractor.

        Parameters
        ----------
        device: device to place the VGG model on

        """
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._setup_feature_blocks()

    def _build_vgg(self) -> nn.Module:
        """Build VGG19, replacing MaxPool with AvgPool, loading matconvnet weights if available."""
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
        model: VGG19 model to load weights into

        Returns
        -------
        True if weights were loaded successfully, False otherwise

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
        """Build sequential feature extraction blocks from VGG19 layers."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(
                *list(self.vgg.features[output_indices[i]:output_indices[i + 1]])
            )
            block = block.to(self.device).eval()
            for param in block.parameters():
                param.requires_grad = False
            self.blocks.append(block)

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run input through all feature blocks and collect intermediate outputs.

        Parameters
        ----------
        x: preprocessed image tensor

        Returns
        -------
        list of feature tensors, one per block

        """
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract ImageNet channel means from images scaled to [0, 255].

        Parameters
        ----------
        x: image tensor in [0, 1]

        Returns
        -------
        preprocessed image tensor

        """
        mean = torch.tensor([123.68, 116.779, 103.939], device=x.device).view(1, 3, 1, 1)
        return x * 255.0 - mean

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale perceptual loss between reconstructed and target images.

        Parameters
        ----------
        x_hat: reconstructed image tensor in [0, 1]
        x: target image tensor in [0, 1]

        Returns
        -------
        scalar perceptual loss

        """
        x_hat = self._preprocess(x_hat)
        x = self._preprocess(x)

        x_hat_f = self._extract_features(x_hat)
        x_f = self._extract_features(x)

        e0 = torch.mean(torch.abs(x - x_hat))
        e1 = torch.mean(torch.abs(x_f[0] - x_hat_f[0])) / 2.6
        e2 = torch.mean(torch.abs(x_f[1] - x_hat_f[1])) / 4.8
        e3 = torch.mean(torch.abs(x_f[2] - x_hat_f[2])) / 3.7
        e4 = torch.mean(torch.abs(x_f[3] - x_hat_f[3])) / 5.6
        e5 = torch.mean(torch.abs(x_f[4] - x_hat_f[4])) * 10.0 / 1.5
        return (e0 + e1 + e2 + e3 + e4 + e5) / 255.0
