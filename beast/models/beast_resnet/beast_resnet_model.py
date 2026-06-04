"""ResNet autoencoder implementation.

Adapted from https://github.com/Horizon2333/imagenet-autoencoder

"""

from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float

from beast.models.base import BaseLightningModel

_RESNET_CONFIGS: dict[str, tuple[list[int], bool]] = {
    'resnet18': ([2, 2, 2, 2], False),
    'resnet34': ([3, 4, 6, 3], False),
    'resnet50': ([3, 4, 6, 3], True),
    'resnet101': ([3, 4, 23, 3], True),
    'resnet152': ([3, 8, 36, 3], True),
}


def get_configs(arch: str = 'resnet18') -> tuple[list[int], bool]:
    """Get number and type of layers for resnet models."""
    if arch not in _RESNET_CONFIGS:
        raise ValueError(f'{arch} is an invalid entry in model.model_params.backbone')
    return _RESNET_CONFIGS[arch]


class ResnetAutoencoder(BaseLightningModel):
    """ResNet autoencoder implementation."""

    def __init__(self, config: dict) -> None:
        """Initialize encoder, decoder, and optional latent bottleneck from config."""
        super().__init__(config)

        resnet_config, bottleneck = get_configs(config['model']['model_params']['backbone'])
        self.encoder = ResNetEncoder(configs=resnet_config, bottleneck=bottleneck)
        self.decoder = ResNetDecoder(configs=resnet_config[::-1], bottleneck=bottleneck)

        # if 'num_latents' exists, create a linear bottleneck layer
        # otherwise, the intermediate bottleneck is a set of high-dimensional feature maps
        self.num_latents = config['model']['model_params'].get('num_latents')
        if self.num_latents:
            self.encoder_to_latents = LatentMapping(
                num_latents=self.num_latents, source='encoder', bottleneck=bottleneck,
            )
            self.latents_to_decoder = LatentMapping(
                num_latents=self.num_latents, source='latents', bottleneck=bottleneck,
            )

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> tuple[
        Float[torch.Tensor, 'batch channels img_height img_width'],         # reconstructions
        Float[torch.Tensor, 'batch features feat_height feat_width']        # latents
        | Float[torch.Tensor, 'batch num_latents'],
    ]:
        """Encode input image to latents and decode back to image space.

        Returns
        -------
        tuple of (reconstructed_images, latents)

        """
        features = self.encoder(x)
        if self.num_latents:
            z = self.encoder_to_latents(features)
            features = self.latents_to_decoder(z)
        else:
            z = features
        xhat = self.decoder(features)
        return xhat, z

    def get_model_outputs(
        self,
        batch_dict: dict,
        return_images: bool = True,
        return_reconstructions: bool = True,
    ) -> dict:
        """Run forward pass and return results dict with optional images and reconstructions.

        Parameters
        ----------
        batch_dict: dict containing 'image' tensor
        return_images: whether to include input images in results
        return_reconstructions: whether to include reconstructions in results

        Returns
        -------
        dict with 'latents', and optionally 'images' and 'reconstructions'

        """
        x = batch_dict['image']
        xhat, z = self.forward(x)
        results_dict = {
            'latents': z,
        }
        if return_images:
            results_dict['images'] = x
        if return_reconstructions:
            results_dict['reconstructions'] = xhat
        return results_dict

    def compute_loss(
        self,
        stage: str | None,
        images: Float[torch.Tensor, 'batch channels img_height img_width'],
        reconstructions: Float[torch.Tensor, 'batch channels img_height img_width'],
        latents: (
            Float[torch.Tensor, 'batch features feat_height feat_width']
            | Float[torch.Tensor, 'batch num_latents']
        ),
        **kwargs,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Compute MSE reconstruction loss between input images and reconstructions.

        Parameters
        ----------
        stage: training stage ('train', 'val', 'test', or None)
        images: original input images
        reconstructions: model reconstructions
        latents: encoder latent representations (unused; required by base class signature)
        **kwargs: additional keyword arguments (ignored)

        Returns
        -------
        tuple of (loss tensor, list of logging dicts)

        """
        mse_loss = nn.functional.mse_loss(images, reconstructions, reduction='mean')
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss}
        ]
        return mse_loss, log_list

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Run inference on a single batch and return latents with metadata.

        Parameters
        ----------
        batch_dict: dict containing 'image', 'video', 'idx', 'image_path'
        batch_idx: index of the current batch

        Returns
        -------
        dict with 'latents', optional 'reconstructions', and 'metadata'

        """
        results_dict = self.get_model_outputs(
            batch_dict,
            return_images=False,
            return_reconstructions=self.return_reconstructions,
        )
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict


class LatentMapping(nn.Module):
    """Linear bottleneck layer mapping between encoder feature maps and a flat latent vector."""

    def __init__(self, num_latents: int, source: str, bottleneck: bool) -> None:
        """Build linear mapping layer between encoder feature maps and a flat latent vector.

        Parameters
        ----------
        num_latents: dimensionality of the flat latent vector
        source: 'encoder' to map feature maps → latents, 'latents' to map latents → feature maps
        bottleneck: whether the encoder uses bottleneck (2048-channel) blocks

        """
        super().__init__()

        self.num_latents = num_latents
        self.source = source
        self.bottleneck = bottleneck

        self.reduce = None  # expanded feature maps to reduced dim feature maps
        self.expand = None  # reduced feature maps to expanded dim feature maps
        if source == 'encoder':
            if bottleneck:
                self.reduce = nn.Conv2d(
                    in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0,
                )
            in_features = 512 * 7 * 7  # feature maps * feature_height * feature_width
            out_features = num_latents
        elif source == 'latents':
            if bottleneck:
                self.expand = nn.Conv2d(
                    in_channels=512, out_channels=2048, kernel_size=1, stride=1, padding=0,
                )
            out_features = 512 * 7 * 7  # feature maps * feature_height * feature_width
            in_features = num_latents
        else:
            raise ValueError(
                f'source argument to LatentMapping must be "encoder" or "latents", not {source}'
            )

        self.layer = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features))

    def forward(
        self,
        x: (
            Float[torch.Tensor, 'batch num_features feature_height feature_width']
            | Float[torch.Tensor, 'batch num_features']
        ),
    ) -> (
        Float[torch.Tensor, 'batch num_features feature_height feature_width']
        | Float[torch.Tensor, 'batch num_features']
    ):
        """Map between encoder feature maps and flat latent vector."""
        if self.source == 'encoder':
            if self.reduce:
                x = self.reduce(x)
            x = x.reshape((x.shape[0], -1))
            x = self.layer(x)
        else:
            x = self.layer(x)
            x = x.reshape((x.shape[0], -1, 7, 7))
            if self.expand:
                x = self.expand(x)
        return x


class ResNetEncoder(nn.Module):
    """ResNet encoder that maps input images to a spatial feature map."""

    def __init__(self, configs: list, bottleneck: bool = False) -> None:
        """Build encoder from per-stage layer counts.

        Parameters
        ----------
        configs: list of four ints specifying the number of layers per stage
        bottleneck: whether to use bottleneck blocks (True for ResNet-50/101/152)

        """
        super().__init__()

        if len(configs) != 4:
            raise ValueError('Only 4 layers can be configued')

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:

            self.conv2 = EncoderBottleneckBlock(
                in_channels=64, hidden_channels=64, up_channels=256, layers=configs[0],
                downsample_method='pool',
            )
            self.conv3 = EncoderBottleneckBlock(
                in_channels=256, hidden_channels=128, up_channels=512, layers=configs[1],
                downsample_method='conv',
            )
            self.conv4 = EncoderBottleneckBlock(
                in_channels=512, hidden_channels=256, up_channels=1024, layers=configs[2],
                downsample_method='conv',
            )
            self.conv5 = EncoderBottleneckBlock(
                in_channels=1024, hidden_channels=512, up_channels=2048, layers=configs[3],
                downsample_method='conv',
            )

        else:

            self.conv2 = EncoderResidualBlock(
                in_channels=64, hidden_channels=64, layers=configs[0], downsample_method='pool',
            )
            self.conv3 = EncoderResidualBlock(
                in_channels=64, hidden_channels=128, layers=configs[1], downsample_method='conv',
            )
            self.conv4 = EncoderResidualBlock(
                in_channels=128, hidden_channels=256, layers=configs[2], downsample_method='conv',
            )
            self.conv5 = EncoderResidualBlock(
                in_channels=256, hidden_channels=512, layers=configs[3], downsample_method='conv',
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to spatial feature map."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class ResNetDecoder(nn.Module):
    """ResNet decoder that upsamples a feature map back to the input image resolution."""

    def __init__(self, configs: list, bottleneck: bool = False) -> None:
        """Build decoder from per-stage layer counts.

        Parameters
        ----------
        configs: list of four ints specifying the number of layers per stage (reversed order)
        bottleneck: whether to use bottleneck blocks (True for ResNet-50/101/152)

        """
        super().__init__()

        if len(configs) != 4:
            raise ValueError('Only 4 layers can be configued')

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(
                in_channels=2048, hidden_channels=512, down_channels=1024, layers=configs[0],
            )
            self.conv2 = DecoderBottleneckBlock(
                in_channels=1024, hidden_channels=256, down_channels=512, layers=configs[1],
            )
            self.conv3 = DecoderBottleneckBlock(
                in_channels=512, hidden_channels=128, down_channels=256, layers=configs[2],
            )
            self.conv4 = DecoderBottleneckBlock(
                in_channels=256, hidden_channels=64, down_channels=64, layers=configs[3],
            )

        else:

            self.conv1 = DecoderResidualBlock(
                hidden_channels=512, output_channels=256, layers=configs[0],
            )
            self.conv2 = DecoderResidualBlock(
                hidden_channels=256, output_channels=128, layers=configs[1],
            )
            self.conv3 = DecoderResidualBlock(
                hidden_channels=128, output_channels=64, layers=configs[2],
            )
            self.conv4 = DecoderResidualBlock(
                hidden_channels=64, output_channels=64, layers=configs[3],
            )

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3,
                output_padding=1, bias=False,
            ),
        )

        # self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode feature map to reconstructed image."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.gate(x)
        return x


class EncoderResidualBlock(nn.Module):
    """Stack of residual layers with optional spatial downsampling for the encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        layers: int,
        downsample_method: Literal['conv', 'pool'] = 'conv',
    ) -> None:
        """Build residual block with given layer count and downsampling strategy.

        Parameters
        ----------
        in_channels: number of input channels
        hidden_channels: number of channels in each residual layer
        layers: number of residual layers
        downsample_method: 'conv' for strided convolution, 'pool' for max pooling

        """
        super().__init__()

        if downsample_method == 'conv':

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        downsample=True,
                    )
                else:
                    layer = EncoderResidualLayer(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )

                self.add_module(f'{i} EncoderLayer', layer)

        elif downsample_method == 'pool':

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )
                else:
                    layer = EncoderResidualLayer(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )

                self.add_module(f'{i + 1} EncoderLayer', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all residual layers sequentially."""
        for _name, layer in self.named_children():
            x = layer(x)
        return x


class EncoderBottleneckBlock(nn.Module):
    """Stack of bottleneck layers with optional spatial downsampling for the encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        up_channels: int,
        layers: int,
        downsample_method: Literal['conv', 'pool'] = 'conv',
    ) -> None:
        """Build bottleneck block with given layer count and downsampling strategy.

        Parameters
        ----------
        in_channels: number of input channels
        hidden_channels: number of channels in the 3x3 convolution
        up_channels: number of output channels (after the 1x1 expansion convolution)
        layers: number of bottleneck layers
        downsample_method: 'conv' for strided convolution, 'pool' for max pooling

        """
        super().__init__()

        if downsample_method == 'conv':

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(
                        in_channels=in_channels, hidden_channels=hidden_channels,
                        up_channels=up_channels, downsample=True,
                    )
                else:
                    layer = EncoderBottleneckLayer(
                        in_channels=up_channels, hidden_channels=hidden_channels,
                        up_channels=up_channels, downsample=False,
                    )

                self.add_module(f'{i} EncoderLayer', layer)

        elif downsample_method == 'pool':

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(
                        in_channels=in_channels, hidden_channels=hidden_channels,
                        up_channels=up_channels, downsample=False,
                    )
                else:
                    layer = EncoderBottleneckLayer(
                        in_channels=up_channels, hidden_channels=hidden_channels,
                        up_channels=up_channels, downsample=False,
                    )

                self.add_module(f'{i + 1} EncoderLayer', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all bottleneck layers sequentially."""
        for _name, layer in self.named_children():
            x = layer(x)
        return x


class DecoderResidualBlock(nn.Module):
    """Stack of residual layers with optional spatial upsampling for the decoder."""

    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        layers: int,
    ) -> None:
        """Build decoder residual block with upsampling applied on the last layer.

        Parameters
        ----------
        hidden_channels: number of channels in each residual layer
        output_channels: number of output channels after upsampling
        layers: number of residual layers

        """
        super().__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(
                    hidden_channels=hidden_channels,
                    output_channels=output_channels,
                    upsample=True,
                )
            else:
                layer = DecoderResidualLayer(
                    hidden_channels=hidden_channels,
                    output_channels=hidden_channels,
                    upsample=False,
                )

            self.add_module(f'{i} EncoderLayer', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all decoder residual layers sequentially."""
        for _name, layer in self.named_children():
            x = layer(x)
        return x


class DecoderBottleneckBlock(nn.Module):
    """Stack of bottleneck layers with optional spatial upsampling for the decoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        down_channels: int,
        layers: int,
    ) -> None:
        """Build decoder bottleneck block with upsampling applied on the last layer.

        Parameters
        ----------
        in_channels: number of input channels
        hidden_channels: number of channels in the 3x3 convolution
        down_channels: number of output channels (after the 1x1 reduction convolution)
        layers: number of bottleneck layers

        """
        super().__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(
                    in_channels=in_channels, hidden_channels=hidden_channels,
                    down_channels=down_channels, upsample=True,
                )
            else:
                layer = DecoderBottleneckLayer(
                    in_channels=in_channels, hidden_channels=hidden_channels,
                    down_channels=in_channels, upsample=False,
                )

            self.add_module(f'{i} EncoderLayer', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all decoder bottleneck layers sequentially."""
        for _name, layer in self.named_children():
            x = layer(x)
        return x


class EncoderResidualLayer(nn.Module):
    """Single residual layer with two convolutions and an optional downsampling skip connection."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        downsample: bool,
    ) -> None:
        """Build residual layer with optional strided convolution for downsampling.

        Parameters
        ----------
        in_channels: number of input channels
        hidden_channels: number of output channels
        downsample: whether to halve the spatial resolution with stride-2 convolution

        """
        super().__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=3,
                    stride=2, padding=1, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=3,
                    stride=1, padding=1, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                    stride=2, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual layer with skip connection."""
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        x = self.relu(x)
        return x


class EncoderBottleneckLayer(nn.Module):
    """Single bottleneck layer with 1x1/3x3/1x1 convolutions and an optional downsampling skip."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        up_channels: int,
        downsample: bool,
    ) -> None:
        """Build bottleneck layer with 1x1/3x3/1x1 convolutions and optional downsampling.

        Parameters
        ----------
        in_channels: number of input channels
        hidden_channels: number of channels in the 3x3 convolution
        up_channels: number of output channels (1x1 expansion)
        downsample: whether to halve the spatial resolution with stride-2 convolution

        """
        super().__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                    stride=2, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                    stride=1, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=up_channels, kernel_size=1,
                stride=1, padding=0, bias=False,
            ),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=up_channels, kernel_size=1,
                    stride=2, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif in_channels != up_channels:
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=up_channels, kernel_size=1,
                    stride=1, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck layer with skip connection."""
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)
        x = x + identity
        x = self.relu(x)
        return x


class DecoderResidualLayer(nn.Module):
    """Single residual layer with two convolutions and an optional upsampling skip connection."""

    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        upsample: bool,
    ) -> None:
        """Build decoder residual layer with optional transposed convolution for upsampling.

        Parameters
        ----------
        hidden_channels: number of input and intermediate channels
        output_channels: number of output channels
        upsample: whether to double the spatial resolution with transposed convolution

        """
        super().__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            ),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels, out_channels=output_channels, kernel_size=3,
                    stride=2, padding=1, output_padding=1, bias=False,
                ),
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=hidden_channels, out_channels=output_channels, kernel_size=3,
                    stride=1, padding=1, bias=False,
                ),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels, out_channels=output_channels, kernel_size=1,
                    stride=2, output_padding=1, bias=False,
                ),
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decoder residual layer with skip connection."""
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        if self.upsample is not None:
            identity = self.upsample(identity)
        x = x + identity
        return x


class DecoderBottleneckLayer(nn.Module):
    """Single bottleneck layer with 1x1/3x3/1x1 convolutions and an optional upsampling skip."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        down_channels: int,
        upsample: bool,
    ) -> None:
        """Build decoder bottleneck layer with 1x1/3x3/1x1 convolutions and optional upsampling.

        Parameters
        ----------
        in_channels: number of input channels
        hidden_channels: number of channels in the 3x3 convolution
        down_channels: number of output channels (1x1 reduction)
        upsample: whether to double the spatial resolution with transposed convolution

        """
        super().__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1,
                padding=0, bias=False,
            ),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            ),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels, out_channels=down_channels, kernel_size=1,
                    stride=2, output_padding=1, bias=False,
                ),
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=hidden_channels, out_channels=down_channels, kernel_size=1,
                    stride=1, padding=0, bias=False,
                ),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2,
                    output_padding=1, bias=False,
                ),
            )
        elif in_channels != down_channels:
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1,
                    padding=0, bias=False,
                ),
            )
        else:
            self.upsample = None
            self.down_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decoder bottleneck layer with skip connection."""
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)
        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)
        x = x + identity
        return x
