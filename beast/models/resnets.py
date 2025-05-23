"""Resnet-based autoencoder implementation.

Adapted from https://github.com/Horizon2333/imagenet-autoencoder

"""

from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float
from typeguard import typechecked

from beast.models.base import BaseLightningModel


@typechecked
def get_configs(arch='resnet18') -> tuple:
    # True or False means wether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError(f'{arch} is an invalid entry in model.model_params.backbone')


@typechecked
class ResnetAutoencoder(BaseLightningModel):
    """Vision Transformer implementation."""

    def __init__(self, config: dict) -> None:

        super().__init__(config)

        resnet_config, bottleneck = get_configs(config['model']['model_params']['backbone'])
        self.encoder = ResNetEncoder(configs=resnet_config, bottleneck=bottleneck)
        self.decoder = ResNetDecoder(configs=resnet_config[::-1], bottleneck=bottleneck)

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> tuple[
        Float[torch.Tensor, 'batch channels img_height img_width'],
        Float[torch.Tensor, 'batch features feat_height feat_width'],
    ]:
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

    def get_model_outputs(self, batch_dict: dict) -> dict:
        xhat, z = self.forward(batch_dict['image'])
        results_dict = {
            'reconstructions': xhat,
            'latents': z,
        }
        return results_dict

    def compute_loss(
        self,
        stage: str,
        reconstructions: torch.tensor,
        latents: torch.tensor,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        pass

    def predict_step(self, batch_dict: dict, batch_idx: int) -> torch.tensor:
        pass


class ResNetEncoder(nn.Module):

    def __init__(self, configs: list, bottleneck: bool = False) -> None:

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class ResNetDecoder(nn.Module):

    def __init__(self, configs: list, bottleneck: bool = False) -> None:

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

        self.gate = nn.Sigmoid()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)
        return x


class EncoderResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        layers: int,
        downsample_method: Literal['conv', 'pool'] = 'conv',
    ) -> None:

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        for name, layer in self.named_children():
            x = layer(x)
        return x


class EncoderBottleneckBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        up_channels: int,
        layers: int,
        downsample_method: Literal['conv', 'pool'] = 'conv',
    ) -> None:

        super(EncoderBottleneckBlock, self).__init__()

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        for name, layer in self.named_children():
            x = layer(x)
        return x


class DecoderResidualBlock(nn.Module):

    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        layers: int,
    ) -> None:

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        for name, layer in self.named_children():
            x = layer(x)
        return x


class DecoderBottleneckBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        down_channels: int,
        layers: int,
    ) -> None:

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        for name, layer in self.named_children():
            x = layer(x)
        return x


class EncoderResidualLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        downsample: bool,
    ) -> None:

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        x = self.relu(x)
        return x


class EncoderBottleneckLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        up_channels: int,
        downsample: bool,
    ) -> None:

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

    def forward(self, x: torch.tensor) -> torch.tensor:
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

    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        upsample: bool,
    ) -> None:

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        if self.upsample is not None:
            identity = self.upsample(identity)
        x = x + identity
        return x


class DecoderBottleneckLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        down_channels: int,
        upsample: bool,
    ) -> None:

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

    def forward(self, x: torch.tensor) -> torch.tensor:
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
