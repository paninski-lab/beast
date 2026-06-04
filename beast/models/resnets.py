"""Compatibility shim — import from beast.models.beast_resnet instead."""

from beast.models.beast_resnet.beast_resnet_model import (
    _RESNET_CONFIGS,
    DecoderBottleneckBlock,
    DecoderBottleneckLayer,
    DecoderResidualBlock,
    DecoderResidualLayer,
    EncoderBottleneckBlock,
    EncoderBottleneckLayer,
    EncoderResidualBlock,
    EncoderResidualLayer,
    LatentMapping,
    ResnetAutoencoder,
    ResNetDecoder,
    ResNetEncoder,
    get_configs,
)

__all__ = [
    '_RESNET_CONFIGS',
    'get_configs',
    'ResnetAutoencoder',
    'LatentMapping',
    'ResNetEncoder',
    'ResNetDecoder',
    'EncoderResidualBlock',
    'EncoderBottleneckBlock',
    'DecoderResidualBlock',
    'DecoderBottleneckBlock',
    'EncoderResidualLayer',
    'EncoderBottleneckLayer',
    'DecoderResidualLayer',
    'DecoderBottleneckLayer',
]
