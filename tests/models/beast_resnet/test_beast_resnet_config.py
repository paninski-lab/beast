"""Tests for beast.models.beast_resnet.beast_resnet_config."""

import pytest
from pydantic import ValidationError

from beast.models.beast_resnet.beast_resnet_config import ResnetBeastConfig, ResnetModelParams

_MINIMAL_RESNET = {
    'model': {'model_class': 'resnet', 'model_params': {}},
    'training': {'train_batch_size': 32, 'val_batch_size': 64},
    'optimizer': {'lr': 1e-4},
    'data': {'data_dir': '/path/to/data'},
}


class TestResnetBeastConfig:
    """Test the ResnetBeastConfig model."""

    def test_valid_config(self) -> None:
        ResnetBeastConfig.model_validate(_MINIMAL_RESNET)

    def test_wrong_model_class_raises(self) -> None:
        raw = {**_MINIMAL_RESNET, 'model': {'model_class': 'vit', 'model_params': {}}}
        with pytest.raises(ValidationError):
            ResnetBeastConfig.model_validate(raw)

    def test_missing_train_batch_size_raises(self) -> None:
        raw = {**_MINIMAL_RESNET, 'training': {'val_batch_size': 64}}
        with pytest.raises(ValidationError):
            ResnetBeastConfig.model_validate(raw)

    @pytest.mark.parametrize('section', ['optimizer', 'data'])
    def test_missing_section_raises(self, section: str) -> None:
        raw = {k: v for k, v in _MINIMAL_RESNET.items() if k != section}
        with pytest.raises(ValidationError):
            ResnetBeastConfig.model_validate(raw)

    def test_model_dump_returns_plain_dicts(self) -> None:
        dumped = ResnetBeastConfig.model_validate(_MINIMAL_RESNET).model_dump()
        assert isinstance(dumped['model']['model_params'], dict)
        assert isinstance(dumped['training'], dict)
        assert isinstance(dumped['optimizer'], dict)
        assert dumped['training']['train_probability'] == 0.95
        assert dumped['training']['imgaug'] == 'default'


class TestResnetModelParams:
    """Test the ResnetModelParams model."""

    def test_defaults_applied(self) -> None:
        cfg = ResnetModelParams()
        assert cfg.backbone == 'resnet18'
        assert cfg.num_latents is None
        assert cfg.image_size == 224
        assert cfg.num_channels == 3

    def test_invalid_backbone_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResnetModelParams(backbone='resnet9')  # type: ignore[arg-type]
