"""Tests for beast.models.msps_vae.msps_vae_config."""

from typing import Any

import pytest
from pydantic import ValidationError

from beast.models.msps_vae.msps_vae_config import MspsVaeBeastConfig, MspsVaeModelParams

_MINIMAL_MODEL_PARAMS: dict[str, Any] = {
    'num_latents_unsupervised': 8,
    'num_latents_background': 4,
}

_MINIMAL_MSPS_VAE = {
    'model': {'model_class': 'msps_vae', 'model_params': _MINIMAL_MODEL_PARAMS},
    'training': {'train_batch_size': 32, 'val_batch_size': 64},
    'optimizer': {'lr': 1e-4},
    'data': {'data_dir': '/path/to/data'},
}


class TestMspsVaeBeastConfig:
    """Test the MspsVaeBeastConfig model."""

    def test_valid_config(self) -> None:
        MspsVaeBeastConfig.model_validate(_MINIMAL_MSPS_VAE)

    def test_wrong_model_class_raises(self) -> None:
        raw = {
            **_MINIMAL_MSPS_VAE,
            'model': {'model_class': 'resnet', 'model_params': _MINIMAL_MODEL_PARAMS},
        }
        with pytest.raises(ValidationError):
            MspsVaeBeastConfig.model_validate(raw)

    def test_missing_train_batch_size_raises(self) -> None:
        raw = {**_MINIMAL_MSPS_VAE, 'training': {'val_batch_size': 64}}
        with pytest.raises(ValidationError):
            MspsVaeBeastConfig.model_validate(raw)

    @pytest.mark.parametrize('section', ['optimizer', 'data'])
    def test_missing_section_raises(self, section: str) -> None:
        raw = {k: v for k, v in _MINIMAL_MSPS_VAE.items() if k != section}
        with pytest.raises(ValidationError):
            MspsVaeBeastConfig.model_validate(raw)

    def test_model_dump_fills_defaults(self) -> None:
        dumped = MspsVaeBeastConfig.model_validate(_MINIMAL_MSPS_VAE).model_dump()
        params = dumped['model']['model_params']
        assert isinstance(params, dict)
        assert params['positive_window'] == 1000
        assert params['use_spatial_loss_weight'] is False
        assert params['spatial_loss_weight_r0'] == 0.5
        assert dumped['training']['train_probability'] == 0.95


class TestMspsVaeModelParams:
    """Test the MspsVaeModelParams model."""

    def test_defaults_applied(self) -> None:
        cfg = MspsVaeModelParams(**_MINIMAL_MODEL_PARAMS)
        assert cfg.backbone == 'resnet18'
        assert cfg.image_size == 224
        assert cfg.num_channels == 3
        assert cfg.triplet_margin == 1.0
        assert cfg.triplet_weight == 1.0
        assert cfg.positive_window == 1000
        assert cfg.orthogonal_matrix_seed == 42
        assert cfg.use_spatial_loss_weight is False
        assert cfg.spatial_loss_weight_r0 == 0.5

    @pytest.mark.parametrize('field', ['num_latents_unsupervised', 'num_latents_background'])
    def test_missing_latents_raises(self, field: str) -> None:
        params = {k: v for k, v in _MINIMAL_MODEL_PARAMS.items() if k != field}
        with pytest.raises(ValidationError):
            MspsVaeModelParams(**params)

    def test_invalid_backbone_raises(self) -> None:
        with pytest.raises(ValidationError):
            MspsVaeModelParams(backbone='resnet9', **_MINIMAL_MODEL_PARAMS)  # type: ignore[arg-type]
