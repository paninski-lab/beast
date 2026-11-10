"""Tests for beast.models.beast3d.beast3d_config."""

import pytest
from pydantic import ValidationError

from beast.models.beast3d.beast3d_config import Beast3DBeastConfig, Beast3DModelConfig
from beast.models.erayzer.erayzer_config import ERayZerTrainingConfig

_MIN_TRANSFORMER = {'d': 768, 'd_head': 64, 'encoder_geom_n_layer': 16}


class TestBeast3DModelConfig:
    """Test the Beast3DModelConfig schema."""

    def test_valid_config_defaults(self) -> None:
        cfg = Beast3DModelConfig.model_validate(
            {'model_class': 'beast3d', 'transformer': _MIN_TRANSFORMER},
        )
        assert cfg.model_class == 'beast3d'
        assert cfg.use_dinov3 is True
        assert cfg.freeze_dinov3 is True
        assert cfg.frustum_constraint is True
        assert cfg.random_background is True
        assert cfg.mask_loss_weight == 0.1

    def test_wrong_model_class_raises(self) -> None:
        with pytest.raises(ValidationError):
            Beast3DModelConfig.model_validate(
                {'model_class': 'erayzer', 'transformer': _MIN_TRANSFORMER},
            )

    def test_missing_transformer_raises(self) -> None:
        with pytest.raises(ValidationError):
            Beast3DModelConfig.model_validate({'model_class': 'beast3d'})

    def test_inherits_erayzer_fields(self) -> None:
        # hard_pixelalign and gaussians come from the ERayZer base config
        cfg = Beast3DModelConfig.model_validate(
            {'model_class': 'beast3d', 'transformer': _MIN_TRANSFORMER, 'hard_pixelalign': True},
        )
        assert cfg.hard_pixelalign is True
        assert cfg.gaussians.sh_degree == 3


class TestBeast3DBeastConfig:
    """Test the Beast3DBeastConfig model."""

    _RAW = {
        'model': {'model_class': 'beast3d', 'transformer': _MIN_TRANSFORMER},
        'training': {
            'train_batch_size': 4,
            'val_batch_size': 2,
            'num_views': 3,
            'num_input_views': 2,
            'num_target_views': 1,
        },
        'optimizer': {'lr': 4e-4},
        'data': {'data_dir': '/path/to/data'},
    }

    def test_valid_config(self) -> None:
        cfg = Beast3DBeastConfig.model_validate(self._RAW)
        assert isinstance(cfg.training, ERayZerTrainingConfig)

    def test_erayzer_model_class_rejected(self) -> None:
        raw = {**self._RAW, 'model': {'model_class': 'erayzer', 'transformer': _MIN_TRANSFORMER}}
        with pytest.raises(ValidationError):
            Beast3DBeastConfig.model_validate(raw)
