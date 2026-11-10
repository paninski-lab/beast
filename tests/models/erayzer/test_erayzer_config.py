"""Tests for beast.models.erayzer.erayzer_config."""

import pytest
from pydantic import ValidationError

from beast.models.erayzer.erayzer_config import ERayZerBeastConfig, ERayZerTrainingConfig

_MINIMAL_ERAYZER = {
    'model': {
        'model_class': 'erayzer',
        'transformer': {'d': 256, 'd_head': 64, 'encoder_geom_n_layer': 4},
    },
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


class TestERayZerTrainingConfig:
    """Test the ERayZerTrainingConfig model."""

    def test_defaults_applied(self) -> None:
        cfg = ERayZerTrainingConfig(
            train_batch_size=4,
            val_batch_size=2,
            num_views=3,
            num_input_views=2,
            num_target_views=1,
        )
        assert cfg.num_epochs == 800
        assert cfg.seed == 0
        assert cfg.train_probability == 0.9
        assert cfg.l2_loss_weight == 1.0
        assert cfg.random_split is False
        assert cfg.ckpt_every_n_epochs is None

    def test_no_step_based_or_duplicate_split_fields(self) -> None:
        assert 'max_fwdbwd_passes' not in ERayZerTrainingConfig.model_fields
        assert 'train_fraction' not in ERayZerTrainingConfig.model_fields

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            ERayZerTrainingConfig(train_batch_size=4, val_batch_size=2)  # type: ignore[call-arg]


class TestERayZerBeastConfig:
    """Test the ERayZerBeastConfig model."""

    def test_valid_config(self) -> None:
        ERayZerBeastConfig.model_validate(_MINIMAL_ERAYZER)

    def test_missing_required_training_field_raises(self) -> None:
        raw = {**_MINIMAL_ERAYZER, 'training': {'train_batch_size': 4, 'val_batch_size': 2}}
        with pytest.raises(ValidationError):
            ERayZerBeastConfig.model_validate(raw)

    def test_wrong_model_class_raises(self) -> None:
        raw = {**_MINIMAL_ERAYZER, 'model': {**_MINIMAL_ERAYZER['model'], 'model_class': 'vit'}}
        with pytest.raises(ValidationError):
            ERayZerBeastConfig.model_validate(raw)

    def test_model_dump_contains_erayzer_fields(self) -> None:
        dumped = ERayZerBeastConfig.model_validate(_MINIMAL_ERAYZER).model_dump()
        assert dumped['training']['num_views'] == 3
        assert 'beta1' in dumped['optimizer']
        assert 'warmup' not in dumped['optimizer']
        assert 'max_fwdbwd_passes' not in dumped['training']
