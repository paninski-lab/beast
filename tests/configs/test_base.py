"""Tests for beast.configs.base."""

import pytest

from beast.configs.base import BaseOptimizerConfig, BaseTrainingConfig
from beast.configs.models import OptimizerConfig, TrainingConfig
from beast.models.erayzer.erayzer_config import ERayZerOptimizerConfig, ERayZerTrainingConfig


class TestBaseTrainingConfig:
    """Test the shared training schema and its subclasses."""

    def test_base_training_config_subclasses(self):
        assert issubclass(TrainingConfig, BaseTrainingConfig)
        assert issubclass(ERayZerTrainingConfig, BaseTrainingConfig)

    def test_base_training_config_shared_fields_present_everywhere(self):
        shared = set(BaseTrainingConfig.model_fields)
        assert shared <= set(TrainingConfig.model_fields)
        assert shared <= set(ERayZerTrainingConfig.model_fields)

    def test_base_training_config_requires_batch_sizes(self):
        with pytest.raises(ValueError):
            BaseTrainingConfig()  # type: ignore[call-arg]

    def test_base_training_config_subclass_default_overrides(self):
        std = TrainingConfig(train_batch_size=4, val_batch_size=4)
        ray = ERayZerTrainingConfig(
            train_batch_size=4,
            val_batch_size=4,
            num_views=4,
            num_input_views=2,
            num_target_views=2,
            max_fwdbwd_passes=10,
        )
        assert (std.num_epochs, std.check_val_every_n_epoch, std.viz_num_frames) == (800, 5, 108)
        assert (ray.num_epochs, ray.check_val_every_n_epoch, ray.viz_num_frames) == (200, 1, 8)
        assert ray.train_frames is None and ray.precision == '32-true'


class TestBaseOptimizerConfig:
    """Test the shared optimizer schema and its subclasses."""

    def test_base_optimizer_config_subclasses(self):
        assert issubclass(OptimizerConfig, BaseOptimizerConfig)
        assert issubclass(ERayZerOptimizerConfig, BaseOptimizerConfig)

    def test_base_optimizer_config_shared_fields_present_everywhere(self):
        shared = set(BaseOptimizerConfig.model_fields)
        assert shared == {'lr', 'wd', 'beta1', 'beta2', 'accumulate_grad_batches'}
        assert shared <= set(OptimizerConfig.model_fields)
        assert shared <= set(ERayZerOptimizerConfig.model_fields)

    def test_base_optimizer_config_requires_lr(self):
        with pytest.raises(ValueError):
            BaseOptimizerConfig()  # type: ignore[call-arg]
