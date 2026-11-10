"""Tests for beast.configs.base."""

import pytest

from beast.configs.base import BaseOptimizerConfig, BaseTrainingConfig
from beast.configs.models import TrainingConfig
from beast.models.erayzer.erayzer_config import ERayZerTrainingConfig


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
        )
        assert (std.num_epochs, std.check_val_every_n_epoch, std.viz_num_frames) == (800, 5, 108)
        assert (ray.num_epochs, ray.check_val_every_n_epoch, ray.viz_num_frames) == (800, 1, 8)
        assert ray.train_frames is None and ray.precision == '32-true'


class TestBaseOptimizerConfig:
    """Test the shared optimizer schema."""

    def test_defaults_applied(self):
        cfg = BaseOptimizerConfig(lr=1e-4)
        assert cfg.type == 'AdamW'
        assert cfg.scheduler == 'cosine'
        assert cfg.accumulate_grad_batches == 1
        assert (cfg.beta1, cfg.beta2) == (0.9, 0.95)
        assert cfg.warmup_pct == 0.15
        assert cfg.steps == []
        assert cfg.scale_lr_by_batch is True

    def test_constant_scheduler_accepted(self):
        assert BaseOptimizerConfig(lr=1e-4, scheduler='constant').scheduler == 'constant'

    def test_unknown_scheduler_raises(self):
        with pytest.raises(ValueError):
            BaseOptimizerConfig(lr=1e-4, scheduler='bogus')  # type: ignore[arg-type]

    def test_base_optimizer_config_requires_lr(self):
        with pytest.raises(ValueError):
            BaseOptimizerConfig()  # type: ignore[call-arg]
