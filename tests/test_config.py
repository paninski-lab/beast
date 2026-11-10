"""Tests for beast.config."""

from pathlib import Path

import pytest

from beast.config import BaseBeastConfig, BaseOptimizerConfig, BaseTrainingConfig
from beast.io import load_config
from beast.models.erayzer.erayzer_config import ERayZerTrainingConfig

_CONFIGS_DIR = Path(__file__).parent.parent / 'configs'
_NON_BEAST_CONFIG_NAMES = {'extraction_pipeline.yaml'}
_CONFIG_FILES = sorted(
    p for p in _CONFIGS_DIR.rglob('*.yaml') if p.name not in _NON_BEAST_CONFIG_NAMES
)


class TestBaseTrainingConfig:
    """Test the shared training schema and its subclasses."""

    def test_base_training_config_subclasses(self):
        assert issubclass(ERayZerTrainingConfig, BaseTrainingConfig)

    def test_base_training_config_shared_fields_present_everywhere(self):
        shared = set(BaseTrainingConfig.model_fields)
        assert shared <= set(ERayZerTrainingConfig.model_fields)

    def test_base_training_config_requires_batch_sizes(self):
        with pytest.raises(ValueError):
            BaseTrainingConfig()  # type: ignore[call-arg]

    def test_base_training_config_subclass_default_overrides(self):
        std = BaseTrainingConfig(train_batch_size=4, val_batch_size=4)
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
        assert (std.train_probability, ray.train_probability) == (0.95, 0.9)

    def test_base_training_config_split_and_augmentation_defaults(self):
        cfg = BaseTrainingConfig(train_batch_size=4, val_batch_size=4)
        assert (cfg.train_probability, cfg.val_probability, cfg.imgaug) == (0.95, 0.05, 'default')
        with pytest.raises(ValueError):
            BaseTrainingConfig(
                train_batch_size=4, val_batch_size=4, imgaug='bogus',  # type: ignore[arg-type]
            )


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


class TestBaseBeastConfig:
    """Test the shared top-level config fields."""

    def test_fields(self):
        assert set(BaseBeastConfig.model_fields) == {'optimizer', 'data', 'inference'}

    def test_requires_optimizer_and_data(self):
        with pytest.raises(ValueError):
            BaseBeastConfig.model_validate({'optimizer': {'lr': 1e-4}})
        cfg = BaseBeastConfig.model_validate(
            {'optimizer': {'lr': 1e-4}, 'data': {'data_dir': '/x'}},
        )
        assert cfg.inference is False


class TestConfigFiles:
    """Validate all config files in the configs/ directory."""

    @pytest.mark.parametrize(
        'config_path',
        _CONFIG_FILES,
        ids=[str(p.relative_to(_CONFIGS_DIR)) for p in _CONFIG_FILES],
    )
    def test_config_is_valid(self, config_path: Path) -> None:
        config = load_config(config_path)
        assert isinstance(config, dict)
