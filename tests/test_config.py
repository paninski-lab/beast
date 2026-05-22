"""Tests for Pydantic configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from beast.config import (
    BeastConfig,
    OptimizerConfig,
    ResnetModelParams,
    TrainingConfig,
    VitModelParams,
)
from beast.io import load_config

_CONFIGS_DIR = Path(__file__).parent.parent / 'configs'
_CONFIG_FILES = list(_CONFIGS_DIR.glob('*.yaml'))

_MINIMAL_RESNET = {
    'model': {'model_class': 'resnet', 'model_params': {}},
    'training': {'train_batch_size': 32, 'val_batch_size': 64},
    'optimizer': {'lr': 1e-4},
    'data': {'data_dir': '/path/to/data'},
}

_MINIMAL_VIT = {
    'model': {'model_class': 'vit', 'model_params': {}},
    'training': {'train_batch_size': 32, 'val_batch_size': 64},
    'optimizer': {'lr': 1e-4},
    'data': {'data_dir': '/path/to/data'},
}


class TestBeastConfig:
    """Test the BeastConfig model."""

    def test_valid_resnet_config(self) -> None:
        # Arrange / Act / Assert
        BeastConfig.model_validate(_MINIMAL_RESNET)

    def test_valid_vit_config(self) -> None:
        BeastConfig.model_validate(_MINIMAL_VIT)

    def test_unknown_model_class_raises(self) -> None:
        raw = {**_MINIMAL_RESNET, 'model': {'model_class': 'unknown', 'model_params': {}}}
        with pytest.raises(ValidationError):
            BeastConfig.model_validate(raw)

    def test_missing_train_batch_size_raises(self) -> None:
        raw = {**_MINIMAL_VIT, 'training': {'val_batch_size': 64}}
        with pytest.raises(ValidationError):
            BeastConfig.model_validate(raw)

    def test_missing_optimizer_raises(self) -> None:
        raw = {k: v for k, v in _MINIMAL_VIT.items() if k != 'optimizer'}
        with pytest.raises(ValidationError):
            BeastConfig.model_validate(raw)

    def test_missing_data_raises(self) -> None:
        raw = {k: v for k, v in _MINIMAL_VIT.items() if k != 'data'}
        with pytest.raises(ValidationError):
            BeastConfig.model_validate(raw)

    def test_model_dump_returns_plain_dicts(self) -> None:
        config = BeastConfig.model_validate(_MINIMAL_VIT)
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        assert isinstance(dumped['model'], dict)
        assert isinstance(dumped['model']['model_params'], dict)
        assert isinstance(dumped['training'], dict)
        assert isinstance(dumped['optimizer'], dict)


class TestTrainingConfig:
    """Test the TrainingConfig model."""

    def test_defaults_applied(self) -> None:
        cfg = TrainingConfig(train_batch_size=32, val_batch_size=64)
        assert cfg.seed == 0
        assert cfg.num_gpus == 1
        assert cfg.num_nodes == 1
        assert cfg.num_workers == 8
        assert cfg.train_probability == 0.95
        assert cfg.ckpt_every_n_epochs is None

    def test_missing_train_batch_size_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(val_batch_size=64)  # type: ignore[call-arg]

    def test_missing_val_batch_size_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(train_batch_size=32)  # type: ignore[call-arg]


class TestOptimizerConfig:
    """Test the OptimizerConfig model."""

    def test_defaults_applied(self) -> None:
        cfg = OptimizerConfig(lr=1e-4)
        assert cfg.type == 'AdamW'
        assert cfg.scheduler == 'cosine'
        assert cfg.accumulate_grad_batches == 1
        assert cfg.warmup_pct == 0.15
        assert cfg.steps == []

    def test_missing_lr_raises(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig()  # type: ignore[call-arg]


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


class TestVitModelParams:
    """Test the VitModelParams model."""

    def test_defaults_applied(self) -> None:
        cfg = VitModelParams()
        assert cfg.hidden_size == 768
        assert cfg.mask_ratio == 0.75
        assert cfg.use_infoNCE is False
        assert cfg.use_perceptual_loss is False


class TestConfigFiles:
    """Validate all config files in the configs/ directory."""

    @pytest.mark.parametrize('config_path', _CONFIG_FILES, ids=[p.name for p in _CONFIG_FILES])
    def test_config_is_valid(self, config_path: Path) -> None:
        config = load_config(config_path)
        assert isinstance(config, dict)
