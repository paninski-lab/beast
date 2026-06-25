"""Tests for Pydantic configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from beast.config import (
    Beast3DBeastConfig,
    BeastConfig,
    ERayZerBeastConfig,
    OptimizerConfig,
    TrainingConfig,
    get_beast_config_class,
)
from beast.io import load_config
from beast.models.beast3d.beast3d_config import Beast3DModelConfig
from beast.models.beast_resnet.beast_resnet_config import ResnetModelParams
from beast.models.beast_vit.beast_vit_config import VitModelParams
from beast.models.erayzer.erayzer_config import ERayZerOptimizerConfig, ERayZerTrainingConfig

_CONFIGS_DIR = Path(__file__).parent.parent / 'configs'
_NON_BEAST_CONFIG_NAMES = {'extraction_pipeline.yaml'}
_CONFIG_FILES = sorted(
    p for p in _CONFIGS_DIR.rglob('*.yaml') if p.name not in _NON_BEAST_CONFIG_NAMES
)

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
        'max_fwdbwd_passes': 50000,
    },
    'optimizer': {'lr': 4e-4},
    'data': {'data_dir': '/path/to/data'},
}

_MIN_TRANSFORMER = {'d': 768, 'd_head': 64, 'encoder_geom_n_layer': 16}


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


class TestERayZerTrainingConfig:
    """Test the ERayZerTrainingConfig model."""

    def test_defaults_applied(self) -> None:
        cfg = ERayZerTrainingConfig(
            train_batch_size=4,
            val_batch_size=2,
            num_views=3,
            num_input_views=2,
            num_target_views=1,
            max_fwdbwd_passes=50000,
        )
        assert cfg.num_epochs == 200
        assert cfg.seed == 0
        assert cfg.train_fraction == 0.9
        assert cfg.l2_loss_weight == 1.0
        assert cfg.random_split is False
        assert cfg.ckpt_every_n_epochs is None

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            ERayZerTrainingConfig(train_batch_size=4, val_batch_size=2)  # type: ignore[call-arg]


class TestERayZerOptimizerConfig:
    """Test the ERayZerOptimizerConfig model."""

    def test_defaults_applied(self) -> None:
        cfg = ERayZerOptimizerConfig(lr=4e-4)
        assert cfg.beta1 == 0.9
        assert cfg.beta2 == 0.95
        assert cfg.wd == 0.05
        assert cfg.warmup == 3000
        assert cfg.div_factor == 1.0
        assert cfg.accumulate_grad_batches == 1

    def test_missing_lr_raises(self) -> None:
        with pytest.raises(ValidationError):
            ERayZerOptimizerConfig()  # type: ignore[call-arg]


class TestERayZerBeastConfig:
    """Test the ERayZerBeastConfig model."""

    def test_valid_config(self) -> None:
        ERayZerBeastConfig.model_validate(_MINIMAL_ERAYZER)

    def test_missing_required_training_field_raises(self) -> None:
        raw = {**_MINIMAL_ERAYZER, 'training': {'train_batch_size': 4, 'val_batch_size': 2}}
        with pytest.raises(ValidationError):
            ERayZerBeastConfig.model_validate(raw)

    def test_model_dump_contains_erayzer_fields(self) -> None:
        config = ERayZerBeastConfig.model_validate(_MINIMAL_ERAYZER)
        dumped = config.model_dump()
        assert dumped['training']['num_views'] == 3
        assert dumped['training']['max_fwdbwd_passes'] == 50000
        assert 'beta1' in dumped['optimizer']


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


class TestGetBeastConfigClass:
    """Test the get_beast_config_class dispatcher."""

    def test_erayzer_returns_erayzer_beast_config(self) -> None:
        assert get_beast_config_class('erayzer') is ERayZerBeastConfig

    def test_beast3d_returns_beast3d_beast_config(self) -> None:
        assert get_beast_config_class('beast3d') is Beast3DBeastConfig

    def test_resnet_falls_back_to_beast_config(self) -> None:
        assert get_beast_config_class('resnet') is BeastConfig

    def test_vit_falls_back_to_beast_config(self) -> None:
        assert get_beast_config_class('vit') is BeastConfig

    def test_unknown_falls_back_to_beast_config(self) -> None:
        assert get_beast_config_class('totally_unknown_model') is BeastConfig

    def test_empty_string_falls_back_to_beast_config(self) -> None:
        assert get_beast_config_class('') is BeastConfig


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
