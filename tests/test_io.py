"""Tests for configuration loading and override utilities."""

import copy

import pytest

from beast.io import apply_config_overrides, load_config


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_valid(self, config_ae_path) -> None:
        # Arrange / Act
        config = load_config(config_ae_path)
        # Assert
        assert isinstance(config, dict)

    def test_load_config_bad_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config('/fake/path')


class TestApplyConfigOverrides:
    """Test the apply_config_overrides function."""

    def test_override_existing_field_with_dict(self, config_ae) -> None:
        # Arrange
        overrides = {'model.seed': 1}
        # Act
        new_config = apply_config_overrides(copy.deepcopy(config_ae), overrides)
        # Assert
        assert new_config['model']['seed'] == overrides['model.seed']

    def test_add_new_fields_with_dict(self, config_ae) -> None:
        overrides = {
            'data': '/path/to/data',
            'model.seed': 2,
            'model.model_params.batchnorm': True,
        }
        new_config = apply_config_overrides(copy.deepcopy(config_ae), overrides)
        assert new_config['data'] == overrides['data']
        assert new_config['model']['seed'] == overrides['model.seed']
        assert new_config['model']['model_params']['batchnorm'] == overrides[
            'model.model_params.batchnorm'
        ]

    def test_override_existing_fields_with_list(self, config_ae) -> None:
        overrides = ['model.seed=1', 'training.imgaug=geometric']
        new_config = apply_config_overrides(copy.deepcopy(config_ae), overrides)
        assert new_config['model']['seed'] == '1'
        assert new_config['training']['imgaug'] == 'geometric'

    def test_creates_new_nested_keys(self, config_ae) -> None:
        # Arrange — 'brand.new.nested' doesn't exist in config; both parent keys must be created
        overrides = {'brand.new.nested': 'value'}
        # Act
        new_config = apply_config_overrides(copy.deepcopy(config_ae), overrides)
        # Assert
        assert new_config['brand']['new']['nested'] == 'value'
