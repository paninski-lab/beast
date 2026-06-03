"""Shared fixtures for beast/preprocess tests."""

import pytest

from beast.preprocess.config_3d import Beast3DConfig


@pytest.fixture
def config_dirs(tmp_path):
    """Create the minimum directory structure required by validate_config."""
    (tmp_path / 'videos').mkdir()
    (tmp_path / 'calibrations').mkdir()
    return tmp_path


@pytest.fixture
def valid_cfg(config_dirs):
    """Minimal Beast3DConfig with all required fields and real tmp directories."""
    return Beast3DConfig(
        name='test_dataset',
        input_dir=str(config_dirs),
        output_dir=str(config_dirs / 'output'),
        anchor_view='cam0',
    )
