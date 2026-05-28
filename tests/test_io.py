"""Tests for configuration loading, override utilities, and camera calibration."""

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from beast.io import (
    apply_config_overrides,
    get_calibration_path,
    get_camera_params_for_view,
    load_anipose_calibration,
    load_bbox_csv,
    load_calibration,
    load_config,
    map_cam_name,
)
from beast.preprocess.config_3d import Beast3DConfig, CalibrationConfig


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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _minimal_cfg(tmp_path: Path, **kwargs) -> Beast3DConfig:
    """Return a minimal Beast3DConfig rooted at tmp_path."""
    return Beast3DConfig(
        name='test',
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / 'out'),
        anchor_view='cam0',
        **kwargs,
    )


@pytest.fixture
def mock_aniposelib():
    """Patch beast.io.CameraGroupAnipose with a mock camera group."""
    mock_cam = MagicMock()
    mock_cam.name = 'Cam0'
    mock_cam.get_camera_matrix.return_value = np.eye(3)
    mock_cam.get_extrinsics_mat.return_value = np.eye(4)
    mock_cam.get_distortions.return_value = np.zeros(5)
    mock_cam.get_size.return_value = (1920, 1080)

    mock_group = MagicMock()
    mock_group.cameras = [mock_cam]

    mock_cg_class = MagicMock()
    mock_cg_class.load.return_value = mock_group

    with patch('beast.io.CameraGroupAnipose', mock_cg_class):
        yield mock_cg_class


# ---------------------------------------------------------------------------
# TestMapCamName
# ---------------------------------------------------------------------------

class TestMapCamName:
    """Test the map_cam_name function."""

    def test_identity_returns_cam_id(self) -> None:
        assert map_cam_name('cam0', 'identity') == 'cam0'

    def test_prefix_prepends_string(self) -> None:
        assert map_cam_name('0', 'prefix_Cam-') == 'Cam-0'

    def test_format_string_interpolates_cam_id(self) -> None:
        assert map_cam_name('3', 'Camera_{cam_id}_top') == 'Camera_3_top'

    def test_unknown_mapping_returns_cam_id(self) -> None:
        assert map_cam_name('cam2', 'unrecognized_rule') == 'cam2'


# ---------------------------------------------------------------------------
# TestGetCalibrationPath
# ---------------------------------------------------------------------------

class TestGetCalibrationPath:
    """Test the get_calibration_path function."""

    def test_returns_path_with_session_in_filename(self, tmp_path) -> None:
        cfg = _minimal_cfg(tmp_path)
        result = get_calibration_path('session01', cfg)
        assert result == tmp_path / 'calibrations' / 'session01.toml'

    def test_custom_file_pattern(self, tmp_path) -> None:
        cfg = _minimal_cfg(tmp_path)
        cfg.calibration.file_pattern = 'cal_{session_id}.json'
        result = get_calibration_path('abc', cfg)
        assert result.name == 'cal_abc.json'


# ---------------------------------------------------------------------------
# TestLoadAniposeCalibration
# ---------------------------------------------------------------------------

class TestLoadAniposeCalibration:
    """Test the load_anipose_calibration function."""

    def test_returns_dict_keyed_by_camera_name(self, tmp_path, mock_aniposelib) -> None:
        result = load_anipose_calibration(tmp_path / 'cal.toml')
        assert 'Cam0' in result

    def test_parameter_dict_has_expected_keys(self, tmp_path, mock_aniposelib) -> None:
        result = load_anipose_calibration(tmp_path / 'cal.toml')
        params = result['Cam0']
        assert set(params.keys()) == {'intrinsics', 'extrinsics', 'distortions', 'width', 'height'}

    def test_intrinsics_shape(self, tmp_path, mock_aniposelib) -> None:
        result = load_anipose_calibration(tmp_path / 'cal.toml')
        assert result['Cam0']['intrinsics'].shape == (3, 3)

    def test_passes_path_as_string_to_loader(self, tmp_path, mock_aniposelib) -> None:
        cal_path = tmp_path / 'cal.toml'
        load_anipose_calibration(cal_path)
        mock_aniposelib.load.assert_called_once_with(str(cal_path))


# ---------------------------------------------------------------------------
# TestLoadCalibration
# ---------------------------------------------------------------------------

class TestLoadCalibration:
    """Test the load_calibration function."""

    def test_missing_file_returns_none(self, tmp_path) -> None:
        cfg = _minimal_cfg(tmp_path)
        result = load_calibration('no_such_session', cfg)
        assert result is None

    def test_anipose_toml_delegates_to_loader(self, tmp_path, mock_aniposelib) -> None:
        # Arrange — create the calibration file so the path check passes
        cal_dir = tmp_path / 'calibrations'
        cal_dir.mkdir()
        cal_file = cal_dir / 'sess1.toml'
        cal_file.touch()
        cfg = _minimal_cfg(tmp_path)
        # Act
        result = load_calibration('sess1', cfg)
        # Assert
        assert result is not None
        assert 'Cam0' in result

    def test_unsupported_format_raises(self, tmp_path) -> None:
        cal_dir = tmp_path / 'calibrations'
        cal_dir.mkdir()
        (cal_dir / 'sess.toml').touch()
        cfg = _minimal_cfg(tmp_path)
        cfg.calibration.format = 'unsupported_fmt'
        with pytest.raises(ValueError, match='unsupported calibration format'):
            load_calibration('sess', cfg)


# ---------------------------------------------------------------------------
# TestGetCameraParamsForView
# ---------------------------------------------------------------------------

class TestGetCameraParamsForView:
    """Test the get_camera_params_for_view function."""

    def test_known_cam_id_returns_params(self) -> None:
        cam_params = {'Cam0': {'intrinsics': np.eye(3), 'width': 1920, 'height': 1080}}
        cfg = Beast3DConfig(
            name='x', input_dir='/i', output_dir='/o', anchor_view='cam0',
            calibration=CalibrationConfig(cam_name_mapping='prefix_Cam'),
        )
        result = get_camera_params_for_view(cam_params, '0', cfg)
        assert result is not None
        assert result['width'] == 1920

    def test_unknown_cam_id_returns_none(self) -> None:
        cam_params = {'Cam0': {'intrinsics': np.eye(3)}}
        cfg = Beast3DConfig(
            name='x', input_dir='/i', output_dir='/o', anchor_view='cam0',
        )
        result = get_camera_params_for_view(cam_params, 'missing', cfg)
        assert result is None


# ---------------------------------------------------------------------------
# TestLoadBboxCsv
# ---------------------------------------------------------------------------

class TestLoadBboxCsv:
    """Test the load_bbox_csv function."""

    def _write_bbox_csv(self, path: Path) -> None:
        path.write_text(
            'Unnamed: 0,x,y,w,h\n'
            '0,10,20,100,200\n'
            '1,15,25,105,205\n'
        )

    def test_has_bboxes_false_returns_none(self, tmp_path) -> None:
        cfg = _minimal_cfg(
            tmp_path, has_bboxes=False, bbox_csv_pattern='{session_id}_{cam_id}.csv',
        )
        assert load_bbox_csv('s', 'cam0', cfg) is None

    def test_no_bbox_pattern_returns_none(self, tmp_path) -> None:
        cfg = _minimal_cfg(tmp_path, has_bboxes=True, bbox_csv_pattern='')
        assert load_bbox_csv('s', 'cam0', cfg) is None

    def test_csv_found_returns_dict(self, tmp_path) -> None:
        videos_dir = tmp_path / 'videos'
        videos_dir.mkdir()
        self._write_bbox_csv(videos_dir / 'sess_cam0.csv')
        cfg = _minimal_cfg(
            tmp_path,
            has_bboxes=True,
            bbox_csv_pattern='{session_id}_{cam_id}.csv',
        )
        result = load_bbox_csv('sess', 'cam0', cfg)
        assert result is not None
        assert result[0] == {'x': 10, 'y': 20, 'w': 100, 'h': 200}
        assert result[1] == {'x': 15, 'y': 25, 'w': 105, 'h': 205}

    def test_csv_not_found_returns_none(self, tmp_path) -> None:
        cfg = _minimal_cfg(tmp_path, has_bboxes=True, bbox_csv_pattern='{session_id}_{cam_id}.csv')
        result = load_bbox_csv('sess', 'cam0', cfg)
        assert result is None

    def test_explicit_bbox_dir_overrides_default(self, tmp_path) -> None:
        custom_dir = tmp_path / 'custom_bboxes'
        custom_dir.mkdir()
        self._write_bbox_csv(custom_dir / 'sess_cam0.csv')
        cfg = _minimal_cfg(
            tmp_path,
            has_bboxes=True,
            bbox_csv_pattern='{session_id}_{cam_id}.csv',
        )
        result = load_bbox_csv('sess', 'cam0', cfg, bbox_dir=custom_dir)
        assert result is not None
        assert len(result) == 2
