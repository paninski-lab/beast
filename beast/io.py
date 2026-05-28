"""Configuration loading, config overrides, and camera calibration for BEAST."""

import csv
import logging
from pathlib import Path

import yaml
from aniposelib.cameras import CameraGroup as CameraGroupAnipose

from beast.config import BeastConfig
from beast.data.config_3d import Beast3DConfig

_logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    """Load yaml configuration file to a nested dictionary structure.

    Parameters
    ----------
    path: absolute path to config yaml file

    Returns
    -------
    nested configuration dictionary

    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path} does not exist')

    # load raw yaml into an untyped dict
    with open(path) as file:
        raw = yaml.safe_load(file)

    # validate against the schema; raises ValidationError on missing required
    # fields, wrong types, or invalid Literal values
    validated = BeastConfig.model_validate(raw)

    # convert back to a plain nested dict so callers don't depend on pydantic types;
    # this also fills in any fields that have defaults but were absent from the yaml
    return validated.model_dump()


def apply_config_overrides(config: dict, overrides: dict | list) -> dict:
    """Apply configuration overrides to a nested dictionary structure.

    Parameters
    ----------
    config: base configuration dictionary to modify
    overrides:
        dictionary with dot-notation keys and values to override
        list with KEY=VALUE entries, keys use dot-notation

    Returns
    -------
    The modified configuration dictionary

    """

    if isinstance(overrides, list):
        overrides = {item.split('=')[0]: item.split('=')[1] for item in overrides}

    for field, value in overrides.items():
        keys = field.split('.')
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    return config


def map_cam_name(cam_id: str, mapping: str) -> str:
    """Map a cam_id from a video filename to the camera name in a calibration file.

    Parameters
    ----------
    cam_id: camera identifier from the video filename
    mapping: mapping rule; 'identity', 'prefix_<str>', or a format string containing '{cam_id}'

    Returns
    -------
    camera name string

    """
    if mapping == 'identity':
        return cam_id
    if mapping.startswith('prefix_'):
        prefix = mapping[len('prefix_'):]
        return f'{prefix}{cam_id}'
    if '{cam_id}' in mapping:
        return mapping.format(cam_id=cam_id)
    return cam_id


def get_calibration_path(session_id: str, cfg: Beast3DConfig) -> Path:
    """Return the calibration file path for a given session.

    Parameters
    ----------
    session_id: session identifier
    cfg: beast3d config

    Returns
    -------
    path to the calibration file

    """
    cal_dir = Path(cfg.input_dir) / 'calibrations'
    filename = cfg.calibration.file_pattern.format(session_id=session_id)
    return cal_dir / filename


def load_anipose_calibration(calibration_file: str | Path) -> dict[str, dict]:
    """Load camera parameters from an anipose-style TOML calibration file.

    Parameters
    ----------
    calibration_file: path to the anipose TOML calibration file

    Returns
    -------
    dict mapping camera_name to a parameter dict with keys: intrinsics (3x3 array),
    extrinsics (4x4 array), distortions array, width, height

    """
    camera_group = CameraGroupAnipose.load(str(calibration_file))
    cam_params = {}
    for cam in camera_group.cameras:
        cam_params[cam.name] = {
            'intrinsics': cam.get_camera_matrix(),
            'extrinsics': cam.get_extrinsics_mat(),
            'distortions': cam.get_distortions(),
            'width': cam.get_size()[0],
            'height': cam.get_size()[1],
        }
    return cam_params


def load_calibration(session_id: str, cfg: Beast3DConfig) -> dict[str, dict] | None:
    """Load camera calibration for a session.

    Parameters
    ----------
    session_id: session identifier
    cfg: beast3d config

    Returns
    -------
    dict mapping camera_name to parameter dict, or None if calibration file not found

    Raises
    ------
    ValueError: if the calibration format is not supported

    """
    cal_path = get_calibration_path(session_id, cfg)
    if not cal_path.exists():
        _logger.warning(f'calibration file not found: {cal_path}')
        return None
    if cfg.calibration.format == 'anipose_toml':
        return load_anipose_calibration(cal_path)
    raise ValueError(f'unsupported calibration format: {cfg.calibration.format}')


def get_camera_params_for_view(
    cam_params: dict[str, dict],
    cam_id: str,
    cfg: Beast3DConfig,
) -> dict | None:
    """Get camera parameters for a specific view.

    Parameters
    ----------
    cam_params: dict mapping camera name to parameter dict
    cam_id: camera identifier from the video filename
    cfg: beast3d config

    Returns
    -------
    parameter dict for the camera, or None if the camera is not found

    """
    cam_name = map_cam_name(cam_id, cfg.calibration.cam_name_mapping)
    if cam_name in cam_params:
        return cam_params[cam_name]
    _logger.warning(f'camera {cam_name} (cam_id={cam_id}) not found in calibration')
    return None


def load_bbox_csv(
    session_id: str,
    cam_id: str,
    cfg: Beast3DConfig,
    bbox_dir: Path | None = None,
) -> dict | None:
    """Load bounding box CSV for a session/camera if configured.

    Parameters
    ----------
    session_id: session identifier
    cam_id: camera identifier
    cfg: beast3d config
    bbox_dir: directory containing bbox CSVs; defaults to cfg.input_dir / cfg.video_subdir

    Returns
    -------
    dict mapping integer frame index to bbox dict with keys x, y, w, h, or None if
    not configured or file not found

    """
    if not cfg.has_bboxes or not cfg.bbox_csv_pattern:
        return None

    if bbox_dir is None:
        bbox_dir = Path(cfg.input_dir) / cfg.video_subdir

    csv_path = bbox_dir / cfg.bbox_csv_pattern.format(
        session_id=session_id, cam_id=cam_id,
    )
    if not csv_path.exists():
        _logger.warning(f'bbox csv not found: {csv_path}')
        return None

    bbox_dict = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for enum_idx, row in enumerate(reader):
            key = int(row['Unnamed: 0']) if 'Unnamed: 0' in row else enum_idx
            bbox_dict[key] = {
                'x': int(row['x']),
                'y': int(row['y']),
                'w': int(row['w']),
                'h': int(row['h']),
            }
    return bbox_dict
