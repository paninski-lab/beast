"""Pydantic config models for BEAST3D dataset creation."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class VideoConfig(BaseModel):
    """Video file naming and discovery settings."""

    model_config = ConfigDict(extra='forbid')

    filename_pattern: str = r'^(?P<session>.+)_(?P<cam>[^_]+)\.mp4$'
    extensions: list[str] = ['mp4', 'avi']


class FrameConfig(BaseModel):
    """Frame extraction settings."""

    model_config = ConfigDict(extra='forbid')

    frames_per_video: int = 1000
    n_digits: int = 8
    extension: str = 'png'
    kmeans_resize: int = 32


class SegmentationConfig(BaseModel):
    """SAM3 segmentation settings."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    text_prompt: str = 'animal'
    num_objects: int | None = None
    threshold: float = 0.5
    clip_size: int = 512


class TrimConfig(BaseModel):
    """Video trim settings.

    Trims every input video to a frame range. Either frame-based
    (start_frame/end_frame) or second-based (start_sec/end_sec) bounds may be
    provided; seconds are converted to frames using the source video's FPS.

    When enabled with has_bboxes=True, matching bbox CSV files are also trimmed
    and re-indexed so row indices start at 0.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    start_frame: int | None = None
    end_frame: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    max_workers: int = 4
    ffmpeg_threads: int | None = None


class DownsampleConfig(BaseModel):
    """Video downsampling settings."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    target_fps: int | None = None
    max_frames: int | None = None
    max_workers: int = 4
    ffmpeg_threads: int | None = None
    phase_offset_frames: int = 0


class AssembleConfig(BaseModel):
    """Dataset assembly settings."""

    model_config = ConfigDict(extra='forbid')

    downsample_selected_frames: bool = False
    downsample_factor: int = 1
    max_workers: int = 4


class ResizeConfig(BaseModel):
    """Post-assembly image resize settings."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    size: int = 256


class CalibrationConfig(BaseModel):
    """Camera calibration file settings.

    Supported formats: 'anipose_toml'.

    cam_name_mapping controls how cam_id from video filenames maps to camera
    names in the calibration file. Options:
      - 'identity': cam_id == cam_name
      - 'prefix_<str>': cam_name = '<str>{cam_id}' (e.g. 'prefix_Cam-')
      - any string containing '{cam_id}': used as a format string
    """

    model_config = ConfigDict(extra='forbid')

    format: str = 'anipose_toml'
    cam_name_mapping: str = 'identity'
    file_pattern: str = '{session_id}.toml'


class Beast3DConfig(BaseModel):
    """Top-level config for BEAST3D dataset creation."""

    model_config = ConfigDict(extra='forbid')

    name: str = ''
    input_dir: str = ''
    output_dir: str = ''
    anchor_view: str = ''
    has_bboxes: bool = False
    bbox_csv_pattern: str = ''
    video_subdir: str = 'videos'
    author: str = 'anonymous'
    seed: int = 42

    video: VideoConfig = VideoConfig()
    frame: FrameConfig = FrameConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    trim: TrimConfig = TrimConfig()
    downsample: DownsampleConfig = DownsampleConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    assemble: AssembleConfig = AssembleConfig()
    resize: ResizeConfig = ResizeConfig()


def load_config_3d(path: str | Path) -> Beast3DConfig:
    """Load a Beast3DConfig from a YAML file.

    Parameters
    ----------
    path: path to the YAML config file

    Returns
    -------
    validated Beast3DConfig object

    Raises
    ------
    FileNotFoundError: if the config file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Beast3DConfig.model_validate(raw)


def validate_config(cfg: Beast3DConfig) -> None:
    """Check filesystem preconditions for a Beast3DConfig.

    Pydantic handles type validation; this function checks that required
    directories and files exist on disk.

    Parameters
    ----------
    cfg: config to validate

    Raises
    ------
    ValueError: if required fields are empty or required paths do not exist
    """
    for field in ('name', 'input_dir', 'output_dir', 'anchor_view'):
        if not getattr(cfg, field):
            raise ValueError(f'Config must specify {field}')

    videos_path = Path(cfg.input_dir) / cfg.video_subdir
    if not videos_path.is_dir():
        raise ValueError(f'Videos directory not found: {videos_path}')

    if cfg.calibration.format == 'anipose_toml':
        cal_path = Path(cfg.input_dir) / 'calibrations'
        if not cal_path.is_dir():
            raise ValueError(f'Calibrations directory not found: {cal_path}')

    if not cfg.calibration.file_pattern:
        raise ValueError('calibration.file_pattern must not be empty')
