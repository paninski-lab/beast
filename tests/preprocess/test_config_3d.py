"""Tests for beast/preprocess/config_3d.py."""

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from beast.preprocess.config_3d import (
    AssembleConfig,
    Beast3DConfig,
    CalibrationConfig,
    DownsampleConfig,
    FrameConfig,
    ResizeConfig,
    SegmentationConfig,
    TrimConfig,
    VideoConfig,
    load_config_3d,
    validate_config,
)


class TestVideoConfig:
    """Test the VideoConfig model."""

    def test_defaults(self) -> None:
        cfg = VideoConfig()
        assert cfg.filename_pattern == r'^(?P<session>.+)_(?P<cam>[^_]+)\.mp4$'
        assert cfg.extensions == ['mp4', 'avi']

    def test_extensions_override(self) -> None:
        cfg = VideoConfig(extensions=['mp4'])
        assert cfg.extensions == ['mp4']

    def test_mutable_default_isolation(self) -> None:
        a = VideoConfig()
        b = VideoConfig()
        a.extensions.append('mov')
        assert b.extensions == ['mp4', 'avi']


class TestFrameConfig:
    """Test the FrameConfig model."""

    def test_defaults(self) -> None:
        cfg = FrameConfig()
        assert cfg.frames_per_video == 1000
        assert cfg.n_digits == 8
        assert cfg.extension == 'png'
        assert cfg.kmeans_resize == 32

    def test_override(self) -> None:
        cfg = FrameConfig(frames_per_video=500, extension='jpg')
        assert cfg.frames_per_video == 500
        assert cfg.extension == 'jpg'


class TestSegmentationConfig:
    """Test the SegmentationConfig model."""

    def test_defaults(self) -> None:
        cfg = SegmentationConfig()
        assert cfg.enabled is True
        assert cfg.text_prompt == 'animal'
        assert cfg.num_objects is None
        assert cfg.threshold == 0.5
        assert cfg.clip_size == 512

    def test_num_objects_nullable(self) -> None:
        cfg = SegmentationConfig(num_objects=3)
        assert cfg.num_objects == 3


class TestTrimConfig:
    """Test the TrimConfig model."""

    def test_defaults(self) -> None:
        cfg = TrimConfig()
        assert cfg.enabled is False
        assert cfg.start_frame is None
        assert cfg.end_frame is None
        assert cfg.start_sec is None
        assert cfg.end_sec is None
        assert cfg.ffmpeg_threads is None

    def test_frame_bounds(self) -> None:
        cfg = TrimConfig(enabled=True, start_frame=100, end_frame=500)
        assert cfg.start_frame == 100
        assert cfg.end_frame == 500

    def test_second_bounds(self) -> None:
        cfg = TrimConfig(enabled=True, start_sec=1.5, end_sec=10.0)
        assert cfg.start_sec == 1.5
        assert cfg.end_sec == 10.0


class TestDownsampleConfig:
    """Test the DownsampleConfig model."""

    def test_defaults(self) -> None:
        cfg = DownsampleConfig()
        assert cfg.enabled is False
        assert cfg.target_fps is None
        assert cfg.phase_offset_frames == 0

    def test_target_fps_set(self) -> None:
        cfg = DownsampleConfig(enabled=True, target_fps=5)
        assert cfg.target_fps == 5


class TestAssembleConfig:
    """Test the AssembleConfig model."""

    def test_instantiates(self) -> None:
        assert AssembleConfig() is not None


class TestResizeConfig:
    """Test the ResizeConfig model."""

    def test_defaults(self) -> None:
        cfg = ResizeConfig()
        assert cfg.enabled is False
        assert cfg.size == 256

    def test_enabled_with_size(self) -> None:
        cfg = ResizeConfig(enabled=True, size=128)
        assert cfg.enabled is True
        assert cfg.size == 128


class TestCalibrationConfig:
    """Test the CalibrationConfig model."""

    def test_defaults(self) -> None:
        cfg = CalibrationConfig()
        assert cfg.format == 'anipose_toml'
        assert cfg.cam_name_mapping == 'identity'
        assert cfg.file_pattern == '{session_id}.toml'

    def test_prefix_mapping(self) -> None:
        cfg = CalibrationConfig(cam_name_mapping='prefix_Cam-')
        assert cfg.cam_name_mapping == 'prefix_Cam-'


class TestBeast3DConfig:
    """Test the Beast3DConfig top-level model."""

    def test_defaults(self) -> None:
        cfg = Beast3DConfig()
        assert cfg.name == ''
        assert cfg.input_dir == ''
        assert cfg.output_dir == ''
        assert cfg.anchor_view == ''
        assert cfg.has_bboxes is False
        assert cfg.bbox_csv_pattern == ''
        assert cfg.video_subdir == 'videos'
        assert cfg.max_workers == 4
        assert cfg.author == 'anonymous'
        assert cfg.seed == 42

    def test_sub_configs_are_defaulted(self) -> None:
        cfg = Beast3DConfig()
        assert isinstance(cfg.video, VideoConfig)
        assert isinstance(cfg.frame, FrameConfig)
        assert isinstance(cfg.segmentation, SegmentationConfig)
        assert isinstance(cfg.trim, TrimConfig)
        assert isinstance(cfg.downsample, DownsampleConfig)
        assert isinstance(cfg.calibration, CalibrationConfig)
        assert isinstance(cfg.assemble, AssembleConfig)
        assert isinstance(cfg.resize, ResizeConfig)

    def test_model_validate_from_dict(self) -> None:
        raw = {
            'name': 'chickadee',
            'input_dir': '/data/in',
            'output_dir': '/data/out',
            'anchor_view': 'cam0',
            'downsample': {'enabled': True, 'target_fps': 5},
            'resize': {'enabled': True, 'size': 128},
        }
        cfg = Beast3DConfig.model_validate(raw)
        assert cfg.name == 'chickadee'
        assert cfg.downsample.enabled is True
        assert cfg.downsample.target_fps == 5
        assert cfg.resize.size == 128

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            Beast3DConfig.model_validate({'nonexistent_field': 'value'})

    def test_mutable_default_isolation(self) -> None:
        a = Beast3DConfig()
        b = Beast3DConfig()
        a.video.extensions.append('mov')
        assert b.video.extensions == ['mp4', 'avi']


class TestLoadConfig3d:
    """Test the load_config_3d function."""

    def test_minimal_yaml(self, tmp_path) -> None:
        yaml_text = textwrap.dedent("""\
            name: test
            input_dir: /some/path
            output_dir: /out
            anchor_view: cam0
        """)
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml_text)
        cfg = load_config_3d(config_file)
        assert cfg.name == 'test'
        assert cfg.anchor_view == 'cam0'

    def test_nested_sections_parsed(self, tmp_path) -> None:
        yaml_text = textwrap.dedent("""\
            name: test
            input_dir: /some/path
            output_dir: /out
            anchor_view: cam0
            frame:
              frames_per_video: 200
              extension: jpg
            downsample:
              enabled: true
              target_fps: 10
        """)
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml_text)
        cfg = load_config_3d(config_file)
        assert cfg.frame.frames_per_video == 200
        assert cfg.frame.extension == 'jpg'
        assert cfg.downsample.enabled is True
        assert cfg.downsample.target_fps == 10

    def test_defaults_applied_for_missing_sections(self, tmp_path) -> None:
        yaml_text = textwrap.dedent("""\
            name: test
            input_dir: /some/path
            output_dir: /out
            anchor_view: cam0
        """)
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml_text)
        cfg = load_config_3d(config_file)
        assert cfg.frame.frames_per_video == 1000
        assert cfg.segmentation.enabled is True
        assert cfg.resize.enabled is False

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config_3d(tmp_path / 'nonexistent.yaml')

    def test_returns_beast3d_config_instance(self, tmp_path) -> None:
        config_file = tmp_path / 'config.yaml'
        config_file.write_text('name: x\ninput_dir: /i\noutput_dir: /o\nanchor_view: c\n')
        cfg = load_config_3d(config_file)
        assert isinstance(cfg, Beast3DConfig)

    def test_accepts_path_object(self, tmp_path) -> None:
        config_file = tmp_path / 'config.yaml'
        config_file.write_text('name: x\ninput_dir: /i\noutput_dir: /o\nanchor_view: c\n')
        cfg = load_config_3d(Path(config_file))
        assert cfg.name == 'x'

    def test_accepts_string_path(self, tmp_path) -> None:
        config_file = tmp_path / 'config.yaml'
        config_file.write_text('name: x\ninput_dir: /i\noutput_dir: /o\nanchor_view: c\n')
        cfg = load_config_3d(str(config_file))
        assert cfg.name == 'x'


class TestValidateConfig:
    """Test the validate_config function."""

    def test_valid_config_passes(self, valid_cfg) -> None:
        validate_config(valid_cfg)  # should not raise

    def test_missing_name_raises(self, valid_cfg) -> None:
        valid_cfg.name = ''
        with pytest.raises(ValueError, match='name'):
            validate_config(valid_cfg)

    def test_missing_input_dir_raises(self, valid_cfg) -> None:
        valid_cfg.input_dir = ''
        with pytest.raises(ValueError, match='input_dir'):
            validate_config(valid_cfg)

    def test_missing_output_dir_raises(self, valid_cfg) -> None:
        valid_cfg.output_dir = ''
        with pytest.raises(ValueError, match='output_dir'):
            validate_config(valid_cfg)

    def test_missing_anchor_view_raises(self, valid_cfg) -> None:
        valid_cfg.anchor_view = ''
        with pytest.raises(ValueError, match='anchor_view'):
            validate_config(valid_cfg)

    def test_missing_videos_dir_raises(self, valid_cfg, config_dirs) -> None:
        (config_dirs / 'videos').rmdir()
        with pytest.raises(ValueError, match='Videos directory not found'):
            validate_config(valid_cfg)

    def test_missing_calibrations_dir_raises(self, valid_cfg, config_dirs) -> None:
        (config_dirs / 'calibrations').rmdir()
        with pytest.raises(ValueError, match='Calibrations directory not found'):
            validate_config(valid_cfg)

    def test_non_anipose_format_skips_calibrations_check(self, valid_cfg, config_dirs) -> None:
        (config_dirs / 'calibrations').rmdir()
        valid_cfg.calibration.format = 'json'
        validate_config(valid_cfg)  # should not raise

    def test_empty_file_pattern_raises(self, valid_cfg) -> None:
        valid_cfg.calibration.file_pattern = ''
        with pytest.raises(ValueError, match='file_pattern'):
            validate_config(valid_cfg)
