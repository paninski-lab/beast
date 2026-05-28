"""Tests for the extract_3d CLI command."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beast.cli.commands.extract_3d import handle
from beast.cli.main import build_parser

_CFG3D = 'beast.preprocess.config_3d'
_EXT3D = 'beast.preprocess.extraction_3d'
_SEG = 'beast.preprocess.segment.pipeline'


class TestRegisterParser:
    """Test the extract_3d command parser registration."""

    def test_defaults(self, tmp_path: Path) -> None:
        config = tmp_path / 'config.yaml'
        config.write_text('name: test\n')
        parser = build_parser()
        args = parser.parse_args(['extract_3d', '--config', str(config)])
        assert args.skip_stats is False

    def test_skip_stats_flag(self, tmp_path: Path) -> None:
        config = tmp_path / 'config.yaml'
        config.write_text('name: test\n')
        parser = build_parser()
        args = parser.parse_args(['extract_3d', '--config', str(config), '--skip-stats'])
        assert args.skip_stats is True

    def test_missing_config_exits(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['extract_3d'])


class TestHandle:
    """Test the extract_3d command handle function."""

    _ALL_PATCHES = [
        (f'{_CFG3D}.load_config_3d', 'load'),
        (f'{_CFG3D}.validate_config', 'validate'),
        (f'{_EXT3D}.run_video_stats', 'stats'),
        (f'{_EXT3D}.run_trim', 'trim'),
        (f'{_EXT3D}.run_downsample', 'downsample'),
        (f'{_EXT3D}.resolve_videos_dir', 'resolve'),
        (f'{_EXT3D}.assemble_dataset', 'assemble'),
        (f'{_EXT3D}.resize_dataset', 'resize'),
        (f'{_SEG}.run_segmentation', 'segment'),
    ]

    def _make_cfg(self) -> MagicMock:
        cfg = MagicMock()
        cfg.cut.enabled = False
        cfg.downsample.enabled = False
        cfg.segmentation.enabled = False
        cfg.resize.enabled = False
        return cfg

    def _run(self, cfg: MagicMock, skip_stats: bool = False) -> dict[str, MagicMock]:
        args = argparse.Namespace(config=Path('/cfg.yaml'), skip_stats=skip_stats)
        mocks: dict[str, MagicMock] = {}
        with ExitStack() as stack:
            for path, name in self._ALL_PATCHES:
                mocks[name] = stack.enter_context(patch(path))
            mocks['load'].return_value = cfg
            handle(args)
        return mocks

    def test_calls_load_and_validate(self) -> None:
        cfg = self._make_cfg()
        mocks = self._run(cfg)
        mocks['load'].assert_called_once_with(Path('/cfg.yaml'))
        mocks['validate'].assert_called_once_with(cfg)

    def test_validate_error_propagates(self) -> None:
        cfg = self._make_cfg()
        args = argparse.Namespace(config=Path('/cfg.yaml'), skip_stats=False)
        with patch(f'{_CFG3D}.load_config_3d', return_value=cfg):
            with patch(f'{_CFG3D}.validate_config', side_effect=ValueError('bad config')):
                with pytest.raises(ValueError, match='bad config'):
                    handle(args)

    def test_runs_stats_by_default(self) -> None:
        cfg = self._make_cfg()
        mocks = self._run(cfg)
        mocks['stats'].assert_called_once_with(cfg)

    def test_skip_stats_skips_stats(self) -> None:
        cfg = self._make_cfg()
        mocks = self._run(cfg, skip_stats=True)
        mocks['stats'].assert_not_called()

    def test_skips_trim_when_disabled(self) -> None:
        cfg = self._make_cfg()
        mocks = self._run(cfg)
        mocks['trim'].assert_not_called()

    def test_runs_trim_when_enabled(self) -> None:
        cfg = self._make_cfg()
        cfg.cut.enabled = True
        mocks = self._run(cfg)
        mocks['trim'].assert_called_once_with(cfg)

    def test_skips_downsample_when_disabled(self) -> None:
        cfg = self._make_cfg()
        mocks = self._run(cfg)
        mocks['downsample'].assert_not_called()

    def test_runs_downsample_when_enabled(self) -> None:
        cfg = self._make_cfg()
        cfg.downsample.enabled = True
        mocks = self._run(cfg)
        mocks['downsample'].assert_called_once_with(cfg)

    def test_skips_segment_when_disabled(self) -> None:
        cfg = self._make_cfg()
        mocks = self._run(cfg)
        mocks['segment'].assert_not_called()
        mocks['resolve'].assert_not_called()

    def test_runs_segment_when_enabled(self) -> None:
        cfg = self._make_cfg()
        cfg.segmentation.enabled = True
        mocks = self._run(cfg)
        mocks['resolve'].assert_called_once_with(cfg)
        mocks['segment'].assert_called_once_with(mocks['resolve'].return_value, cfg)

    def test_always_runs_assemble(self) -> None:
        cfg = self._make_cfg()
        mocks = self._run(cfg)
        mocks['assemble'].assert_called_once_with(cfg)

    def test_skips_resize_when_disabled(self) -> None:
        cfg = self._make_cfg()
        mocks = self._run(cfg)
        mocks['resize'].assert_not_called()

    def test_runs_resize_when_enabled(self) -> None:
        cfg = self._make_cfg()
        cfg.resize.enabled = True
        mocks = self._run(cfg)
        mocks['resize'].assert_called_once_with(cfg)
