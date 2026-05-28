"""Tests for beast/preprocess/segment/pipeline.py."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beast.preprocess.config_3d import Beast3DConfig
from beast.preprocess.segment.pipeline import (
    _get_physical_gpu_ids,
    _segment_worker,
    run_segmentation,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_PIPELINE = 'beast.preprocess.segment.pipeline'
_SAM3_MODULE = 'beast.preprocess.segment.sam3'
_CUDA_AVAIL = f'{_PIPELINE}.torch.cuda.is_available'
_CUDA_COUNT = f'{_PIPELINE}.torch.cuda.device_count'


def _minimal_cfg(tmp_path: Path, **kwargs) -> Beast3DConfig:
    return Beast3DConfig(
        name='test',
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / 'out'),
        anchor_view='cam0',
        **kwargs,
    )


def _make_videos(videos_dir: Path, names: list[str]) -> list[Path]:
    videos_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = videos_dir / name
        p.touch()
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# TestGetPhysicalGpuIds
# ---------------------------------------------------------------------------

class TestGetPhysicalGpuIds:
    """Test the _get_physical_gpu_ids function."""

    def test_cuda_visible_devices_returns_those_ids(self) -> None:
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,5'}):
            result = _get_physical_gpu_ids()
        assert result == ['2', '5']

    def test_cuda_visible_devices_single_gpu(self) -> None:
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}):
            result = _get_physical_gpu_ids()
        assert result == ['3']

    def test_cuda_available_without_env_var(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'}
        with patch.dict(os.environ, env, clear=True):
            with patch(_CUDA_AVAIL, return_value=True):
                with patch(_CUDA_COUNT, return_value=2):
                    result = _get_physical_gpu_ids()
        assert result == ['0', '1']

    def test_no_cuda_returns_empty_list(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'}
        with patch.dict(os.environ, env, clear=True):
            with patch(_CUDA_AVAIL, return_value=False):
                result = _get_physical_gpu_ids()
        assert result == []

    def test_cuda_visible_devices_empty_string_falls_back_to_torch(self) -> None:
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with patch(_CUDA_AVAIL, return_value=True):
                with patch(_CUDA_COUNT, return_value=1):
                    result = _get_physical_gpu_ids()
        assert result == ['0']


# ---------------------------------------------------------------------------
# TestSegmentWorker
# ---------------------------------------------------------------------------

class TestSegmentWorker:
    """Test the _segment_worker function."""

    def _mock_process_video(self) -> MagicMock:
        return MagicMock(return_value=None)

    def test_successful_video_creates_complete_marker(self, tmp_path) -> None:
        output_dir = tmp_path / 'out' / 'vid'
        mock_pv = self._mock_process_video()
        with patch.dict('sys.modules', {_SAM3_MODULE: MagicMock(process_video=mock_pv)}):
            failures = _segment_worker(
                physical_gpu_id='0',
                video_list=[str(tmp_path / 'vid.mp4')],
                output_dirs=[str(output_dir)],
                text_prompt='animal',
                num_objects=None,
                threshold=0.5,
                clip_size=512,
            )
        assert failures == []
        assert (output_dir / '_COMPLETE').exists()

    def test_failed_video_appended_to_failures(self, tmp_path) -> None:
        mock_pv = MagicMock(side_effect=RuntimeError('SAM3 crash'))
        with patch.dict('sys.modules', {_SAM3_MODULE: MagicMock(process_video=mock_pv)}):
            failures = _segment_worker(
                physical_gpu_id='0',
                video_list=[str(tmp_path / 'bad.mp4')],
                output_dirs=[str(tmp_path / 'bad_out')],
                text_prompt='animal',
                num_objects=None,
                threshold=0.5,
                clip_size=512,
            )
        assert len(failures) == 1
        assert failures[0]['error_type'] == 'RuntimeError'
        assert failures[0]['video'] == 'bad.mp4'

    def test_incomplete_output_dir_is_removed_before_retry(self, tmp_path) -> None:
        output_dir = tmp_path / 'incomplete_out'
        output_dir.mkdir()
        stale_file = output_dir / 'stale.png'
        stale_file.touch()
        mock_pv = self._mock_process_video()
        with patch.dict('sys.modules', {_SAM3_MODULE: MagicMock(process_video=mock_pv)}):
            _segment_worker(
                physical_gpu_id='0',
                video_list=[str(tmp_path / 'vid.mp4')],
                output_dirs=[str(output_dir)],
                text_prompt='animal',
                num_objects=None,
                threshold=0.5,
                clip_size=512,
            )
        assert not stale_file.exists()

    def test_failure_appended_to_shared_list(self, tmp_path) -> None:
        mock_pv = MagicMock(side_effect=ValueError('bad input'))
        shared = []
        with patch.dict('sys.modules', {_SAM3_MODULE: MagicMock(process_video=mock_pv)}):
            _segment_worker(
                physical_gpu_id='1',
                video_list=[str(tmp_path / 'vid.mp4')],
                output_dirs=[str(tmp_path / 'out')],
                text_prompt='animal',
                num_objects=None,
                threshold=0.5,
                clip_size=512,
                failed_list=shared,
            )
        assert len(shared) == 1
        assert shared[0]['gpu_id'] == '1'

    def test_missing_sam3_raises_import_error(self, tmp_path) -> None:
        import sys
        modules = {k: v for k, v in sys.modules.items()}
        modules.pop(_SAM3_MODULE, None)
        with patch.dict('sys.modules', {_SAM3_MODULE: None}):  # type: ignore[dict-item]
            with pytest.raises((ImportError, AttributeError)):
                _segment_worker(
                    physical_gpu_id='0',
                    video_list=[str(tmp_path / 'vid.mp4')],
                    output_dirs=[str(tmp_path / 'out')],
                    text_prompt='animal',
                    num_objects=None,
                    threshold=0.5,
                    clip_size=512,
                )


# ---------------------------------------------------------------------------
# TestRunSegmentation
# ---------------------------------------------------------------------------

class TestRunSegmentation:
    """Test the run_segmentation function."""

    def _make_cfg_with_videos(self, tmp_path: Path, n_videos: int = 2) -> Beast3DConfig:
        videos_dir = tmp_path / 'videos'
        _make_videos(videos_dir, [f'sess_{i}_cam0.mp4' for i in range(n_videos)])
        cfg = _minimal_cfg(tmp_path)
        cfg.video_subdir = 'videos'
        return cfg

    def test_no_videos_returns_early(self, tmp_path) -> None:
        cfg = _minimal_cfg(tmp_path)
        (tmp_path / 'videos').mkdir()
        with patch(f'{_PIPELINE}._segment_worker') as mock_worker:
            run_segmentation(tmp_path / 'videos', cfg)
        mock_worker.assert_not_called()

    def test_all_complete_returns_early(self, tmp_path) -> None:
        cfg = self._make_cfg_with_videos(tmp_path)
        videos_dir = tmp_path / 'videos'
        seg_dir = tmp_path / 'out' / 'segmentation_masks'
        for vf in videos_dir.glob('*.mp4'):
            marker_dir = seg_dir / vf.stem
            marker_dir.mkdir(parents=True)
            (marker_dir / '_COMPLETE').touch()
        with patch(f'{_PIPELINE}._segment_worker') as mock_worker:
            run_segmentation(videos_dir, cfg)
        mock_worker.assert_not_called()

    def test_single_gpu_calls_worker_directly(self, tmp_path) -> None:
        cfg = self._make_cfg_with_videos(tmp_path)
        videos_dir = tmp_path / 'videos'
        with patch(f'{_PIPELINE}._get_physical_gpu_ids', return_value=['0']):
            with patch(f'{_PIPELINE}._segment_worker', return_value=[]) as mock_worker:
                run_segmentation(videos_dir, cfg)
        mock_worker.assert_called_once()
        assert mock_worker.call_args.kwargs['physical_gpu_id'] == '0'

    def test_no_gpu_falls_back_to_id_zero(self, tmp_path) -> None:
        cfg = self._make_cfg_with_videos(tmp_path, n_videos=1)
        videos_dir = tmp_path / 'videos'
        with patch(f'{_PIPELINE}._get_physical_gpu_ids', return_value=[]):
            with patch(f'{_PIPELINE}._segment_worker', return_value=[]) as mock_worker:
                run_segmentation(videos_dir, cfg)
        assert mock_worker.call_args.kwargs['physical_gpu_id'] == '0'

    def test_failures_written_to_json(self, tmp_path) -> None:
        cfg = self._make_cfg_with_videos(tmp_path, n_videos=1)
        videos_dir = tmp_path / 'videos'
        failure = {
            'video': 'sess_0_cam0.mp4',
            'video_path': str(videos_dir / 'sess_0_cam0.mp4'),
            'gpu_id': '0',
            'error': 'boom',
            'error_type': 'RuntimeError',
            'traceback': '...',
        }
        with patch(f'{_PIPELINE}._get_physical_gpu_ids', return_value=['0']):
            with patch(f'{_PIPELINE}._segment_worker', return_value=[failure]):
                run_segmentation(videos_dir, cfg)
        failures_file = tmp_path / 'out' / 'segmentation_masks' / 'failed_videos.json'
        assert failures_file.exists()
        data = json.loads(failures_file.read_text())
        assert len(data) == 1
        assert data[0]['video'] == 'sess_0_cam0.mp4'
        assert 'traceback' not in data[0]

    def test_old_failures_file_removed_on_clean_run(self, tmp_path) -> None:
        cfg = self._make_cfg_with_videos(tmp_path, n_videos=1)
        videos_dir = tmp_path / 'videos'
        seg_dir = tmp_path / 'out' / 'segmentation_masks'
        seg_dir.mkdir(parents=True)
        stale = seg_dir / 'failed_videos.json'
        stale.write_text('[]')
        with patch(f'{_PIPELINE}._get_physical_gpu_ids', return_value=['0']):
            with patch(f'{_PIPELINE}._segment_worker', return_value=[]):
                run_segmentation(videos_dir, cfg)
        assert not stale.exists()

    def test_pending_videos_passed_to_worker(self, tmp_path) -> None:
        cfg = self._make_cfg_with_videos(tmp_path, n_videos=3)
        videos_dir = tmp_path / 'videos'
        seg_dir = tmp_path / 'out' / 'segmentation_masks'
        done_dir = seg_dir / 'sess_0_cam0'
        done_dir.mkdir(parents=True)
        (done_dir / '_COMPLETE').touch()
        with patch(f'{_PIPELINE}._get_physical_gpu_ids', return_value=['0']):
            with patch(f'{_PIPELINE}._segment_worker', return_value=[]) as mock_worker:
                run_segmentation(videos_dir, cfg)
        video_list = mock_worker.call_args.kwargs['video_list']
        assert len(video_list) == 2
        assert all('sess_0_cam0' not in v for v in video_list)
