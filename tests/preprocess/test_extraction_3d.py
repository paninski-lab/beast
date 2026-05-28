"""Tests for beast/preprocess/extraction_3d.py."""

import csv
import json
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from beast.preprocess.config_3d import Beast3DConfig, CutConfig, DownsampleConfig
from beast.preprocess.extraction_3d import (
    _compute_new_size,
    _copy_masks_for_frames,
    _get_trim_videos_dir,
    _process_view_task,
    _resolve_frame_range,
    _save_camera_params_for_frames,
    _scale_intrinsics,
    _trim_bbox_csv,
    _trim_one_video,
    assemble_dataset,
    resize_dataset,
    resolve_videos_dir,
    run_downsample,
    run_trim,
    run_video_stats,
)

_EXT3D = 'beast.preprocess.extraction_3d'


def _minimal_cfg(tmp_path: Path, **kwargs) -> Beast3DConfig:
    return Beast3DConfig(
        name='test',
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / 'out'),
        anchor_view='cam0',
        **kwargs,
    )


def _make_video_files(videos_dir: Path, names: list[str]) -> None:
    videos_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (videos_dir / name).touch()


def _fake_video_stats(**overrides) -> dict:
    base = {'fps': 30.0, 'width': 640, 'height': 480, 'total_frames': 300, 'duration_sec': 10.0,
            'codec': 'avc1'}
    base.update(overrides)
    return base


class _SyncExecutor:
    """Synchronous stand-in for ProcessPoolExecutor — runs submitted tasks inline."""

    def __init__(self, max_workers=1):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args, **kwargs):
        f: Future = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as exc:
            f.set_exception(exc)
        return f


# ---------------------------------------------------------------------------
# TestGetTrimVideosDir
# ---------------------------------------------------------------------------

class TestGetTrimVideosDir:
    """Test the _get_trim_videos_dir function."""

    def test_returns_output_dir_with_trim_suffix(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        result = _get_trim_videos_dir(cfg)
        assert result == tmp_path / 'out' / 'videos_trim'


# ---------------------------------------------------------------------------
# TestResolveVideosDir
# ---------------------------------------------------------------------------

class TestResolveVideosDir:
    """Test the resolve_videos_dir function."""

    def test_returns_raw_input_when_nothing_enabled(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        result = resolve_videos_dir(cfg)
        assert result == tmp_path / 'videos'

    def test_returns_trim_dir_when_cut_enabled_and_exists(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        cfg.cut.enabled = True
        trim_dir = tmp_path / 'out' / 'videos_trim'
        trim_dir.mkdir(parents=True)
        result = resolve_videos_dir(cfg)
        assert result == trim_dir

    def test_returns_downsample_dir_when_enabled_and_exists(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        cfg.downsample.enabled = True
        ds_dir = tmp_path / 'out' / 'videos'
        ds_dir.mkdir(parents=True)
        result = resolve_videos_dir(cfg)
        assert result == ds_dir

    def test_downsample_dir_missing_falls_through_to_trim(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        cfg.downsample.enabled = True
        cfg.cut.enabled = True
        # ds dir absent; trim dir present
        trim_dir = tmp_path / 'out' / 'videos_trim'
        trim_dir.mkdir(parents=True)
        result = resolve_videos_dir(cfg)
        assert result == trim_dir


# ---------------------------------------------------------------------------
# TestRunVideoStats
# ---------------------------------------------------------------------------

class TestRunVideoStats:
    """Test the run_video_stats function."""

    def test_no_videos_returns_empty_dict(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        (tmp_path / 'videos').mkdir()
        result = run_video_stats(cfg)
        assert result == {}

    def test_writes_csv_and_json(self, tmp_path: Path) -> None:
        videos_dir = tmp_path / 'videos'
        _make_video_files(videos_dir, ['s0_cam0.mp4', 's1_cam0.mp4'])
        cfg = _minimal_cfg(tmp_path)
        with patch(f'{_EXT3D}.get_video_stats', return_value=_fake_video_stats()):
            run_video_stats(cfg)
        assert (tmp_path / 'out' / 'video_stats.csv').exists()
        assert (tmp_path / 'out' / 'video_stats.json').exists()

    def test_summary_has_expected_keys(self, tmp_path: Path) -> None:
        videos_dir = tmp_path / 'videos'
        _make_video_files(videos_dir, ['s0_cam0.mp4'])
        cfg = _minimal_cfg(tmp_path)
        with patch(f'{_EXT3D}.get_video_stats', return_value=_fake_video_stats()):
            summary = run_video_stats(cfg)
        expected = {'num_videos', 'fps_avg', 'fps_min', 'fps_max', 'total_frames',
                    'total_duration_sec', 'avg_frames_per_video', 'resolutions', 'videos_dir'}
        assert expected.issubset(summary.keys())
        assert summary['num_videos'] == 1
        assert summary['fps_avg'] == 30.0


# ---------------------------------------------------------------------------
# TestResolveFrameRange
# ---------------------------------------------------------------------------

class TestResolveFrameRange:
    """Test the _resolve_frame_range function."""

    def test_uses_start_frame_and_end_frame_when_set(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path, cut=CutConfig(start_frame=10, end_frame=50))
        start, end = _resolve_frame_range(cfg, tmp_path / 'v.mp4')
        assert start == 10
        assert end == 50

    def test_defaults_to_zero_and_total_minus_one(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        with patch(f'{_EXT3D}.get_video_stats', return_value=_fake_video_stats(total_frames=100)):
            start, end = _resolve_frame_range(cfg, tmp_path / 'v.mp4')
        assert start == 0
        assert end == 99

    def test_converts_seconds_to_frames(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path, cut=CutConfig(start_sec=1.0, end_sec=3.0))
        with patch(f'{_EXT3D}.get_video_stats', return_value=_fake_video_stats(fps=10.0)):
            start, end = _resolve_frame_range(cfg, tmp_path / 'v.mp4')
        assert start == 10   # 1.0 * 10
        assert end == 29     # int(round(3.0 * 10)) - 1 = 29

    def test_raises_when_start_greater_than_end(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path, cut=CutConfig(start_frame=100, end_frame=50))
        with pytest.raises(ValueError, match='invalid trim range'):
            _resolve_frame_range(cfg, tmp_path / 'v.mp4')


# ---------------------------------------------------------------------------
# TestTrimBboxCsv
# ---------------------------------------------------------------------------

class TestTrimBboxCsv:
    """Test the _trim_bbox_csv function."""

    def _write_csv(self, path: Path, rows: list[dict], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_filters_to_frame_range_and_reindexes(self, tmp_path: Path) -> None:
        src = tmp_path / 'src.csv'
        self._write_csv(src, [
            {'frame': '0', 'x': '1'},
            {'frame': '5', 'x': '2'},
            {'frame': '10', 'x': '3'},
            {'frame': '15', 'x': '4'},
        ], ['frame', 'x'])
        dst = tmp_path / 'dst.csv'
        n = _trim_bbox_csv(src, dst, start_frame=5, end_frame=10)
        assert n == 2
        with open(dst, newline='') as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['frame'] == '0'   # re-indexed
        assert rows[1]['frame'] == '1'

    def test_empty_input_writes_header_only(self, tmp_path: Path) -> None:
        src = tmp_path / 'src.csv'
        self._write_csv(src, [], ['frame', 'x'])
        dst = tmp_path / 'dst.csv'
        n = _trim_bbox_csv(src, dst, start_frame=0, end_frame=100)
        assert n == 0
        with open(dst, newline='') as f:
            rows = list(csv.DictReader(f))
        assert rows == []

    def test_all_rows_outside_range_returns_zero(self, tmp_path: Path) -> None:
        src = tmp_path / 'src.csv'
        self._write_csv(src, [{'frame': '0', 'x': '1'}], ['frame', 'x'])
        dst = tmp_path / 'dst.csv'
        n = _trim_bbox_csv(src, dst, start_frame=10, end_frame=20)
        assert n == 0


# ---------------------------------------------------------------------------
# TestTrimOneVideo
# ---------------------------------------------------------------------------

class TestTrimOneVideo:
    """Test the _trim_one_video function."""

    def test_skips_when_output_already_exists(self, tmp_path: Path) -> None:
        output = tmp_path / 'out.mp4'
        output.touch()
        with patch(f'{_EXT3D}.trim_video') as mock_trim:
            _, _, _, skipped = _trim_one_video(
                tmp_path / 'in.mp4', output, 0, 99, threads=None,
            )
        assert skipped is True
        mock_trim.assert_not_called()

    def test_calls_trim_video_when_output_missing(self, tmp_path: Path) -> None:
        with patch(f'{_EXT3D}.trim_video') as mock_trim:
            _, _, _, skipped = _trim_one_video(
                tmp_path / 'in.mp4', tmp_path / 'out.mp4', 10, 50, threads=2,
            )
        assert skipped is False
        mock_trim.assert_called_once_with(
            tmp_path / 'in.mp4', tmp_path / 'out.mp4', 10, 50, threads=2,
        )


# ---------------------------------------------------------------------------
# TestRunTrim
# ---------------------------------------------------------------------------

class TestRunTrim:
    """Test the run_trim function."""

    def test_raises_when_no_videos(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        (tmp_path / 'videos').mkdir()
        with pytest.raises(FileNotFoundError):
            run_trim(cfg)

    def test_returns_output_dir_and_trims(self, tmp_path: Path) -> None:
        _make_video_files(tmp_path / 'videos', ['s0_cam0.mp4'])
        cfg = _minimal_cfg(tmp_path)
        with patch(f'{_EXT3D}.ProcessPoolExecutor', _SyncExecutor):
            with patch(f'{_EXT3D}.trim_video'):
                with patch(f'{_EXT3D}.get_video_stats', return_value=_fake_video_stats()):
                    result = run_trim(cfg)
        assert result == tmp_path / 'out' / 'videos_trim'


# ---------------------------------------------------------------------------
# TestRunDownsample
# ---------------------------------------------------------------------------

class TestRunDownsample:
    """Test the run_downsample function."""

    def test_raises_when_no_videos(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path, downsample=DownsampleConfig(enabled=True))
        (tmp_path / 'videos').mkdir()
        with pytest.raises(FileNotFoundError):
            run_downsample(cfg)

    def test_raises_for_negative_phase_offset(self, tmp_path: Path) -> None:
        _make_video_files(tmp_path / 'videos', ['s0_cam0.mp4'])
        cfg = _minimal_cfg(
            tmp_path,
            downsample=DownsampleConfig(enabled=True, phase_offset_frames=-1),
        )
        with pytest.raises(ValueError, match='phase_offset_frames must be >= 0'):
            run_downsample(cfg)

    def test_returns_output_dir_and_downsamples(self, tmp_path: Path) -> None:
        _make_video_files(tmp_path / 'videos', ['s0_cam0.mp4'])
        cfg = _minimal_cfg(tmp_path, downsample=DownsampleConfig(enabled=True, target_fps=10))
        with patch(f'{_EXT3D}.ProcessPoolExecutor', _SyncExecutor):
            with patch(f'{_EXT3D}.downsample_video'):
                result = run_downsample(cfg)
        assert result == tmp_path / 'out' / 'videos'


# ---------------------------------------------------------------------------
# TestSaveCameraParamsForFrames
# ---------------------------------------------------------------------------

class TestSaveCameraParamsForFrames:
    """Test the _save_camera_params_for_frames function."""

    def _cam_params(self) -> dict:
        return {
            'intrinsics': np.eye(3),
            'extrinsics': np.eye(4),
            'distortions': np.zeros(5),
            'width': 640,
            'height': 480,
        }

    def test_creates_npy_for_each_frame(self, tmp_path: Path) -> None:
        frame_idxs = np.array([0, 1, 2])
        _save_camera_params_for_frames(self._cam_params(), None, frame_idxs, tmp_path)
        for idx in frame_idxs:
            assert (tmp_path / f'img{idx:08d}.npy').exists()

    def test_includes_bbox_when_matching_orig_idx(self, tmp_path: Path) -> None:
        frame_idxs = np.array([0])
        bbox_dict = {0: {'x': 10, 'y': 20, 'w': 100, 'h': 200}}
        _save_camera_params_for_frames(self._cam_params(), bbox_dict, frame_idxs, tmp_path)
        data = np.load(str(tmp_path / 'img00000000.npy'), allow_pickle=True).item()
        assert 'bbox' in data
        assert data['bbox']['x'] == 10

    def test_no_bbox_when_bbox_dict_is_none(self, tmp_path: Path) -> None:
        frame_idxs = np.array([0])
        _save_camera_params_for_frames(self._cam_params(), None, frame_idxs, tmp_path)
        data = np.load(str(tmp_path / 'img00000000.npy'), allow_pickle=True).item()
        assert 'bbox' not in data

    def test_original_frame_idxs_used_for_bbox_lookup(self, tmp_path: Path) -> None:
        # file indices come from frame_idxs; bbox lookup uses original_frame_idxs
        frame_idxs = np.array([0])
        orig_idxs = np.array([5])
        bbox_dict = {5: {'x': 99}}
        _save_camera_params_for_frames(
            self._cam_params(), bbox_dict, frame_idxs, tmp_path,
            original_frame_idxs=orig_idxs,
        )
        data = np.load(str(tmp_path / 'img00000000.npy'), allow_pickle=True).item()
        assert data['bbox']['x'] == 99


# ---------------------------------------------------------------------------
# TestCopyMasksForFrames
# ---------------------------------------------------------------------------

class TestCopyMasksForFrames:
    """Test the _copy_masks_for_frames function."""

    def _write_masks(self, src_dir: Path, indices: list[int]) -> None:
        src_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            (src_dir / f'mask{idx:08d}.png').touch()

    def test_returns_count_of_copied_masks(self, tmp_path: Path) -> None:
        src_dir = tmp_path / 'src'
        self._write_masks(src_dir, [0, 1, 2])
        dst_dir = tmp_path / 'dst'
        dst_dir.mkdir()
        n = _copy_masks_for_frames(src_dir, np.array([0, 1, 2, 3]), dst_dir)
        assert n == 3   # frame 3 has no mask

    def test_missing_masks_silently_skipped(self, tmp_path: Path) -> None:
        src_dir = tmp_path / 'src'
        src_dir.mkdir()
        dst_dir = tmp_path / 'dst'
        dst_dir.mkdir()
        n = _copy_masks_for_frames(src_dir, np.array([0, 1]), dst_dir)
        assert n == 0

    def test_copies_with_correct_filename(self, tmp_path: Path) -> None:
        src_dir = tmp_path / 'src'
        self._write_masks(src_dir, [7])
        dst_dir = tmp_path / 'dst'
        dst_dir.mkdir()
        _copy_masks_for_frames(src_dir, np.array([7]), dst_dir)
        assert (dst_dir / 'mask00000007.png').exists()


# ---------------------------------------------------------------------------
# TestProcessViewTask
# ---------------------------------------------------------------------------

class TestProcessViewTask:
    """Test the _process_view_task function."""

    def _task(self, tmp_path: Path, with_masks: bool = False) -> dict:
        view_dir = tmp_path / 'view'
        mask_src = tmp_path / 'masks'
        if with_masks:
            mask_src.mkdir()
        return {
            'session_id': 'sess0',
            'cam_id': 'cam0',
            'video_path': tmp_path / 'vid.mp4',
            'view_output_dir': view_dir,
            'mask_source': mask_src,
            'frame_idxs': np.array([0, 1]),
            'original_frame_idxs': np.array([0, 1]),
            'cam_params': None,
            'bbox_dict': None,
            'n_digits': 8,
            'extension': 'png',
        }

    def test_calls_export_frames(self, tmp_path: Path) -> None:
        task = self._task(tmp_path)
        with patch(f'{_EXT3D}.export_frames') as mock_export:
            _process_view_task(task)
        mock_export.assert_called_once()

    def test_returns_result_dict_with_expected_keys(self, tmp_path: Path) -> None:
        task = self._task(tmp_path)
        with patch(f'{_EXT3D}.export_frames'):
            result = _process_view_task(task)
        assert result['session_id'] == 'sess0'
        assert result['cam_id'] == 'cam0'
        assert result['n_frames'] == 2
        assert result['n_masks'] == 0

    def test_skips_camera_params_when_none(self, tmp_path: Path) -> None:
        task = self._task(tmp_path)
        with patch(f'{_EXT3D}.export_frames'):
            with patch(f'{_EXT3D}._save_camera_params_for_frames') as mock_cam:
                _process_view_task(task)
        mock_cam.assert_not_called()


# ---------------------------------------------------------------------------
# TestAssembleDataset
# ---------------------------------------------------------------------------

class TestAssembleDataset:
    """Test the assemble_dataset function."""

    def test_raises_when_no_videos_found(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        with patch(f'{_EXT3D}.resolve_videos_dir', return_value=tmp_path / 'v'):
            with patch(f'{_EXT3D}.discover_videos', return_value={}):
                with pytest.raises(ValueError, match='no videos found'):
                    assemble_dataset(cfg)

    def test_raises_when_anchor_view_missing(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        video_dict = {'sess0': {'cam1': tmp_path / 'vid.mp4'}}
        with patch(f'{_EXT3D}.resolve_videos_dir', return_value=tmp_path / 'v'):
            with patch(f'{_EXT3D}.discover_videos', return_value=video_dict):
                with pytest.raises(ValueError, match='anchor view'):
                    assemble_dataset(cfg)

    def test_writes_info_json(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        video_dict = {'sess0': {'cam0': tmp_path / 'vid.mp4'}}
        task_result = {'session_id': 'sess0', 'cam_id': 'cam0', 'n_frames': 3, 'n_masks': 0}
        with patch(f'{_EXT3D}.resolve_videos_dir', return_value=tmp_path / 'v'):
            with patch(f'{_EXT3D}.discover_videos', return_value=video_dict):
                with patch(f'{_EXT3D}.select_frame_idxs_kmeans', return_value=np.array([0, 1, 2])):
                    with patch(f'{_EXT3D}.load_calibration', return_value=None):
                        with patch(f'{_EXT3D}.ProcessPoolExecutor', _SyncExecutor):
                            with patch(f'{_EXT3D}._process_view_task', return_value=task_result):
                                result = assemble_dataset(cfg)
        info = json.loads((result / 'info.json').read_text())
        assert info['dataset'] == 'test'
        assert info['number_of_videos'] == 1


# ---------------------------------------------------------------------------
# TestComputeNewSize
# ---------------------------------------------------------------------------

class TestComputeNewSize:
    """Test the _compute_new_size function."""

    def test_already_small_returns_unchanged(self) -> None:
        assert _compute_new_size(30, 40, 256) == (30, 40)

    def test_portrait_scales_height_to_target(self) -> None:
        # h < w → h is shorter side
        new_h, new_w = _compute_new_size(200, 400, 100)
        assert new_h == 100
        assert new_w == 200

    def test_landscape_scales_width_to_target(self) -> None:
        # w < h → w is shorter side
        new_h, new_w = _compute_new_size(400, 200, 100)
        assert new_w == 100
        assert new_h == 200

    def test_square_scales_both_sides(self) -> None:
        new_h, new_w = _compute_new_size(400, 400, 200)
        assert new_h == 200
        assert new_w == 200


# ---------------------------------------------------------------------------
# TestScaleIntrinsics
# ---------------------------------------------------------------------------

class TestScaleIntrinsics:
    """Test the _scale_intrinsics function."""

    def test_3x3_matrix_scales_fx_fy_cx_cy(self) -> None:
        K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
        result = _scale_intrinsics(K, scale_x=0.5, scale_y=0.5)
        assert result[0, 0] == pytest.approx(250.)   # fx
        assert result[1, 1] == pytest.approx(250.)   # fy
        assert result[0, 2] == pytest.approx(160.)   # cx
        assert result[1, 2] == pytest.approx(120.)   # cy

    def test_4_element_array_scales_correctly(self) -> None:
        params = np.array([500., 500., 320., 240.])
        result = _scale_intrinsics(params, scale_x=2.0, scale_y=0.5)
        assert result[0] == pytest.approx(1000.)   # fx * scale_x
        assert result[1] == pytest.approx(250.)    # fy * scale_y
        assert result[2] == pytest.approx(640.)    # cx * scale_x
        assert result[3] == pytest.approx(120.)    # cy * scale_y

    def test_identity_scale_returns_unchanged(self) -> None:
        K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
        result = _scale_intrinsics(K.copy(), scale_x=1.0, scale_y=1.0)
        np.testing.assert_array_almost_equal(result, K)


# ---------------------------------------------------------------------------
# TestResizeDataset
# ---------------------------------------------------------------------------

class TestResizeDataset:
    """Test the resize_dataset function."""

    def _make_dataset(self, dataset_dir: Path, h: int, w: int) -> Path:
        cam_dir = dataset_dir / 'sess0' / 'cam0'
        cam_dir.mkdir(parents=True)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(cam_dir / 'img00000000.png'), img)
        return cam_dir

    def test_raises_when_dataset_dir_missing(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        with pytest.raises(FileNotFoundError, match='dataset directory not found'):
            resize_dataset(cfg)

    def test_no_cam_dirs_returns_early(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        cfg.resize.size = 64
        dataset_dir = tmp_path / 'out' / 'dataset'
        dataset_dir.mkdir(parents=True)
        resize_dataset(cfg)   # should not raise

    def test_images_larger_than_target_are_resized(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        cfg.resize.size = 50
        dataset_dir = tmp_path / 'out' / 'dataset'
        cam_dir = self._make_dataset(dataset_dir, h=200, w=400)
        img_path = cam_dir / 'img00000000.png'
        resize_dataset(cfg)
        resized = cv2.imread(str(img_path))
        assert resized is not None
        new_h, new_w = resized.shape[:2]
        assert min(new_h, new_w) == 50

    def test_npy_intrinsics_updated_on_resize(self, tmp_path: Path) -> None:
        cfg = _minimal_cfg(tmp_path)
        cfg.resize.size = 50
        dataset_dir = tmp_path / 'out' / 'dataset'
        cam_dir = self._make_dataset(dataset_dir, h=200, w=400)
        npy_path = cam_dir / 'img00000000.npy'
        data = {
            'intrinsics': np.array([[500., 0., 200.], [0., 500., 100.], [0., 0., 1.]]),
            'width': 400, 'height': 200,
        }
        np.save(str(npy_path), data)
        resize_dataset(cfg)
        updated = np.load(str(npy_path), allow_pickle=True).item()
        # shorter side is h=200, target=50 → scale 0.25; new size (50, 100)
        assert updated['height'] == 50
        assert updated['width'] == 100
        assert updated['intrinsics'][0, 0] == pytest.approx(125.)   # fx * 0.25
