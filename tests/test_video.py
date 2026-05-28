"""Tests for video I/O utilities."""

import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from beast.video import (
    _get_video_files,
    check_codec_format,
    compute_video_motion_energy,
    copy_and_reformat_video_directory,
    copy_and_reformat_video_file,
    cut_video,
    discover_videos,
    downsample_video,
    get_frames_from_idxs,
    get_video_stats,
    read_nth_frames,
    reencode_video,
)

_ROOT = Path(__file__).parent


@pytest.fixture
def videos_dir() -> Path:
    return _ROOT / 'testing_data' / 'videos'


class TestCheckCodecFormat:
    """Test the check_codec_format function."""

    def test_valid_h264_video(self, video_file) -> None:
        assert check_codec_format(video_file)

    def test_non_h264_video_returns_false(self, tmp_path) -> None:
        # Arrange — mock ffprobe output to omit 'h264'
        mock_result = Mock()
        mock_result.stderr = 'mpeg4, yuv420p'
        with patch('beast.video.subprocess.run', return_value=mock_result):
            result = check_codec_format(tmp_path / 'fake.mp4')
        # Assert
        assert result is False


class TestReencodeVideo:
    """Test the reencode_video function."""

    def test_reencoded_video_is_h264(self, video_file, tmp_path) -> None:

        video_file_new = tmp_path / 'test.mp4'
        reencode_video(video_file, video_file_new)
        assert check_codec_format(video_file_new)

    def test_nonexistent_input_raises(self, tmp_path) -> None:
        # Arrange
        nonexistent = tmp_path / 'ghost.mp4'
        # Act / Assert
        with pytest.raises(FileNotFoundError):
            reencode_video(nonexistent, tmp_path / 'output.mp4')


class TestCopyAndReformatVideoFile:
    """Test the copy_and_reformat_video_file function."""

    def test_copy_valid_video(self, video_file, tmp_path) -> None:

        result = copy_and_reformat_video_file(video_file, tmp_path, remove_old=False)
        assert video_file.is_file()
        assert result is not None
        assert check_codec_format(result)

    def test_copy_to_nonexistent_subdir(self, video_file, tmp_path) -> None:

        dst_dir = tmp_path / 'subdir'
        result = copy_and_reformat_video_file(video_file, dst_dir, remove_old=False)
        assert video_file.is_file()
        assert result is not None
        assert check_codec_format(result)

    def test_dst_already_exists_returns_early(self, video_file, tmp_path) -> None:
        # Arrange — copy once so the destination already exists

        result_first = copy_and_reformat_video_file(video_file, tmp_path, remove_old=False)
        assert result_first is not None
        # Act — second call with same dst should return early (line 74)
        result_second = copy_and_reformat_video_file(video_file, tmp_path, remove_old=False)
        # Assert — returns the existing path without re-copying
        assert result_second == result_first

    def test_nonexistent_src_returns_none(self, tmp_path) -> None:
        # Arrange — source file does not exist

        nonexistent = tmp_path / 'ghost.mp4'
        # Act
        result = copy_and_reformat_video_file(nonexistent, tmp_path)
        # Assert — function prints a warning and returns None
        assert result is None

    def test_copy_with_remove_old_correct_codec(self, video_file, tmp_path) -> None:
        # Arrange — copy fixture so remove_old=True doesn't delete the original
        src = tmp_path / 'src.mp4'
        shutil.copyfile(video_file, src)
        # Act — exercises the remove_old=True branch when codec is already correct
        result = copy_and_reformat_video_file(src, tmp_path / 'dst', remove_old=True)
        # Assert — function returns the destination path without raising
        assert result is not None

    def test_copy_reencodes_non_h264_video(self, video_file, tmp_path) -> None:
        # Arrange — mock codec check so the reencode branch is taken
        with patch('beast.video.check_codec_format', return_value=False):
            # Act
            result = copy_and_reformat_video_file(video_file, tmp_path / 'dst', remove_old=False)
        # Assert
        assert result is not None
        assert result.is_file()

    def test_copy_reencodes_non_h264_video_with_remove_old(self, video_file, tmp_path) -> None:
        # Arrange — copy fixture so remove_old=True can safely unlink the copy
        src = tmp_path / 'src.mp4'
        shutil.copyfile(video_file, src)
        with patch('beast.video.check_codec_format', return_value=False):
            # Act
            result = copy_and_reformat_video_file(src, tmp_path / 'dst', remove_old=True)
        # Assert — src was unlinked and dst was created
        assert not src.exists()
        assert result is not None
        assert result.is_file()


class TestCopyAndReformatVideoDirectory:
    """Test the copy_and_reformat_video_directory function."""

    def test_copies_videos_in_directory(self, video_file, tmp_path) -> None:

        src_dir = video_file.parent
        dst_dir = tmp_path
        copy_and_reformat_video_directory(src_dir, dst_dir)
        assert video_file.is_file()
        copied = list(dst_dir.glob('*'))
        assert len(copied) > 0, 'no files were copied to dst_dir'
        for file in copied:
            assert check_codec_format(file)


class TestGetFramesFromIdxs:
    """Test the get_frames_from_idxs function."""

    def test_basic_frame_loading(self, video_file) -> None:

        n_frames = 3
        frames = get_frames_from_idxs(video_file, np.arange(n_frames))
        assert frames.shape == (n_frames, 3, 406, 396)
        assert frames.dtype == np.uint8

    def test_no_video_file_or_cap_raises(self) -> None:
        # Arrange — both video_file and cap are None

        # Act / Assert
        with pytest.raises(ValueError, match='video_file must be provided when cap is None'):
            get_frames_from_idxs(video_file=None, idxs=np.arange(3))

    def test_beyond_end_of_video_returns_partial_frames(self, video_file) -> None:
        # Arrange — request an index far past the last frame
        cap = cv2.VideoCapture(str(video_file))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idxs = np.array([0, n_frames + 100])
        # Act
        frames = get_frames_from_idxs(video_file, idxs)
        # Assert — array has the right first dimension; trailing frame is blank zeros
        assert frames.shape[0] == 2
        assert np.all(frames[1] == 0)


class TestComputeVideoMotionEnergy:
    """Test the compute_video_motion_energy function."""

    def test_motion_energy_shape_and_values(self, video_file) -> None:

        me = compute_video_motion_energy(video_file)
        cap = cv2.VideoCapture(video_file)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        assert n_frames == len(me)
        assert np.isnan(me).sum() == 0


class TestReadNthFrames:
    """Test the read_nth_frames function."""

    def test_basic_frame_reading(self, video_file) -> None:

        resize_dims = 8
        frames = read_nth_frames(video_file=video_file, n=10, resize_dims=resize_dims)
        assert frames.shape == (100, resize_dims, resize_dims, 3)

    def test_invalid_video_file_raises(self) -> None:

        with pytest.raises(OSError, match='Error opening video file'):
            read_nth_frames('/nonexistent/video.mp4')


class TestGetVideoStats:
    """Test the get_video_stats function."""

    def test_returns_expected_keys(self, video_file) -> None:
        stats = get_video_stats(video_file)
        assert set(stats.keys()) == {
            'fps', 'width', 'height', 'total_frames', 'duration_sec', 'codec',
        }

    def test_fps_is_positive(self, video_file) -> None:
        stats = get_video_stats(video_file)
        assert stats['fps'] > 0

    def test_dimensions_are_positive(self, video_file) -> None:
        stats = get_video_stats(video_file)
        assert stats['width'] > 0
        assert stats['height'] > 0

    def test_duration_consistent_with_frame_count(self, video_file) -> None:
        stats = get_video_stats(video_file)
        expected = round(stats['total_frames'] / stats['fps'], 2)
        assert abs(stats['duration_sec'] - expected) < 0.1

    def test_invalid_path_raises(self, tmp_path) -> None:
        with pytest.raises(OSError, match='Could not open video'):
            get_video_stats(tmp_path / 'nonexistent.mp4')


class TestGetVideoFiles:
    """Test the _get_video_files function."""

    def test_finds_files_by_extension(self, tmp_path) -> None:
        (tmp_path / 'a.mp4').touch()
        (tmp_path / 'b.avi').touch()
        (tmp_path / 'c.txt').touch()
        assert [f.name for f in _get_video_files(tmp_path, ['mp4'])] == ['a.mp4']

    def test_multiple_extensions(self, tmp_path) -> None:
        (tmp_path / 'a.mp4').touch()
        (tmp_path / 'b.avi').touch()
        names = {f.name for f in _get_video_files(tmp_path, ['mp4', 'avi'])}
        assert names == {'a.mp4', 'b.avi'}

    def test_returns_sorted(self, tmp_path) -> None:
        (tmp_path / 'b.mp4').touch()
        (tmp_path / 'a.mp4').touch()
        assert [f.name for f in _get_video_files(tmp_path, ['mp4'])] == ['a.mp4', 'b.mp4']

    def test_empty_directory_returns_empty_list(self, tmp_path) -> None:
        assert _get_video_files(tmp_path, ['mp4']) == []


class TestDiscoverVideos:
    """Test the discover_videos function."""

    # default pattern matches <session>_<cam>.mp4
    _PATTERN = r'^(?P<session>.+)_(?P<cam>[^_]+)\.mp4$'

    def test_groups_by_session_and_cam(self, videos_dir) -> None:
        # Arrange — testing_data/videos has test_vid_with_fr_{b,g,r}.mp4
        result = discover_videos(videos_dir, self._PATTERN, ['mp4'])
        assert len(result) == 1
        (session, cams) = next(iter(result.items()))
        assert session == 'test_vid_with_fr'
        assert set(cams.keys()) == {'b', 'g', 'r'}

    def test_paths_are_resolved(self, videos_dir) -> None:
        result = discover_videos(videos_dir, self._PATTERN, ['mp4'])
        for cams in result.values():
            for path in cams.values():
                assert path.is_absolute()

    def test_unmatched_files_skipped(self, tmp_path) -> None:
        # 'video.mp4' has no underscore so it won't match the pattern
        (tmp_path / 'video.mp4').touch()
        result = discover_videos(tmp_path, self._PATTERN, ['mp4'])
        assert result == {}

    def test_extension_filter_applied(self, videos_dir) -> None:
        # requesting only .avi should return nothing from an mp4-only directory
        result = discover_videos(videos_dir, self._PATTERN, ['avi'])
        assert result == {}


class TestCutVideo:
    """Test the cut_video function."""

    def test_calls_ffmpeg_with_trim_filter(self, tmp_path) -> None:
        mock_result = Mock(returncode=0)
        with patch('beast.video.subprocess.run', return_value=mock_result) as mock_run:
            cut_video(Path('input.mp4'), tmp_path / 'output.mp4', start_frame=10, end_frame=50)
        cmd = mock_run.call_args[0][0]
        vf_arg = next((a for a in cmd if 'trim=start_frame' in a), None)
        assert vf_arg is not None
        assert 'start_frame=10' in vf_arg
        assert 'end_frame=51' in vf_arg  # end_frame + 1

    def test_threads_arg_included_when_set(self, tmp_path) -> None:
        mock_result = Mock(returncode=0)
        with patch('beast.video.subprocess.run', return_value=mock_result) as mock_run:
            cut_video(Path('in.mp4'), tmp_path / 'out.mp4', 0, 10, threads=4)
        cmd = mock_run.call_args[0][0]
        assert '-threads' in cmd
        assert '4' in cmd

    def test_creates_parent_directory(self, tmp_path) -> None:
        mock_result = Mock(returncode=0)
        subdir = tmp_path / 'new_subdir'
        with patch('beast.video.subprocess.run', return_value=mock_result):
            cut_video(Path('in.mp4'), subdir / 'out.mp4', 0, 10)
        assert subdir.exists()

    def test_ffmpeg_failure_raises(self, tmp_path) -> None:
        mock_result = Mock(returncode=1, stderr='some error')
        with patch('beast.video.subprocess.run', return_value=mock_result):
            with pytest.raises(RuntimeError, match='ffmpeg cut failed'):
                cut_video(Path('in.mp4'), tmp_path / 'out.mp4', 0, 10)


class TestDownsampleVideo:
    """Test the downsample_video function."""

    def test_basic_fps_filter(self, tmp_path) -> None:
        mock_result = Mock(returncode=0)
        with patch('beast.video.subprocess.run', return_value=mock_result) as mock_run:
            downsample_video(Path('in.mp4'), tmp_path / 'out.mp4', target_fps=5.0)
        cmd = mock_run.call_args[0][0]
        assert any('fps=5.0' in a for a in cmd)

    def test_max_frames_flag_included(self, tmp_path) -> None:
        mock_result = Mock(returncode=0)
        with patch('beast.video.subprocess.run', return_value=mock_result) as mock_run:
            downsample_video(Path('in.mp4'), tmp_path / 'out.mp4', target_fps=5.0, max_frames=100)
        cmd = mock_run.call_args[0][0]
        assert '-frames:v' in cmd
        assert '100' in cmd

    def test_no_target_fps_omits_vf(self, tmp_path) -> None:
        mock_result = Mock(returncode=0)
        with patch('beast.video.subprocess.run', return_value=mock_result) as mock_run:
            downsample_video(Path('in.mp4'), tmp_path / 'out.mp4', target_fps=None, max_frames=50)
        cmd = mock_run.call_args[0][0]
        assert '-vf' not in cmd
        assert '-frames:v' in cmd

    def test_phase_offset_uses_trim_filter(self, tmp_path) -> None:
        # source_fps=30, target_fps=5 → stride=6, phase_offset_frames=1 is valid
        mock_result = Mock(returncode=0)
        mock_stats = {'fps': 30.0, 'width': 640, 'height': 480, 'total_frames': 300,
                      'duration_sec': 10.0, 'codec': 'avc1'}
        with patch('beast.video.subprocess.run', return_value=mock_result) as mock_run, \
                patch('beast.video.get_video_stats', return_value=mock_stats):
            downsample_video(Path('in.mp4'), tmp_path / 'out.mp4', target_fps=5.0,
                             phase_offset_frames=1)
        cmd = mock_run.call_args[0][0]
        vf_arg = next((a for a in cmd if 'trim=start_frame' in a), None)
        assert vf_arg is not None
        assert 'fps=5.0' in vf_arg

    def test_phase_offset_too_large_raises(self, tmp_path) -> None:
        # stride=6, phase_offset_frames=6 → 6 >= 6 → raises
        mock_stats = {'fps': 30.0, 'width': 640, 'height': 480, 'total_frames': 300,
                      'duration_sec': 10.0, 'codec': 'avc1'}
        with patch('beast.video.get_video_stats', return_value=mock_stats):
            with pytest.raises(RuntimeError, match='overlap the K=0 training set'):
                downsample_video(Path('in.mp4'), tmp_path / 'out.mp4', target_fps=5.0,
                                 phase_offset_frames=6)

    def test_stride_too_small_raises(self, tmp_path) -> None:
        # source_fps=25, target_fps=25 → stride=1 < 2 → raises
        mock_stats = {'fps': 25.0, 'width': 640, 'height': 480, 'total_frames': 250,
                      'duration_sec': 10.0, 'codec': 'avc1'}
        with patch('beast.video.get_video_stats', return_value=mock_stats):
            with pytest.raises(RuntimeError, match='phase-shift requires stride'):
                downsample_video(Path('in.mp4'), tmp_path / 'out.mp4', target_fps=25.0,
                                 phase_offset_frames=1)

    def test_ffmpeg_failure_raises(self, tmp_path) -> None:
        mock_result = Mock(returncode=1, stderr='error msg')
        with patch('beast.video.subprocess.run', return_value=mock_result):
            with pytest.raises(RuntimeError, match='ffmpeg downsample failed'):
                downsample_video(Path('in.mp4'), tmp_path / 'out.mp4', target_fps=5.0)
