"""Tests for video I/O utilities."""

import shutil
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from beast.video import (
    check_codec_format,
    compute_video_motion_energy,
    copy_and_reformat_video_directory,
    copy_and_reformat_video_file,
    get_frames_from_idxs,
    read_nth_frames,
    reencode_video,
)


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
