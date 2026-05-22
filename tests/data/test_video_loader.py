"""Tests for the video frame iterator."""

import pytest

from beast.data.video import VideoFrameIterator


class TestVideoFrameIterator:
    """Test the VideoFrameIterator class."""

    def test_basic_iteration(self, video_file) -> None:
        iterator = VideoFrameIterator(video_file=video_file, batch_size=8)
        batch = next(iterator)
        assert batch['image'].shape == (8, 3, 224, 224)
        assert 'video' in batch
        assert 'idx' in batch
        assert 'image_path' in batch

    def test_different_batch_size(self, video_file) -> None:
        iterator = VideoFrameIterator(video_file=video_file, batch_size=4)
        batch = next(iterator)
        assert batch['image'].shape == (4, 3, 224, 224)

    def test_all_frames_consumed(self, video_file) -> None:
        iterator = VideoFrameIterator(video_file=video_file, batch_size=32)
        n_frames = 0
        for batch in iterator:
            n_frames += batch['image'].shape[0]
        assert n_frames == iterator.total_frames

    def test_invalid_video_path_raises(self) -> None:
        with pytest.raises(ValueError, match='Cannot open video file'):
            VideoFrameIterator(video_file='/nonexistent/video.mp4')

    def test_reset_restarts_iteration(self, video_file) -> None:
        iterator = VideoFrameIterator(video_file=video_file, batch_size=8)
        # advance a few batches
        next(iterator)
        next(iterator)
        assert iterator.current_frame > 0
        # Act
        iterator.reset()
        # Assert
        assert iterator.current_frame == 0
        batch = next(iterator)
        assert batch['image'].shape == (8, 3, 224, 224)
