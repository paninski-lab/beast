"""Tests for frame extraction utilities."""

import numpy as np
import pytest


class TestExtractFrames:
    """Test the extract_frames function."""

    def test_extract_frames_pca_kmeans(self, video_file, tmp_path) -> None:
        from beast.extraction import extract_frames
        frames_per_video = 5
        details = extract_frames(
            input_path=video_file.parent,
            output_dir=tmp_path,
            frames_per_video=frames_per_video,
            method='pca_kmeans',
        )
        assert details['total_frames'] == frames_per_video
        assert details['total_videos'] == 1
        output_dir = tmp_path / video_file.stem
        assert output_dir.is_dir()
        assert (output_dir / 'selected_frames.csv').is_file()

    def test_unimplemented_method_raises(self, video_file, tmp_path) -> None:
        from beast.extraction import extract_frames
        with pytest.raises(NotImplementedError):
            extract_frames(
                input_path=video_file.parent,
                output_dir=tmp_path,
                frames_per_video=5,
                method='uniform',
            )


class TestRunKmeans:
    """Test the _run_kmeans function."""

    def test_cluster_shapes(self) -> None:
        from beast.extraction import _run_kmeans
        n_samples = 50
        n_features = 5
        n_clusters = 10
        data = np.random.rand(n_samples, n_features)
        cluster, centers = _run_kmeans(data, n_clusters)
        assert len(cluster) == n_samples
        assert len(np.unique(cluster)) == n_clusters
        assert centers.shape == (n_clusters, n_features)


class TestSelectFrameIdxsKmeans:
    """Test the select_frame_idxs_kmeans function."""

    def test_returns_correct_number_of_frames(self, video_file) -> None:
        from beast.extraction import select_frame_idxs_kmeans
        n_clusters = 5
        idxs = select_frame_idxs_kmeans(
            video_file=video_file,
            resize_dims=8,
            n_frames_to_select=n_clusters,
        )
        assert len(idxs) == n_clusters

    def test_too_many_frames_raises(self, video_file) -> None:
        from beast.extraction import select_frame_idxs_kmeans
        with pytest.raises(AssertionError):
            select_frame_idxs_kmeans(
                video_file=video_file,
                resize_dims=8,
                n_frames_to_select=1000,
            )

    def test_large_number_of_frames(self, video_file) -> None:
        from beast.extraction import select_frame_idxs_kmeans
        n_clusters = 990
        idxs = select_frame_idxs_kmeans(
            video_file=video_file,
            resize_dims=8,
            n_frames_to_select=n_clusters,
        )
        assert len(idxs) == n_clusters


class TestExportFrames:
    """Test the export_frames function."""

    def test_export_multiple_frames_no_context(self, video_file, tmp_path) -> None:
        from beast.extraction import export_frames
        save_dir = tmp_path / 'frames-0'
        idxs = np.array([0, 2, 4, 6, 8, 10])
        export_frames(
            video_file=video_file, output_dir=save_dir, frame_idxs=idxs, context_frames=0,
        )
        assert len(list(save_dir.glob('*'))) == len(idxs)

    def test_export_multiple_frames_with_context(self, video_file, tmp_path) -> None:
        from beast.extraction import export_frames
        save_dir = tmp_path / 'frames-1'
        idxs = np.array([5, 10, 15, 20])
        export_frames(
            video_file=video_file, output_dir=save_dir, frame_idxs=idxs, context_frames=2,
        )
        assert len(list(save_dir.glob('*'))) == 5 * len(idxs)

    def test_export_single_frame_with_context(self, video_file, tmp_path) -> None:
        from beast.extraction import export_frames
        save_dir = tmp_path / 'frames-2'
        idxs = np.array([10])
        export_frames(
            video_file=video_file, output_dir=save_dir, frame_idxs=idxs, context_frames=2,
        )
        assert len(list(save_dir.glob('*'))) == 5 * len(idxs)
