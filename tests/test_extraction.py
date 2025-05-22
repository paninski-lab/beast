from pathlib import Path
import pytest

import numpy as np


def test_extract_frames(video_file, tmpdir):

    from beast.extraction import extract_frames

    tmpdir = Path(tmpdir)
    frames_per_video = 5

    details = extract_frames(
        input_path=video_file.parent,
        output_dir=tmpdir,
        frames_per_video=frames_per_video,
        method='pca_kmeans',
    )
    assert details['total_frames'] == frames_per_video
    assert details['total_videos'] == 1

    with pytest.raises(NotImplementedError):
        extract_frames(
            input_path=video_file.parent,
            output_dir=tmpdir,
            frames_per_video=frames_per_video,
            method='uniform',
        )

def test_run_kmeans():

    from beast.extraction import _run_kmeans

    n_samples = int(50)
    n_features = int(5)
    n_clusters = 10

    data_to_cluster = np.random.rand(n_samples, n_features)
    cluster, centers = _run_kmeans(data_to_cluster, n_clusters)

    assert len(cluster) == n_samples
    assert len(np.unique(cluster)) == n_clusters
    assert centers.shape == (n_clusters, n_features)


def test_select_frame_idxs_kmeans(video_file):

    from beast.extraction import select_frame_idxs_kmeans

    # test: small number of idxs
    resize_dims = 8
    n_clusters = 5
    idxs = select_frame_idxs_kmeans(
        video_file=video_file,
        resize_dims=resize_dims,
        n_frames_to_select=n_clusters,
    )
    assert len(idxs) == n_clusters

    # test: too many idxs
    n_clusters = 1000
    with pytest.raises(AssertionError):
        select_frame_idxs_kmeans(
            video_file=video_file,
            resize_dims=resize_dims,
            n_frames_to_select=n_clusters,
        )

    # test: very large number of idxs
    n_clusters = 990
    idxs = select_frame_idxs_kmeans(
        video_file=video_file,
        resize_dims=resize_dims,
        n_frames_to_select=n_clusters,
    )
    assert len(idxs) == n_clusters


def test_export_frames(video_file, tmpdir):

    from beast.extraction import export_frames

    tmpdir = Path(tmpdir)

    # multiple frames, no context
    save_dir_0 = tmpdir.joinpath('labeled-frames-0')
    idxs = np.array([0, 2, 4, 6, 8, 10])
    export_frames(
        video_file=video_file,
        output_dir=save_dir_0,
        frame_idxs=idxs,
        context_frames=0,
    )
    assert len(list(save_dir_0.glob('*'))) == len(idxs)

    # multiple frames, 2-frame context
    save_dir_1 = tmpdir.joinpath('labeled-frames-1')
    idxs = np.array([5, 10, 15, 20])
    export_frames(
        video_file=video_file,
        output_dir=save_dir_1,
        frame_idxs=idxs,
        context_frames=2,
    )
    assert len(list(save_dir_1.glob('*'))) == 5 * len(idxs)

    # single frame, 2-frame context
    save_dir_2 = tmpdir.joinpath('labeled-frames-2')
    idxs = np.array([10])
    export_frames(
        video_file=video_file,
        output_dir=save_dir_2,
        frame_idxs=idxs,
        context_frames=2,
    )
    assert len(list(save_dir_2.glob('*'))) == 5 * len(idxs)
