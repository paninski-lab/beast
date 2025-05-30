
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typeguard import typechecked

from beast.video import (
    compute_video_motion_energy,
    get_frames_from_idxs,
)


@typechecked
def extract_frames(
    input_path: Path | str,
    output_dir: Path | str,
    frames_per_video: int = 500,
    method: str = 'pca_kmeans',
    num_workers: int = 8,
) -> dict:
    """Extract representative frames from videos using intelligent sampling methods.

    Processes all video files in a directory and extracts a subset of frames using
    advanced selection algorithms to capture diverse visual content. Each video's
    frames are saved to a separate subdirectory named after the video file.

    Parameters
    ----------
    input_path: directory containing video files (.mp4 and .avi formats supported)
    output_dir: directory where extracted frames will be saved; creates subdirectories
        for each video named after the video filename (without extension)
    frames_per_video: maximum number of frames to extract from each video
    method: frame selection method; currently supported:
      - 'pca_kmeans': Uses PCA dimensionality reduction followed by k-means clustering to select
        diverse, representative frames
    num_workers: number of parallel workers for processing (currently unused but reserved
      for future parallel processing implementation)

    Returns
    -------
    Summary statistics containing:
        - 'total_frames': Total number of frames extracted across all videos
        - 'total_videos': Number of videos processed

    Raises
    ------
    NotImplementedError
        If an unsupported frame selection method is specified

    Examples
    --------
    Extract frames from all videos in a directory:

    >>> results = extract_frames(
    ...     input_path="./videos",
    ...     output_dir="./extracted_frames",
    ...     frames_per_video=300,
    ...     method='pca_kmeans'
    ... )
    >>> print(f"Extracted {results['total_frames']} frames from {results['total_videos']} videos")

    Process fewer frames per video:

    >>> results = extract_frames(
    ...     input_path=Path("./raw_videos"),
    ...     output_dir=Path("./training_data"),
    ...     frames_per_video=100
    ... )

    Directory Structure
    -------------------
    Input directory:
        input_path/
        ├── video1.mp4
        ├── video2.avi
        └── video3.mp4

    Output directory:
        output_dir/
        ├── video1/
        │   ├── frame_001.png
        │   ├── frame_045.png
        │   └── ...
        ├── video2/
        │   ├── frame_012.png
        │   └── ...
        └── video3/
            └── ...

    Notes
    -----
    - Only processes .mp4 and .avi video files
    - The PCA-kmeans method resizes frames to 32x32 pixels for analysis, but exports frames at
      original resolution
    - Context frames (neighboring frames) are included in the selection
    - Progress information is logged during processing
    - Frame selection aims to maximize visual diversity while avoiding redundant or similar frames

    """

    print(f'Extracting frames from: {input_path}')
    print(f'Saving to: {output_dir}')
    print(f'Method: {method}')
    print(f'Frames per video: {frames_per_video}')

    video_files = list(input_path.glob('*.mp4')) + list(input_path.glob('*.avi'))
    total_videos = 0
    total_frames = 0
    for video_file in video_files:
        if method == 'pca_kmeans':
            idxs = select_frame_idxs_kmeans(
                video_file=video_file,
                resize_dims=32,
                n_frames_to_select=frames_per_video,
            )
        else:
            raise NotImplementedError

        n_digits = 8
        extension = 'png'
        save_dir = output_dir.joinpath(video_file.stem)
        export_frames(
            video_file=video_file,
            output_dir=save_dir,
            frame_idxs=idxs,
            context_frames=1,
            n_digits=n_digits,
            extension=extension,
        )

        # save csv file inside same output directory
        frames_to_label = np.array([
            "img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in idxs
        ])
        csv_path = save_dir / 'selected_frames.csv'
        np.savetxt(csv_path, np.sort(frames_to_label), delimiter=',', fmt='%s')
        print(f'Saved selected frame list to: {csv_path}')

        total_videos += 1
        total_frames += len(idxs)

    return {
        'total_frames': total_frames,
        'total_videos': total_videos,
    }


@typechecked
def _run_kmeans(data: np.ndarray, n_clusters: int, seed: int = 0) -> tuple:
    np.random.seed(seed)
    kmeans_obj = KMeans(n_clusters, n_init='auto')
    kmeans_obj.fit(data)
    cluster_labels = kmeans_obj.labels_
    cluster_centers = kmeans_obj.cluster_centers_
    return cluster_labels, cluster_centers


@typechecked
def select_frame_idxs_kmeans(
    video_file: str | Path,
    resize_dims: int = 64,
    n_frames_to_select: int = 20,
    frame_range: list = [0, 1],
) -> np.ndarray:
    """Select distinct frames during movement using kmeans on motion-energy thresholded frame PCs.

    Parameters
    ----------
    video_file: absolute path to video file
    resize_dims: number of pixels (in both dimensions) to downsample video before computing motion
        energy; exported frames will retain original resolution
    n_frames_to_select: number of anchor frames to select per video
    frame_range: define range of video considered for frame extraction; for example, [0, 1] uses
        the full video, while [0.25, 0.75] uses the central 50% of the video

    Returns
    -------
    frames array of shape (n_frames_to_select, n_channels, ypix, xpix)

    """

    # check inputs
    assert frame_range[0] >= 0
    assert frame_range[1] <= 1

    # read all frames, reshape, chop off unwanted portions of beginning/end
    print('computing motion energy...')
    me, frames = compute_video_motion_energy(
        video_file=video_file,
        resize_dims=resize_dims,
        return_frames=True,
    )
    frame_count = me.shape[0]
    beg_frame = int(float(frame_range[0]) * frame_count)
    end_frame = int(float(frame_range[1]) * frame_count) - 2  # leave room for context
    assert (end_frame - beg_frame) >= n_frames_to_select, 'valid video segment too short!'

    # find high me frames, defined as those with me larger than nth percentile me
    prctile = 50 if frame_count < 1e5 else 75  # take fewer frames if there are many
    idxs_high_me = np.where(me > np.percentile(me, prctile))[0]
    # just use all frames if the user wants to extract a large fraction of the frames
    # (helpful for very short videos)
    if len(idxs_high_me) < n_frames_to_select:
        idxs_high_me = np.arange(me.shape[0])

    # compute pca over high me frames
    print('performing pca over high motion energy frames...')
    pca_obj = PCA(n_components=np.min([frames[idxs_high_me].shape[0], 32]))
    embedding = pca_obj.fit_transform(X=frames[idxs_high_me])
    del frames  # free up memory

    # cluster low-d pca embeddings
    print('performing kmeans clustering...')
    _, centers = _run_kmeans(data=embedding, n_clusters=n_frames_to_select)
    # centers is initially of shape (n_clusters, n_pcs); reformat
    centers = centers.T[None, :]

    # find high me frame that is closest to each cluster center
    # embedding is shape (n_frames, n_pcs)
    # centers is shape (1, n_pcs, n_clusters)
    dists = np.linalg.norm(embedding[:, :, None] - centers, axis=1)
    # dists is shape (n_frames, n_clusters)
    idxs_prototypes_ = np.argmin(dists, axis=0)
    # now index into high me frames to get overall indices, add offset
    idxs_prototypes = idxs_high_me[idxs_prototypes_] + beg_frame

    return idxs_prototypes


@typechecked
def export_frames(
    video_file: str | Path,
    output_dir: str | Path,
    frame_idxs: np.ndarray,
    extension: str = 'png',
    n_digits: int = 8,
    context_frames: int = 1,
) -> None:
    """Export selected frames from a video to individual png files.

    Parameters
    ----------
    video_file: absolute path to video file from which to select frames
    output_dir: absolute path to parent directory in which selected frames are saved
    frame_idxs: indices of frames to export
    extension: only 'png' currently supported
    n_digits: number of digits in image names
    context_frames: number of frames on either side of selected frame to also save

    """

    # expand frame_idxs to include context frames
    if context_frames > 0:
        cap = cv2.VideoCapture(video_file)
        context_vec = np.arange(-context_frames, context_frames + 1)
        frame_idxs = (frame_idxs[None, :] + context_vec[:, None]).flatten()
        frame_idxs.sort()
        frame_idxs = frame_idxs[frame_idxs >= 0]
        frame_idxs = frame_idxs[frame_idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_idxs = np.unique(frame_idxs)
        cap.release()

    # load frames from video
    frames = get_frames_from_idxs(video_file, frame_idxs)

    # save out frames
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, idx in zip(frames, frame_idxs):
        cv2.imwrite(
            filename=output_dir.joinpath(f'img{str(idx).zfill(n_digits)}.{extension}'),
            img=cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2BGR),
        )
