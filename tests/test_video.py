from pathlib import Path

import cv2
import numpy as np

from beast.video import check_codec_format


def test_check_codec_format(video_file):
    assert check_codec_format(video_file)


def test_reencode_video(video_file, tmpdir):
    from beast.video import reencode_video
    video_file_new = Path(tmpdir).joinpath('test.mp4')
    reencode_video(video_file, video_file_new)
    assert check_codec_format(video_file_new)


def test_copy_and_reformat_video_file(video_file, tmpdir):
    from beast.video import copy_and_reformat_video_file
    tmpdir = Path(tmpdir)
    # check when dst_dir exists
    video_file_new_1 = copy_and_reformat_video_file(video_file, tmpdir, remove_old=False)
    assert video_file.is_file()
    assert check_codec_format(video_file_new_1)
    # check when dst_dir does not exist
    dst_dir = tmpdir.joinpath('subdir')
    video_file_new_2 = copy_and_reformat_video_file(video_file, dst_dir, remove_old=False)
    assert video_file.is_file()
    assert check_codec_format(video_file_new_2)


def test_copy_and_reformat_video_directory(video_file, tmpdir):
    from beast.video import copy_and_reformat_video_directory
    src_dir = video_file.parent
    dst_dir = Path(tmpdir)
    copy_and_reformat_video_directory(src_dir, dst_dir)
    assert video_file.is_file()
    files = dst_dir.glob('*')
    for file in files:
        assert check_codec_format(file)


def test_get_frames_from_idxs(video_file):
    from beast.video import get_frames_from_idxs
    n_frames = 3
    frames = get_frames_from_idxs(video_file, np.arange(n_frames))
    assert frames.shape == (n_frames, 3, 406, 396)
    assert frames.dtype == np.uint8


def test_compute_video_motion_energy(video_file):
    from beast.video import compute_video_motion_energy
    me = compute_video_motion_energy(video_file)
    cap = cv2.VideoCapture(video_file)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert n_frames == len(me)
    assert np.isnan(me).sum() == 0


def test_read_nth_frames(video_file):
    from beast.video import read_nth_frames
    resize_dims = 8
    frames = read_nth_frames(video_file=video_file, n=10, resize_dims=resize_dims)
    assert frames.shape == (100, resize_dims, resize_dims, 3)
