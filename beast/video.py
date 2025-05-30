import logging
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
from typeguard import typechecked

_logger = logging.getLogger('BEAST.VIDEO')


@typechecked
def check_codec_format(input_file: str | Path) -> bool:
    """Run FFprobe command to check if video codec and pixel format match DALI requirements."""
    ffmpeg_cmd = f'ffmpeg -i {str(input_file)}'
    output_str = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    # stderr because the ffmpeg command has no output file, but the stderr still has codec info.
    output_str = output_str.stderr
    # search for correct codec (h264) and pixel format (yuv420p)
    if output_str.find('h264') != -1 and output_str.find('yuv420p') != -1:
        # print('Video uses H.264 codec')
        is_codec = True
    else:
        # print('Video does not use H.264 codec')
        is_codec = False
    return is_codec


@typechecked
def reencode_video(input_file: str | Path, output_file: str | Path) -> None:
    """Reencode video into H.264 format using ffmpeg from a subprocess.

    Parameters
    ----------
    input_file: absolute path to existing video
    output_file: absolute path to new video

    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    # check input file exists
    assert input_file.is_file(), 'input video does not exist.'
    # check directory for saving outputs exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = (
        f'ffmpeg -i {str(input_file)} '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p '
        f'-c:a copy -y {str(output_file)}'
    )
    subprocess.run(ffmpeg_cmd, shell=True)


@typechecked
def copy_and_reformat_video_file(
    video_file: str | Path,
    dst_dir: str | Path,
    remove_old: bool = False
) -> Path | None:
    """Copy a single video and reencode to be DALI compatible if necessary.

    Parameters
    ----------
    video_file: absolute path to existing video
    dst_dir: absolute path to parent directory for copied video
    remove_old: delete original video after copy is made

    """

    src = Path(video_file)

    # make sure copied vid has mp4 extension
    dst = Path(dst_dir).joinpath(video_file.stem + '.mp4')

    # check 0: do we even need to reformat?
    if dst.is_file():
        return dst

    # check 1: does file exist?
    if not src.is_file():
        _logger.info(f'{src} does not exist! skipping')
        return None

    # check 2: is file in the correct format for DALI?
    video_file_correct_codec = check_codec_format(src)

    # reencode/rename
    if not video_file_correct_codec:
        _logger.info(f're-encoding {src} to be compatable with DALI video reader')
        reencode_video(src, dst)
        # remove old video
        if remove_old:
            src.unlink()
    else:
        # make dir to write into
        dst_dir.mkdir(parents=True, exist_ok=True)
        # rename
        if remove_old:
            src.rename(src)
        else:
            shutil.copyfile(src, dst)

    return dst


@typechecked
def copy_and_reformat_video_directory(
    src_dir: str | Path,
    dst_dir: str | Path,
    remove_old: bool = False
) -> None:
    """Copy a directory of videos and reencode to be DALI compatible if necessary.

    Parameters
    ----------
    src_dir: absolute path to existing directory of videos
    dst_dir: absolute path to parent directory for copied video
    remove_old: delete original video after copy is made

    """

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    video_dir_contents = src_dir.rglob('*')
    for file_or_dir in video_dir_contents:
        src = src_dir.joinpath(file_or_dir.name)
        if not src.is_file():
            # don't copy subdirectories in video directory
            continue
        elif src.suffix in ['mp4', 'avi']:
            copy_and_reformat_video_file(src, dst_dir, remove_old)


@typechecked
def get_frames_from_idxs(video_file: str | Path, idxs: np.ndarray) -> np.ndarray:
    """Load frames from specific indices into memory.

    Parameters
    ----------
    video_file: absolute path to mp4
    idxs: frame indices into video

    Returns
    -------
    frames array of shape (n_frames, n_channels, ypix, xpix)

    """
    cap = cv2.VideoCapture(video_file)
    try:
        is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
        n_frames = len(idxs)
        for fr, i in enumerate(idxs):
            if fr == 0 or not is_contiguous:
                cap.set(1, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if fr == 0:
                    height, width, _ = frame_rgb.shape
                    frames = np.zeros((n_frames, 3, height, width), dtype='uint8')
                frames[fr] = frame_rgb.transpose(2, 0, 1)
            else:
                _logger.debug(
                    'warning! reached end of video; returning blank frames for remainder of '
                    'requested indices'
                )
                break
    finally:
        cap.release()
    return frames


@typechecked
def compute_video_motion_energy(
    video_file: str | Path,
    resize_dims: int = 32,
    return_frames: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Compute the absolute pixel difference in consecutive downsampled frames.

    Paramters
    ---------
    video_file: absolute path to mp4
    resize_dims: number of pixels (in both dimensions) to downsample video before computing motion
        energy

    Returns
    -------
    motion energy array of shape (n_frames,)

    """

    # read all frames, reshape
    frames = read_nth_frames(
        video_file=video_file,
        n=1,
        resize_dims=resize_dims,
    )
    frame_count = frames.shape[0]
    batches = np.reshape(frames, (frame_count, -1))

    # take temporal diffs
    me = np.diff(batches, axis=0, prepend=0)

    # take absolute values and sum over all pixels to get motion energy
    me = np.sum(np.abs(me), axis=1)

    if return_frames:
        return me, batches
    else:
        return me


@typechecked
def read_nth_frames(
    video_file: str | Path,
    n: int = 1,
    resize_dims: int = 64,
) -> np.ndarray:
    """Read every nth frame from a video file and return results in a numpy array.

    video_file: absolute path to mp4
    n: number of frames to advance after successfully loading a frame
    resize_dims: number of pixels (in both dimensions) to downsample video before computing motion
        energy

    Returns
    -------
    frames array of shape (n_frames, n_channels, ypix, xpix)

    """

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        _logger.error(f'Error opening video file {video_file}')

    frames = []
    frame_counter = 0
    frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    with tqdm(total=int(frame_total)) as pbar:
        while cap.isOpened():
            # Read the next frame
            ret, frame = cap.read()
            if ret:
                # If the frame was successfully read, then process it
                if frame_counter % n == 0:
                    frame_resize = cv2.resize(frame, (resize_dims, resize_dims))
                    frame_rgb = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb.astype(np.float16))
                frame_counter += 1
                pbar.update(1)
            else:
                # If we couldn't read a frame, we've probably reached the end
                break

    # When everything is done, release the video capture object
    cap.release()

    return np.array(frames)
