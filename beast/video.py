"""Video utilities: codec checking, re-encoding, frame extraction, motion energy, and ffmpeg."""

import logging
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_logger = logging.getLogger(__name__)


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
    if not input_file.is_file():
        raise FileNotFoundError(f'{input_file} does not exist')
    # check directory for saving outputs exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = (
        f'ffmpeg -i {str(input_file)} '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p '
        f'-c:a copy -y {str(output_file)}'
    )
    subprocess.run(ffmpeg_cmd, shell=True)


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
    dst_dir = Path(dst_dir)
    dst = dst_dir.joinpath(src.stem + '.mp4')

    # check 0: do we even need to reformat?
    if dst.is_file():
        return dst

    # check 1: does file exist?
    if not src.is_file():
        _logger.warning(f'{src} does not exist, skipping')
        return None

    # check 2: is file in the correct format for DALI?
    video_file_correct_codec = check_codec_format(src)

    # reencode/rename
    if not video_file_correct_codec:
        _logger.info(f're-encoding {src} to be compatible with DALI video reader')
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
        elif src.suffix in ['.mp4', '.avi']:
            copy_and_reformat_video_file(src, dst_dir, remove_old)


def get_frames_from_idxs(
    video_file: str | Path | None,
    idxs: np.ndarray,
    cap: cv2.VideoCapture | None = None,
) -> np.ndarray:
    """Load frames from specific indices into memory.

    Parameters
    ----------
    video_file: absolute path to mp4
    idxs: frame indices into video
    cap: already-created video capture object

    Returns
    -------
    frames array of shape (n_frames, n_channels, ypix, xpix)

    """
    should_release = False
    if cap is None:
        if video_file is None:
            raise ValueError('video_file must be provided when cap is None')
        cap = cv2.VideoCapture(video_file)
        should_release = True

    try:
        is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
        n_frames = len(idxs)
        for fr, idx in enumerate(idxs):
            if fr == 0 or not is_contiguous:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if fr == 0:
                    height, width, _ = frame_rgb.shape
                    frames = np.zeros((n_frames, 3, height, width), dtype='uint8')
                frames[fr] = frame_rgb.transpose(2, 0, 1)
            else:
                _logger.warning(
                    'reached end of video; returning blank frames for remainder of '
                    'requested indices'
                )
                break
    finally:
        if should_release:
            cap.release()

    return frames


def compute_video_motion_energy(
    video_file: str | Path,
    resize_dims: int = 32,
    return_frames: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the absolute pixel difference in consecutive downsampled frames.

    Parameters
    ----------
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


def read_nth_frames(
    video_file: str | Path,
    n: int = 1,
    resize_dims: int = 64,
) -> np.ndarray:
    """Read every nth frame from a video file and return results in a numpy array.

    Parameters
    ----------
    video_file: absolute path to mp4
    n: number of frames to advance after successfully loading a frame
    resize_dims: number of pixels (in both dimensions) to downsample video before computing
        motion energy

    Returns
    -------
    frames array of shape (n_frames, n_channels, ypix, xpix)

    """

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        raise OSError(f'Error opening video file {video_file}')

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


def get_video_stats(video_path: str | Path) -> dict:
    """Return basic statistics for a single video file.

    Parameters
    ----------
    video_path: path to the video file

    Returns
    -------
    dict with keys: fps, width, height, total_frames, duration_sec, codec

    Raises
    ------
    OSError: if the video file cannot be opened
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'Could not open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    return {
        'fps': round(fps, 2),
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'duration_sec': round(duration_sec, 2),
        'codec': codec.strip(),
    }


def merge_videos(video_paths: list[str], output_path: str, fps: int) -> None:
    """Concatenate multiple video files into one using ffmpeg.

    Parameters
    ----------
    video_paths: ordered list of video file paths
    output_path: destination file path
    fps: output frame rate

    Raises
    ------
    subprocess.CalledProcessError: if ffmpeg exits with a non-zero code
    """
    if not video_paths:
        return
    if len(video_paths) == 1:
        shutil.copyfile(video_paths[0], output_path)
        return
    file_list_path = Path(output_path).parent / 'video_list.txt'
    with open(file_list_path, 'w') as f:
        for vp in video_paths:
            f.write(f"file '{Path(vp).absolute()}'\n")
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(file_list_path), '-c', 'copy', str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'ffmpeg merge failed:\n{result.stderr}')
        _logger.info(f'merged {len(video_paths)} clips into {output_path}')
    finally:
        if file_list_path.exists():
            file_list_path.unlink()


def _get_video_files(videos_dir: Path, extensions: list[str]) -> list[Path]:
    """Return a sorted list of video files in a directory.

    Parameters
    ----------
    videos_dir: directory to search (non-recursive)
    extensions: file extensions to include, without leading dot (e.g. ['mp4', 'avi'])

    Returns
    -------
    sorted list of matching paths
    """
    files: list[Path] = []
    for ext in extensions:
        files.extend(videos_dir.glob(f'*.{ext}'))
    return sorted(files)


def discover_videos(
    videos_dir: Path,
    pattern: str,
    extensions: list[str],
) -> dict[str, dict[str, Path]]:
    """Discover video files and group them by session_id and cam_id.

    The regex pattern must contain two named capture groups:
    ``(?P<session>...)`` and ``(?P<cam>...)``.

    Parameters
    ----------
    videos_dir: directory containing video files
    pattern: regex pattern to parse filenames into session and cam components
    extensions: file extensions to include (e.g. ['mp4', 'avi'])

    Returns
    -------
    mapping of session_id -> cam_id -> resolved Path
    """
    compiled = re.compile(pattern)
    video_dict: dict[str, dict[str, Path]] = {}
    for video_file in _get_video_files(videos_dir, extensions):
        match = compiled.match(video_file.name)
        if not match:
            _logger.warning(f'skipping unmatched file: {video_file.name}')
            continue
        session_id = match.group('session')
        cam_id = match.group('cam')
        video_dict.setdefault(session_id, {})[cam_id] = video_file.resolve()
    return video_dict


def trim_video(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    threads: int | None = None,
) -> None:
    """Trim a video to an inclusive frame range using ffmpeg.

    Uses the decoder frame counter rather than timestamp seeking, so the
    output is frame-index-accurate even for VFR sources.

    Parameters
    ----------
    input_path: source video file
    output_path: destination file; parent directory is created if needed
    start_frame: first frame to keep (inclusive, zero-indexed)
    end_frame: last frame to keep (inclusive)
    threads: number of ffmpeg threads; None lets ffmpeg decide

    Raises
    ------
    RuntimeError: if ffmpeg returns a non-zero exit code
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f'trim=start_frame={start_frame}:end_frame={end_frame + 1},setpts=PTS-STARTPTS'
    cmd = ['ffmpeg', '-y']
    if threads is not None:
        cmd += ['-threads', str(threads)]
    cmd += [
        '-i', str(input_path),
        '-vf', vf,
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        '-an',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg cut failed for {input_path}:\n{result.stderr}')


def downsample_video(
    input_path: Path,
    output_path: Path,
    target_fps: float | None,
    max_frames: int | None = None,
    threads: int | None = None,
    phase_offset_frames: int = 0,
) -> None:
    """Downsample a video to a target FPS using ffmpeg.

    If target_fps is None, fps resampling is skipped and only max_frames
    is applied. If max_frames is set, only the first N output frames are kept.

    When phase_offset_frames > 0, frames are selected starting at source-frame
    index K with stride S = round(source_fps / target_fps): {K, K+S, K+2S, ...}.
    K must satisfy 1 <= K < S to stay disjoint from the K=0 training set.

    Parameters
    ----------
    input_path: source video file
    output_path: destination file; parent directory is created if needed
    target_fps: output frame rate; None skips fps filtering
    max_frames: cap on output frame count; None keeps all frames
    threads: number of ffmpeg threads; None lets ffmpeg decide
    phase_offset_frames: source-frame offset K for phase-shifted eval set construction

    Raises
    ------
    RuntimeError: if ffmpeg fails or if phase_offset_frames is invalid
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y']
    if threads is not None:
        cmd += ['-threads', str(threads)]
    cmd += ['-i', str(input_path)]
    if target_fps is not None:
        if phase_offset_frames > 0:
            src_fps = get_video_stats(input_path)['fps']
            stride = int(round(src_fps / target_fps))
            if stride < 2:
                raise RuntimeError(
                    f'{input_path.name}: source_fps={src_fps:.3f} too close to '
                    f'target_fps={target_fps} (stride={stride}); '
                    f'phase-shift requires stride >= 2'
                )
            if phase_offset_frames >= stride:
                raise RuntimeError(
                    f'{input_path.name}: phase_offset_frames={phase_offset_frames} '
                    f'must be < stride={stride}; '
                    f'otherwise frames overlap the K=0 training set'
                )
            k = phase_offset_frames
            cmd += ['-vf', f'trim=start_frame={k},setpts=PTS-STARTPTS,fps={target_fps}']
        else:
            cmd += ['-vf', f'fps={target_fps}']
    cmd += ['-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-an']
    if max_frames is not None:
        cmd += ['-frames:v', str(max_frames)]
    cmd.append(str(output_path))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg downsample failed for {input_path}:\n{result.stderr}')
