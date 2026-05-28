"""BEAST3D dataset creation pipeline: video stats, trim, downsample, assemble, and resize."""

import csv
import json
import logging
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from beast.io import get_camera_params_for_view, load_bbox_csv, load_calibration
from beast.preprocess.config_3d import Beast3DConfig
from beast.preprocess.extraction import export_frames, select_frame_idxs_kmeans
from beast.video import (
    _get_video_files,
    discover_videos,
    downsample_video,
    get_video_stats,
    trim_video,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def _get_trim_videos_dir(cfg: Beast3DConfig) -> Path:
    return Path(cfg.output_dir) / f'{cfg.video_subdir}_trim'


def resolve_videos_dir(cfg: Beast3DConfig) -> Path:
    """Return the videos directory to use for downstream steps.

    Priority order: downsample output > trim output > raw input.

    Parameters
    ----------
    cfg: beast3d config

    Returns
    -------
    path to the resolved videos directory

    """
    if cfg.downsample.enabled:
        ds_dir = Path(cfg.output_dir) / cfg.video_subdir
        if ds_dir.is_dir():
            return ds_dir
    if cfg.cut.enabled:
        trim_dir = _get_trim_videos_dir(cfg)
        if trim_dir.is_dir():
            return trim_dir
    return Path(cfg.input_dir) / cfg.video_subdir


# ---------------------------------------------------------------------------
# stats step
# ---------------------------------------------------------------------------

def run_video_stats(cfg: Beast3DConfig) -> dict:
    """Scan input videos and write per-video stats to CSV and summary to JSON.

    Writes ``video_stats.csv`` and ``video_stats.json`` into ``cfg.output_dir``.

    Parameters
    ----------
    cfg: beast3d config

    Returns
    -------
    summary dict with aggregate statistics including fps_avg, total_frames, etc.

    """
    videos_dir = Path(cfg.input_dir) / cfg.video_subdir
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = _get_video_files(videos_dir, cfg.video.extensions)
    if not video_files:
        _logger.warning(f'no videos found in {videos_dir}')
        return {}

    rows = []
    all_fps = []
    all_resolutions: set = set()
    total_frames = 0
    total_duration = 0.0

    for vf in video_files:
        stats = get_video_stats(vf)
        rows.append({'filename': vf.name, **stats})
        all_fps.append(stats['fps'])
        all_resolutions.add((stats['width'], stats['height']))
        total_frames += stats['total_frames']
        total_duration += stats['duration_sec']

    avg_fps = sum(all_fps) / len(all_fps) if all_fps else 0.0
    avg_frames = total_frames / len(rows) if rows else 0.0

    _logger.info(f'video stats for {videos_dir}')
    _logger.info(f'  videos: {len(rows)}')
    _logger.info(f'  resolutions: {", ".join(f"{w}x{h}" for w, h in sorted(all_resolutions))}')
    _logger.info(f'  fps: min={min(all_fps):.1f} max={max(all_fps):.1f} avg={avg_fps:.1f}')
    _logger.info(f'  total frames: {total_frames:,}  avg per video: {avg_frames:,.0f}')
    _logger.info(f'  total duration: {total_duration:,.1f}s ({total_duration / 3600:.2f}h)')

    # write per-video CSV
    csv_path = output_dir / 'video_stats.csv'
    fieldnames = ['filename', 'fps', 'width', 'height', 'total_frames', 'duration_sec', 'codec']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    _logger.info(f'per-video stats written to {csv_path}')

    summary = {
        'videos_dir': str(videos_dir),
        'num_videos': len(rows),
        'resolutions': [f'{w}x{h}' for w, h in sorted(all_resolutions)],
        'fps_min': round(min(all_fps), 2) if all_fps else 0.0,
        'fps_max': round(max(all_fps), 2) if all_fps else 0.0,
        'fps_avg': round(avg_fps, 2),
        'total_frames': total_frames,
        'total_duration_sec': round(total_duration, 2),
        'avg_frames_per_video': round(avg_frames, 1),
    }
    json_path = output_dir / 'video_stats.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    _logger.info(f'summary stats written to {json_path}')

    return summary


# ---------------------------------------------------------------------------
# trim step
# ---------------------------------------------------------------------------

def _resolve_frame_range(cfg: Beast3DConfig, video_path: Path) -> tuple[int, int]:
    """Resolve the inclusive [start_frame, end_frame] cut range for one video.

    Frame bounds take precedence over second bounds. Defaults: 0 and total_frames-1.

    Parameters
    ----------
    cfg: beast3d config
    video_path: path to the video file

    Returns
    -------
    (start_frame, end_frame) inclusive frame indices

    Raises
    ------
    ValueError: if the resolved range is invalid (start > end)

    """
    cut = cfg.cut
    _stats: dict | None = None

    def _probe() -> dict:
        nonlocal _stats
        if _stats is None:
            _stats = get_video_stats(video_path)
        return _stats

    if cut.start_frame is not None:
        start_frame = int(cut.start_frame)
    elif cut.start_sec is not None:
        start_frame = int(round(cut.start_sec * _probe()['fps']))
    else:
        start_frame = 0

    if cut.end_frame is not None:
        end_frame = int(cut.end_frame)
    elif cut.end_sec is not None:
        end_frame = int(round(cut.end_sec * _probe()['fps'])) - 1
    else:
        end_frame = _probe()['total_frames'] - 1

    start_frame = max(0, start_frame)
    if end_frame < start_frame:
        raise ValueError(
            f'invalid trim range for {video_path.name}: '
            f'start_frame={start_frame} end_frame={end_frame}'
        )
    return start_frame, end_frame


def _trim_bbox_csv(
    input_csv: Path,
    output_csv: Path,
    start_frame: int,
    end_frame: int,
) -> int:
    """Filter a bbox CSV to [start_frame, end_frame] and re-index first column to 0.

    Parameters
    ----------
    input_csv: source bbox CSV file
    output_csv: destination file; parent directory is created if needed
    start_frame: first frame to keep (inclusive)
    end_frame: last frame to keep (inclusive)

    Returns
    -------
    number of rows written

    """
    with open(input_csv, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows or not fieldnames:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return 0

    key_col = fieldnames[0]
    filtered = [r for r in rows if start_frame <= int(r[key_col]) <= end_frame]
    for new_idx, row in enumerate(filtered):
        row[key_col] = str(new_idx)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    return len(filtered)


def _trim_one_video(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    threads: int | None,
) -> tuple[Path, int, int, bool]:
    """Worker: trim one video. Returns (input_path, start_frame, end_frame, was_skipped)."""
    if output_path.exists():
        return input_path, start_frame, end_frame, True
    trim_video(input_path, output_path, start_frame, end_frame, threads=threads)
    return input_path, start_frame, end_frame, False


def run_trim(cfg: Beast3DConfig) -> Path:
    """Trim all input videos to the configured frame range.

    Videos are trimmed in parallel; bbox CSVs are re-indexed sequentially.
    Writes trimmed videos (and optionally trimmed bbox CSVs) to
    ``output_dir/<video_subdir>_trim/``.

    Parameters
    ----------
    cfg: beast3d config with cut.enabled=True

    Returns
    -------
    path to the directory containing trimmed videos

    Raises
    ------
    FileNotFoundError: if no video files are found in the input directory

    """
    input_videos_dir = Path(cfg.input_dir) / cfg.video_subdir
    output_videos_dir = _get_trim_videos_dir(cfg)
    output_videos_dir.mkdir(parents=True, exist_ok=True)

    video_files = _get_video_files(input_videos_dir, cfg.video.extensions)
    if not video_files:
        raise FileNotFoundError(
            f'no video files found in {input_videos_dir} '
            f'with extensions {cfg.video.extensions}'
        )

    max_workers = max(1, int(cfg.cut.max_workers))
    if cfg.cut.ffmpeg_threads is not None:
        ffmpeg_threads = int(cfg.cut.ffmpeg_threads)
    else:
        cpu_count = os.cpu_count() or max_workers
        ffmpeg_threads = max(1, math.ceil(cpu_count / max_workers))

    _logger.info(
        f'trimming {len(video_files)} videos '
        f'(workers={max_workers}, ffmpeg_threads={ffmpeg_threads})'
    )

    jobs = []
    for video_file in video_files:
        start_frame, end_frame = _resolve_frame_range(cfg, video_file)
        jobs.append((video_file, output_videos_dir / video_file.name, start_frame, end_frame))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_trim_one_video, vf, out, s, e, ffmpeg_threads): (vf, s, e)
            for vf, out, s, e in jobs
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            vf, s, e = futures[fut]
            try:
                _, _, _, skipped = fut.result()
            except Exception as exc:
                raise RuntimeError(f'trim failed for {vf.name}: {exc}') from exc
            tag = 'skip (exists)' if skipped else f'[{s}..{e}] ({e - s + 1} frames)'
            _logger.info(f'  [{done}/{len(jobs)}] {vf.name} {tag}')

    if cfg.has_bboxes and cfg.bbox_csv_pattern:
        import re
        name_pattern = re.compile(cfg.video.filename_pattern)
        for video_file, _, start_frame, end_frame in jobs:
            m = name_pattern.match(video_file.name)
            if not m:
                continue
            session_id = m.group('session')
            cam_id = m.group('cam')
            bbox_name = cfg.bbox_csv_pattern.format(session_id=session_id, cam_id=cam_id)
            src_csv = input_videos_dir / bbox_name
            dst_csv = output_videos_dir / bbox_name
            if not src_csv.exists():
                continue
            if dst_csv.exists():
                _logger.info(f'  bbox csv exists: {dst_csv.name}')
                continue
            n_rows = _trim_bbox_csv(src_csv, dst_csv, start_frame, end_frame)
            _logger.info(f'  wrote {dst_csv.name} ({n_rows} rows, re-indexed from 0)')

    _logger.info(f'trimmed videos saved to {output_videos_dir}')
    return output_videos_dir


# ---------------------------------------------------------------------------
# downsample step
# ---------------------------------------------------------------------------

def _downsample_one(
    input_path: Path,
    output_path: Path,
    target_fps: float | None,
    max_frames: int | None,
    threads: int | None,
    phase_offset_frames: int,
) -> tuple[Path, bool]:
    """Worker: downsample one video. Returns (input_path, was_skipped)."""
    if output_path.exists():
        return input_path, True
    downsample_video(
        input_path,
        output_path,
        target_fps,
        max_frames=max_frames,
        threads=threads,
        phase_offset_frames=phase_offset_frames,
    )
    return input_path, False


def run_downsample(cfg: Beast3DConfig) -> Path:
    """Downsample all videos to the configured target FPS.

    Reads from the trim output directory if cut was enabled, otherwise from the
    raw input directory. Writes downsampled videos to ``output_dir/<video_subdir>/``.

    Parameters
    ----------
    cfg: beast3d config with downsample.enabled=True

    Returns
    -------
    path to the directory containing downsampled videos

    Raises
    ------
    FileNotFoundError: if no video files are found in the source directory
    ValueError: if phase_offset_frames is negative

    """
    input_videos_dir = _get_trim_videos_dir(cfg) if cfg.cut.enabled else (
        Path(cfg.input_dir) / cfg.video_subdir
    )
    output_videos_dir = Path(cfg.output_dir) / cfg.video_subdir
    output_videos_dir.mkdir(parents=True, exist_ok=True)

    video_files = _get_video_files(input_videos_dir, cfg.video.extensions)
    if not video_files:
        raise FileNotFoundError(
            f'no video files found in {input_videos_dir} '
            f'with extensions {cfg.video.extensions}'
        )

    phase_offset = int(cfg.downsample.phase_offset_frames)
    if phase_offset < 0:
        raise ValueError(
            f'downsample.phase_offset_frames must be >= 0, got {phase_offset}'
        )

    max_workers = max(1, int(cfg.downsample.max_workers))
    if cfg.downsample.ffmpeg_threads is not None:
        ffmpeg_threads = int(cfg.downsample.ffmpeg_threads)
    else:
        cpu_count = os.cpu_count() or max_workers
        ffmpeg_threads = max(1, math.ceil(cpu_count / max_workers))

    phase_tag = f', phase_offset_frames={phase_offset}' if phase_offset > 0 else ''
    if cfg.downsample.target_fps is None:
        _logger.info(
            f'trimming {len(video_files)} videos to max_frames={cfg.downsample.max_frames} '
            f'(workers={max_workers}, ffmpeg_threads={ffmpeg_threads})'
        )
    else:
        _logger.info(
            f'downsampling {len(video_files)} videos to {cfg.downsample.target_fps} fps '
            f'(workers={max_workers}, ffmpeg_threads={ffmpeg_threads}{phase_tag})'
        )

    jobs = [(vf, output_videos_dir / vf.name) for vf in video_files]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _downsample_one,
                inp, out,
                cfg.downsample.target_fps,
                cfg.downsample.max_frames,
                ffmpeg_threads,
                phase_offset,
            ): inp
            for inp, out in jobs
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            inp = futures[fut]
            try:
                _, skipped = fut.result()
            except Exception as exc:
                raise RuntimeError(f'downsample failed for {inp.name}: {exc}') from exc
            tag = 'skip (exists)' if skipped else 'done'
            _logger.info(f'  [{done}/{len(jobs)}] {inp.name} {tag}')

    _logger.info(f'downsampled videos saved to {output_videos_dir}')
    return output_videos_dir


# ---------------------------------------------------------------------------
# assemble step
# ---------------------------------------------------------------------------

def _save_camera_params_for_frames(
    cam_params: dict,
    bbox_dict: dict | None,
    frame_idxs: np.ndarray,
    output_dir: Path,
    n_digits: int = 8,
    original_frame_idxs: np.ndarray | None = None,
) -> None:
    """Save per-frame camera parameters as .npy files alongside frame images.

    Parameters
    ----------
    cam_params: camera parameter dict with keys intrinsics, extrinsics, distortions,
        width, height
    bbox_dict: optional frame-index → bbox dict from load_bbox_csv
    frame_idxs: frame indices used for output filename numbering
    output_dir: directory to write .npy files into
    n_digits: zero-padding width for filename indices
    original_frame_idxs: frame indices in the original (pre-downsample) video space for
        bbox lookup; defaults to frame_idxs when None

    """
    if original_frame_idxs is None:
        original_frame_idxs = frame_idxs
    base = {
        'intrinsics': cam_params['intrinsics'],
        'extrinsics': cam_params['extrinsics'],
        'distortions': cam_params['distortions'],
        'width': cam_params['width'],
        'height': cam_params['height'],
    }
    for frame_idx, orig_idx in zip(frame_idxs, original_frame_idxs, strict=True):
        data = base.copy()
        if bbox_dict is not None and orig_idx in bbox_dict:
            data['bbox'] = bbox_dict[orig_idx]
        np.save(str(output_dir / f'img{frame_idx:0{n_digits}d}.npy'), data)  # type: ignore[arg-type]


def _copy_masks_for_frames(
    mask_source_dir: Path,
    frame_idxs: np.ndarray,
    output_dir: Path,
    n_digits: int = 8,
) -> int:
    """Copy segmentation masks for selected frames.

    Parameters
    ----------
    mask_source_dir: directory containing mask PNG files named mask{idx}.png
    frame_idxs: frame indices to copy masks for
    output_dir: destination directory
    n_digits: zero-padding width for filename indices

    Returns
    -------
    number of masks successfully copied

    """
    copied = 0
    for frame_idx in frame_idxs:
        src = mask_source_dir / f'mask{frame_idx:0{n_digits}d}.png'
        if src.exists():
            shutil.copy(src, output_dir / src.name)
            copied += 1
    return copied


def _process_view_task(task: dict) -> dict:
    """Worker: export frames, save camera params, and copy masks for one view.

    Parameters
    ----------
    task: self-contained dict with keys: session_id, cam_id, video_path,
        view_output_dir, mask_source, frame_idxs, original_frame_idxs, cam_params,
        bbox_dict, n_digits, extension

    Returns
    -------
    result dict with keys: session_id, cam_id, n_frames, n_masks

    """
    view_output_dir: Path = task['view_output_dir']
    view_output_dir.mkdir(parents=True, exist_ok=True)

    frame_idxs = task['frame_idxs']
    n_digits = task['n_digits']

    export_frames(
        task['video_path'],
        view_output_dir,
        frame_idxs,
        extension=task['extension'],
        n_digits=n_digits,
        context_frames=0,
    )

    if task['cam_params'] is not None:
        _save_camera_params_for_frames(
            task['cam_params'],
            task['bbox_dict'],
            frame_idxs,
            view_output_dir,
            n_digits,
            original_frame_idxs=task['original_frame_idxs'],
        )

    n_masks = 0
    mask_source: Path = task['mask_source']
    if mask_source.is_dir():
        n_masks = _copy_masks_for_frames(mask_source, frame_idxs, view_output_dir, n_digits)

    return {
        'session_id': task['session_id'],
        'cam_id': task['cam_id'],
        'n_frames': len(frame_idxs),
        'n_masks': n_masks,
    }


def assemble_dataset(cfg: Beast3DConfig) -> Path:
    """Assemble the final BEAST3D dataset from videos, calibrations, and masks.

    Steps:
    1. Discover videos grouped by session/cam.
    2. For each session, select frames via kmeans on the anchor view.
    3. Fan out per-view tasks (frame export, camera params, mask copy) to a process pool.
    4. Write selected_frames.csv per session and info.json for the dataset.

    Parameters
    ----------
    cfg: beast3d config

    Returns
    -------
    path to the assembled dataset directory

    Raises
    ------
    ValueError: if no videos are found or the anchor view is missing
    FileNotFoundError: if video_stats.json is required for bbox mapping but missing

    """
    videos_dir = resolve_videos_dir(cfg)
    dataset_dir = Path(cfg.output_dir) / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    video_dict = discover_videos(videos_dir, cfg.video.filename_pattern, cfg.video.extensions)
    if not video_dict:
        raise ValueError(f'no videos found in {videos_dir}')

    first_session = next(iter(video_dict.values()))
    avail_views = sorted(first_session.keys())
    _logger.info(f'found {len(video_dict)} sessions with views: {avail_views}')

    if cfg.anchor_view not in avail_views:
        raise ValueError(
            f'anchor view "{cfg.anchor_view}" not found in available views: {avail_views}'
        )

    seg_masks_dir = Path(cfg.output_dir) / 'segmentation_masks'
    session_ids = sorted(video_dict.keys())

    # read original fps from stats JSON if needed for bbox frame-index mapping
    orig_fps: float | None = None
    if cfg.downsample.enabled and cfg.downsample.target_fps is not None and cfg.has_bboxes:
        stats_json = Path(cfg.output_dir) / 'video_stats.json'
        if not stats_json.exists():
            raise FileNotFoundError(
                f'video_stats.json not found at {stats_json}; '
                'run the stats step before assembling when bbox mapping is required'
            )
        with open(stats_json) as f:
            orig_fps = json.load(f)['fps_avg']

    # phase 1 (sequential): per-session frame selection and task construction
    tasks: list[dict] = []
    processed_sessions: list[str] = []
    total_frames = 0

    for i, session_id in enumerate(session_ids):
        views = video_dict[session_id]
        anchor_path = views.get(cfg.anchor_view)
        if anchor_path is None:
            _logger.warning(
                f'session {session_id} missing anchor view {cfg.anchor_view}, skipping'
            )
            continue

        _logger.info(f'[{i + 1}/{len(session_ids)}] session: {session_id}')

        try:
            frame_idxs = select_frame_idxs_kmeans(
                video_file=anchor_path,
                resize_dims=cfg.frame.kmeans_resize,
                n_frames_to_select=cfg.frame.frames_per_video,
            )
            frame_idxs = np.sort(frame_idxs)
        except ValueError as exc:
            if 'valid video segment too short' in str(exc):
                cap = cv2.VideoCapture(str(anchor_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                sample_size = min(cfg.frame.frames_per_video, total)
                frame_idxs = np.sort(np.random.choice(total, size=sample_size, replace=False))
                _logger.warning(
                    f'  video too short for kmeans, randomly sampled {sample_size} frames'
                )
            else:
                raise

        if cfg.assemble.downsample_selected_frames and cfg.assemble.downsample_factor > 1:
            factor = cfg.assemble.downsample_factor
            frame_idxs = frame_idxs[::factor]
            _logger.info(
                f'  downsampled selected frames by factor {factor} → {len(frame_idxs)} frames'
            )

        _logger.info(f'  selected {len(frame_idxs)} frames')

        # map downsampled frame indices back to original video frame space for bbox lookup
        if orig_fps is not None:
            assert cfg.downsample.target_fps is not None
            original_frame_idxs = np.round(
                frame_idxs * orig_fps / cfg.downsample.target_fps
            ).astype(int)
        else:
            original_frame_idxs = frame_idxs

        calibration = load_calibration(session_id, cfg)

        for cam_id, video_path in sorted(views.items()):
            cam_params = None
            bbox_dict = None
            if calibration is not None:
                cam_params = get_camera_params_for_view(calibration, cam_id, cfg)
                if cam_params is not None and cfg.has_bboxes:
                    bbox_dict = load_bbox_csv(session_id, cam_id, cfg)

            tasks.append({
                'session_id': session_id,
                'cam_id': cam_id,
                'video_path': video_path,
                'view_output_dir': dataset_dir / session_id / cam_id,
                'mask_source': seg_masks_dir / video_path.stem / 'masks',
                'frame_idxs': frame_idxs,
                'original_frame_idxs': original_frame_idxs,
                'cam_params': cam_params,
                'bbox_dict': bbox_dict,
                'n_digits': cfg.frame.n_digits,
                'extension': cfg.frame.extension,
            })

        frames_to_label = np.array([
            f'img{idx:0{cfg.frame.n_digits}d}.{cfg.frame.extension}'
            for idx in frame_idxs
        ])
        (dataset_dir / session_id).mkdir(parents=True, exist_ok=True)
        np.savetxt(
            str(dataset_dir / session_id / 'selected_frames.csv'),
            frames_to_label,
            delimiter=',',
            fmt='%s',
        )
        total_frames += len(frame_idxs)
        processed_sessions.append(session_id)

    # phase 2 (parallel): fan out per-view export / cam-param / mask-copy work
    max_workers = max(1, int(cfg.assemble.max_workers))
    _logger.info(f'exporting {len(tasks)} views (workers={max_workers})')

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_view_task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            t = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                raise RuntimeError(
                    f'assemble failed for {t["session_id"]}/{t["cam_id"]}: {exc}'
                ) from exc
            mask_note = f', masks={result["n_masks"]}' if result['n_masks'] else ''
            _logger.info(
                f'  [{done}/{len(tasks)}] {result["session_id"]}/'
                f'{result["cam_id"]} frames={result["n_frames"]}{mask_note}'
            )

    session_ids = processed_sessions
    dataset_info = {
        'dataset': cfg.name,
        'description': 'BEAST3D multi-view self-supervised dataset',
        'available_views': avail_views,
        'anchor_view': cfg.anchor_view,
        'video_ids': session_ids,
        'number_of_videos': len(session_ids),
        'input_directory': str(cfg.input_dir),
        'output_directory': str(dataset_dir),
        'frames_per_video': cfg.frame.frames_per_video,
        'n_digits': cfg.frame.n_digits,
        'extension': cfg.frame.extension,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'author': cfg.author,
        'seed': cfg.seed,
        'total_ssl_frames': total_frames * len(avail_views),
    }
    with open(dataset_dir / 'info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)

    _logger.info(f'dataset assembled: {dataset_dir}')
    _logger.info(f'  sessions: {len(session_ids)}  views: {len(avail_views)}')
    _logger.info(f'  total frames: {total_frames * len(avail_views)}')

    return dataset_dir


# ---------------------------------------------------------------------------
# resize step
# ---------------------------------------------------------------------------

def _compute_new_size(h: int, w: int, size: int) -> tuple[int, int]:
    """Return (new_h, new_w) with shorter side equal to size, preserving aspect ratio.

    Parameters
    ----------
    h: original image height
    w: original image width
    size: target shorter-side size in pixels

    Returns
    -------
    (new_h, new_w) or (h, w) unchanged when already at or below target size

    """
    if min(h, w) <= size:
        return h, w
    if h < w:
        return size, int(round(w * size / h))
    return int(round(h * size / w)), size


def _scale_intrinsics(
    intrinsics: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Scale focal lengths and principal point in a camera intrinsics array.

    Parameters
    ----------
    intrinsics: (3, 3) camera matrix or 4-element [fx, fy, cx, cy] array
    scale_x: horizontal scale factor (new_w / orig_w)
    scale_y: vertical scale factor (new_h / orig_h)

    Returns
    -------
    scaled intrinsics array of the same shape

    """
    intr = np.array(intrinsics, dtype=np.float64)
    if intr.shape == (3, 3):
        intr[0, 0] *= scale_x   # fx
        intr[1, 1] *= scale_y   # fy
        intr[0, 2] *= scale_x   # cx
        intr[1, 2] *= scale_y   # cy
    elif intr.ndim == 1 and intr.shape[0] == 4:
        intr[0] *= scale_x   # fx
        intr[1] *= scale_y   # fy
        intr[2] *= scale_x   # cx
        intr[3] *= scale_y   # cy
    return intr


def resize_dataset(cfg: Beast3DConfig) -> None:
    """Resize all images and masks in the assembled dataset in-place.

    Scales the shorter side of every img*.png and mask*.png to cfg.resize.size
    while preserving aspect ratio. Updates width, height, and intrinsics in the
    matching .npy camera parameter files. Already-small images are skipped.

    Parameters
    ----------
    cfg: beast3d config with resize.enabled=True and resize.size set

    Raises
    ------
    FileNotFoundError: if the dataset directory does not exist

    """
    dataset_dir = Path(cfg.output_dir) / 'dataset'
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f'dataset directory not found: {dataset_dir}; run the assemble step first'
        )

    target_size = cfg.resize.size
    cam_dirs = sorted(p for p in dataset_dir.rglob('*') if p.is_dir() and any(p.glob('img*.png')))
    if not cam_dirs:
        _logger.info(f'no image directories found in {dataset_dir}')
        return

    _logger.info(f'resizing {len(cam_dirs)} cam directories (shorter side → {target_size}px)')
    total_imgs = 0
    total_masks = 0
    skipped = 0

    for cam_dir in tqdm(cam_dirs, desc='resizing'):
        img_files = sorted(cam_dir.glob('img*.png'))
        if not img_files:
            continue

        first = cv2.imread(str(img_files[0]))
        if first is None:
            continue
        orig_h, orig_w = first.shape[:2]
        new_h, new_w = _compute_new_size(orig_h, orig_w, target_size)

        if new_h == orig_h and new_w == orig_w:
            skipped += len(img_files)
            continue

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            cv2.imwrite(
                str(img_path),
                cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA),
            )
            total_imgs += 1

            npy_path = img_path.with_suffix('.npy')
            if npy_path.exists():
                data = np.load(str(npy_path), allow_pickle=True).item()
                if 'intrinsics' in data:
                    data['intrinsics'] = _scale_intrinsics(data['intrinsics'], scale_x, scale_y)
                data['width'] = new_w
                data['height'] = new_h
                np.save(str(npy_path), data)

        for mask_path in sorted(cam_dir.glob('mask*.png')):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            cv2.imwrite(
                str(mask_path),
                cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST),
            )
            total_masks += 1

    _logger.info(
        f'resized {total_imgs} images, {total_masks} masks '
        f'({skipped} images already at target size, skipped)'
    )
