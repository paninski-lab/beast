"""Segmentation pipeline: GPU dispatch, worker orchestration, and failure reporting.

Runs SAM3 video tracking on each video to produce per-frame segmentation masks.
Multi-GPU dispatch: videos are distributed round-robin across available GPUs and
processed in parallel, one video per GPU at a time.
"""

import json
import logging
import os
import shutil
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp

from beast.preprocess.config_3d import Beast3DConfig
from beast.video import _get_video_files

_logger = logging.getLogger(__name__)


def _get_physical_gpu_ids() -> list[str]:
    """Return the list of physical GPU device IDs available to this process.

    Reads ``CUDA_VISIBLE_DEVICES`` when set (e.g. by SLURM or a job scheduler);
    otherwise queries torch for the device count.

    Returns
    -------
    list of physical GPU ID strings, e.g. ['0', '1'] or ['2', '5']
    """
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cvd:
        return [d.strip() for d in cvd.split(',') if d.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def _segment_worker(
    physical_gpu_id: str,
    video_list: list[str],
    output_dirs: list[str],
    text_prompt: str,
    num_objects: int | None,
    threshold: float,
    clip_size: int,
    failed_list: list | None = None,
) -> list[dict]:
    """Process a list of videos on a single GPU.

    Sets ``CUDA_VISIBLE_DEVICES`` to the assigned physical GPU before any CUDA
    calls so that the SAM3 accelerator targets the correct device. Incomplete
    output directories from a previous interrupted run are cleaned automatically.

    Parameters
    ----------
    physical_gpu_id: physical CUDA device ID string to pin this worker to
    video_list: paths to video files to segment
    output_dirs: output directory paths corresponding to each video
    text_prompt: text prompt passed to SAM3 for object detection
    num_objects: number of objects to track; None lets SAM3 decide
    threshold: SAM3 confidence threshold
    clip_size: clip length (frames) passed to SAM3
    failed_list: optional shared list for multi-process failure collection

    Returns
    -------
    list of failure dicts; each dict has keys video, video_path, gpu_id,
    error, error_type, traceback
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = physical_gpu_id
    # force offline mode so workers use the pre-cached models
    os.environ['HF_HUB_OFFLINE'] = '1'
    # suppress noisy httpx probe requests for optional HF files that don't exist for SAM3
    logging.getLogger('httpx').setLevel(logging.WARNING)

    try:
        from beast.preprocess.segment.sam3 import process_video
    except ImportError as exc:
        raise ImportError(
            'SAM3 segmentation requires the sam3 optional dependencies. '
            'Install them with: pip install beast[sam3]'
        ) from exc

    _logger.info(f'[GPU {physical_gpu_id}] starting worker with {len(video_list)} videos')
    local_failures = []

    for i, (video_path, output_dir) in enumerate(zip(video_list, output_dirs, strict=True)):
        video_name = Path(video_path).name
        _logger.info(
            f'[GPU {physical_gpu_id}] [{i + 1}/{len(video_list)}] segmenting: {video_name}'
        )
        try:
            complete_marker = Path(output_dir) / '_COMPLETE'
            if Path(output_dir).exists() and not complete_marker.exists():
                _logger.info(
                    f'[GPU {physical_gpu_id}] cleaning incomplete output: {output_dir}'
                )
                shutil.rmtree(output_dir)

            process_video(
                video_path=video_path,
                output_dir=output_dir,
                text_prompt=text_prompt,
                num_object=num_objects,
                threshold=threshold,
                clip_size=clip_size,
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            complete_marker.touch()

        except Exception as exc:
            tb = traceback.format_exc()
            _logger.error(f'[GPU {physical_gpu_id}] error on {video_name}: {exc}')
            failure = {
                'video': video_name,
                'video_path': video_path,
                'gpu_id': physical_gpu_id,
                'error': str(exc),
                'error_type': type(exc).__name__,
                'traceback': tb,
            }
            local_failures.append(failure)
            if failed_list is not None:
                failed_list.append(failure)

    succeeded = len(video_list) - len(local_failures)
    _logger.info(
        f'[GPU {physical_gpu_id}] worker finished: '
        f'{succeeded}/{len(video_list)} succeeded, {len(local_failures)} failed'
    )
    return local_failures


def run_segmentation(
    videos_dir: Path,
    cfg: Beast3DConfig,
) -> None:
    """Run SAM3 segmentation on all videos in videos_dir.

    Videos already segmented (marked with a ``_COMPLETE`` sentinel file) are
    skipped. When multiple GPUs are available, videos are distributed
    round-robin and processed in parallel (one video per GPU at a time).
    Failure details are written to ``output_dir/segmentation_masks/failed_videos.json``.

    Parameters
    ----------
    videos_dir: directory containing videos to segment
    cfg: beast3d config
    """
    seg_dir = Path(cfg.output_dir) / 'segmentation_masks'
    seg_dir.mkdir(parents=True, exist_ok=True)

    video_files = _get_video_files(videos_dir, cfg.video.extensions)
    if not video_files:
        _logger.warning(f'no videos found in {videos_dir}, skipping segmentation')
        return

    pending_videos = []
    pending_outputs = []
    for video_file in video_files:
        output_path = seg_dir / video_file.stem
        if (output_path / '_COMPLETE').exists():
            _logger.info(f'skipping (complete): {video_file.name}')
            continue
        pending_videos.append(str(video_file))
        pending_outputs.append(str(output_path))

    if not pending_videos:
        _logger.info(f'all {len(video_files)} videos already segmented')
        return

    physical_gpus = _get_physical_gpu_ids()
    num_gpus = len(physical_gpus)

    _logger.info(f'running SAM3 segmentation on {len(pending_videos)} videos')
    _logger.info(f'  text prompt: "{cfg.segmentation.text_prompt}"')
    _logger.info(f'  num objects: {cfg.segmentation.num_objects}')
    _logger.info(f'  available GPUs: {num_gpus} (physical IDs: {physical_gpus})')

    logging.getLogger('httpx').setLevel(logging.WARNING)
    from beast.preprocess.segment.sam3 import _precache_sam3_models
    _precache_sam3_models()

    worker_kwargs = dict(
        text_prompt=cfg.segmentation.text_prompt,
        num_objects=cfg.segmentation.num_objects,
        threshold=cfg.segmentation.threshold,
        clip_size=cfg.segmentation.clip_size,
    )

    all_failures: list[dict] = []

    if num_gpus <= 1:
        phys_id = physical_gpus[0] if physical_gpus else '0'
        all_failures = _segment_worker(
            physical_gpu_id=phys_id,
            video_list=pending_videos,
            output_dirs=pending_outputs,
            **worker_kwargs,
        )
    else:
        per_gpu_videos: list[list[str]] = [[] for _ in range(num_gpus)]
        per_gpu_outputs: list[list[str]] = [[] for _ in range(num_gpus)]
        for idx, (video, output) in enumerate(zip(pending_videos, pending_outputs, strict=True)):
            per_gpu_videos[idx % num_gpus].append(video)
            per_gpu_outputs[idx % num_gpus].append(output)

        for i, phys_id in enumerate(physical_gpus):
            _logger.info(f'  GPU {phys_id}: assigned {len(per_gpu_videos[i])} videos')

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        shared_failures = manager.list()

        processes = []
        for i, phys_id in enumerate(physical_gpus):
            if not per_gpu_videos[i]:
                continue
            p = mp.Process(
                target=_segment_worker,
                args=(
                    phys_id,
                    per_gpu_videos[i],
                    per_gpu_outputs[i],
                    cfg.segmentation.text_prompt,
                    cfg.segmentation.num_objects,
                    cfg.segmentation.threshold,
                    cfg.segmentation.clip_size,
                    shared_failures,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        all_failures = list(shared_failures)

    num_succeeded = len(pending_videos) - len(all_failures)
    _logger.info(f'segmentation complete: {num_succeeded}/{len(pending_videos)} succeeded')
    _logger.info(f'segmentation masks saved to {seg_dir}')

    failures_file = seg_dir / 'failed_videos.json'
    if all_failures:
        for f in all_failures:
            _logger.error(f'  [{f["error_type"]}] {f["video"]} (GPU {f["gpu_id"]}): {f["error"]}')
        failures_json = [
            {k: v for k, v in f.items() if k != 'traceback'}
            for f in all_failures
        ]
        with open(failures_file, 'w') as fp:
            json.dump(failures_json, fp, indent=2)
        _logger.info(f'failed video details saved to {failures_file}')
    elif failures_file.exists():
        failures_file.unlink()
