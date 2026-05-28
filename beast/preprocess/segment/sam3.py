"""SAM3 video segmentation and tracking for BEAST3D dataset creation.

Wraps the SAM3 transformer-based video tracker to produce per-frame binary
segmentation masks. Objects are detected via text prompt on the first clip,
then propagated through the rest of the video in clip-sized chunks.
"""

import gc
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from matplotlib.colors import hsv_to_rgb
from PIL import Image
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)

from beast.preprocess.segment.utils import (
    extract_bbox_from_mask,
    visualize_detected_objects,
    visualize_tracks,
)
from beast.video import get_frames_from_idxs, get_video_stats, merge_videos

_logger = logging.getLogger(__name__)


def _precache_sam3_models() -> None:
    """Pre-download SAM3 models to the HF cache before spawning workers.

    Avoids N workers racing to download simultaneously, which can cause
    timeouts. Falls back gracefully if already cached locally.
    """
    old_timeout = os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '')
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
    try:
        _logger.info('pre-caching SAM3 models...')
        Sam3TrackerVideoModel.from_pretrained('facebook/sam3')
        Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3')
        Sam3Model.from_pretrained('facebook/sam3')
        Sam3Processor.from_pretrained('facebook/sam3')
        _logger.info('SAM3 models cached')
    except Exception as exc:
        _logger.warning(f'pre-cache download failed ({type(exc).__name__}: {exc})')
        try:
            Sam3Model.from_pretrained('facebook/sam3', local_files_only=True)
            Sam3Processor.from_pretrained('facebook/sam3', local_files_only=True)
            _logger.info('SAM3 models found in local cache, proceeding')
        except Exception:
            _logger.warning(
                'SAM3 models not in local cache; workers will attempt download individually'
            )
    finally:
        if old_timeout:
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = old_timeout
        else:
            os.environ.pop('HF_HUB_DOWNLOAD_TIMEOUT', None)


def detect_objects_with_text_prompt(
    frame: np.ndarray,
    text_prompt: str,
    device: torch.device,
    num_object: int | None = None,
    threshold: float = 0.5,
) -> tuple[
    list[list[list[list[int]]]],
    list[list[list[int]]],
    list[int],
    list[list[list[float]]],
]:
    """Detect objects in a frame using SAM3 text-prompt segmentation.

    Parameters
    ----------
    frame: (H, W, 3) RGB frame
    text_prompt: text query for object detection
    device: torch device to run inference on
    num_object: maximum detections to keep; None keeps all
    threshold: confidence threshold for filtering detections

    Returns
    -------
    tuple of (input_points, input_labels, obj_ids, input_boxes)
    """
    frame_pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
    _logger.info('loading SAM3 model for text-based detection')
    sam3_model = Sam3Model.from_pretrained('facebook/sam3').to(device, dtype=torch.bfloat16)
    sam3_processor = Sam3Processor.from_pretrained('facebook/sam3')
    _logger.info(f'detecting objects with text prompt: "{text_prompt}"')
    inputs = sam3_processor(images=frame_pil, text=text_prompt, return_tensors='pt')
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            processed_inputs[k] = (
                v.to(device, dtype=torch.bfloat16) if v.dtype.is_floating_point
                else v.to(device)
            )
        else:
            processed_inputs[k] = v
    with torch.no_grad():
        outputs = sam3_model(**processed_inputs)
    results = sam3_processor.post_process_instance_segmentation(
        outputs=outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=[frame_pil.size[::-1]],
    )[0]
    boxes = results['boxes'].cpu().float().numpy()
    scores = results['scores'].cpu().float().numpy()
    _logger.info(f'found {len(boxes)} objects')
    if num_object is not None and len(boxes) > num_object:
        top_indices = np.argsort(scores)[::-1][:num_object]
        boxes = boxes[top_indices]
        _logger.info(f'selected top {num_object} objects by score')
    input_points = []
    input_labels = []
    obj_ids = []
    box_coords_list = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        input_points.append([[[int((x1 + x2) / 2), int((y1 + y2) / 2)]]])
        input_labels.append([[1]])
        obj_ids.append(i + 1)
        box_coords_list.append([float(x1), float(y1), float(x2), float(y2)])
    input_boxes = [box_coords_list]
    del sam3_model, sam3_processor, inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return input_points, input_labels, obj_ids, input_boxes


def process_sam3_video_outputs(
    outputs: dict,
    frame_idx: int,
    selected_object_ids: list[int] | None = None,
) -> dict[int, dict]:
    """Extract tracked objects from SAM3 video predictor outputs.

    Parameters
    ----------
    outputs: SAM3 video predictor output dict
    frame_idx: current frame index (used for context only)
    selected_object_ids: object IDs to include; None includes all

    Returns
    -------
    dict mapping track_id to dict with keys mask, box, score
    """
    tracked_objects: dict[int, dict] = {}
    if not outputs:
        return tracked_objects
    if 'masks' in outputs and isinstance(outputs['masks'], dict):
        masks_dict = outputs['masks']
        scores_dict = outputs.get('scores', {})
        object_ids = outputs.get('object_ids', list(masks_dict.keys()))
        if selected_object_ids is not None:
            object_ids = [oid for oid in object_ids if oid in selected_object_ids]
        for object_id in object_ids:
            if object_id not in masks_dict:
                continue
            mask = masks_dict[object_id]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().float().numpy() if mask.dtype == torch.bfloat16 \
                    else mask.cpu().numpy()
            mask_for_bbox = mask.copy()
            if len(mask_for_bbox.shape) > 2:
                mask_for_bbox = mask_for_bbox.squeeze()
                if len(mask_for_bbox.shape) > 2:
                    mask_for_bbox = mask_for_bbox[0]
            if mask_for_bbox.max() > 1.0:
                mask_for_bbox = mask_for_bbox / 255.0
            box = extract_bbox_from_mask(mask_for_bbox)
            score = 1.0
            if object_id in scores_dict:
                s = scores_dict[object_id]
                score = (
                    s.cpu().float().item() if isinstance(s, torch.Tensor) and
                    s.dtype == torch.bfloat16
                    else s.cpu().item() if isinstance(s, torch.Tensor)
                    else float(s)
                )
            tracked_objects[int(object_id)] = {'mask': mask, 'box': box, 'score': score}
    else:
        # legacy output format
        object_ids = outputs.get('object_ids')
        if object_ids is None:
            return tracked_objects

        def _to_numpy(t: torch.Tensor | None) -> np.ndarray | None:
            if t is None:
                return None
            return (
                t.cpu().float().numpy() if isinstance(t, torch.Tensor) and
                t.dtype == torch.bfloat16
                else t.cpu().numpy() if isinstance(t, torch.Tensor)
                else t
            )

        object_ids = _to_numpy(object_ids)
        scores = _to_numpy(outputs.get('scores'))
        boxes = _to_numpy(outputs.get('boxes'))
        masks = _to_numpy(outputs.get('masks'))
        if selected_object_ids is not None:
            sel = np.array(selected_object_ids)
            keep = np.isin(object_ids, sel)
            object_ids = object_ids[keep]
            if scores is not None:
                scores = scores[keep]
            if boxes is not None:
                boxes = boxes[keep]
            if masks is not None:
                masks = masks[keep]
        for idx in range(len(object_ids)):
            object_id = int(object_ids[idx])
            score = float(scores[idx]) if scores is not None and idx < len(scores) else 1.0
            box = boxes[idx] if boxes is not None and idx < len(boxes) else None
            mask = masks[idx] if masks is not None and idx < len(masks) else None
            if mask is not None and isinstance(mask, torch.Tensor):
                mask = _to_numpy(mask)
            if box is None and mask is not None:
                box = extract_bbox_from_mask(mask)
            tracked_objects[object_id] = {'mask': mask, 'box': box, 'score': score}
    return tracked_objects


def process_video_clip(
    clip_frames: np.ndarray,
    clip_idx: int,
    global_frame_offset: int,
    model: Sam3TrackerVideoModel,
    processor: Sam3TrackerVideoProcessor,
    device: torch.device,
    output_path: Path,
    video_path: str,
    text_prompt: str | None = None,
    num_object: int | None = None,
    threshold: float = 0.5,
    prev_clip_last_frame_tracking: dict | None = None,
    track_colors: dict[int, np.ndarray] | None = None,
    downsample_ratio: int = 1,
    initial_boxes: list[list[float]] | None = None,
) -> tuple[dict[int, np.ndarray], dict, int]:
    """Run SAM3 tracking on a single clip and save masks and visualization.

    Parameters
    ----------
    clip_frames: (N, H, W, 3) RGB frames for this clip
    clip_idx: zero-based clip index
    global_frame_offset: frame index of clip_frames[0] in the original video
    model: SAM3 tracker model
    processor: SAM3 tracker processor
    device: torch device for inference
    output_path: root output directory for this video
    video_path: path to the original video (used for frame reading in vis)
    text_prompt: text prompt for detection on the first clip
    num_object: maximum objects to track
    threshold: detection confidence threshold
    prev_clip_last_frame_tracking: tracking state from the previous clip's last frame
    track_colors: persistent color map keyed by track_id
    downsample_ratio: temporal stride used when loading clip_frames
    initial_boxes: user-supplied [[x1, y1, x2, y2], ...] boxes for the first clip

    Returns
    -------
    tuple of (track_colors, last_frame_tracking, processed_frame_count)
    """
    _logger.info(f'processing clip {clip_idx + 1} ({len(clip_frames)} frames)')
    inference_session = processor.init_video_session(
        video=clip_frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )
    outputs_dir = output_path / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    masks_dir = output_path / 'masks'
    masks_dir.mkdir(exist_ok=True)
    clips_dir = output_path / 'clips'
    clips_dir.mkdir(exist_ok=True)
    ann_frame_idx = 0
    if clip_idx == 0 or prev_clip_last_frame_tracking is None:
        first_frame = clip_frames[0]
        if clip_idx == 0 and initial_boxes is not None:
            _logger.info(f'using {len(initial_boxes)} user-provided bounding boxes')
            obj_ids = list(range(1, len(initial_boxes) + 1))
            input_boxes = [initial_boxes]
            input_points = [
                [[[int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2)]]]
                for bx in initial_boxes
            ]
            input_labels = [[[1]]] * len(initial_boxes)
        else:
            input_points, input_labels, obj_ids, input_boxes = detect_objects_with_text_prompt(
                frame=first_frame,
                text_prompt=text_prompt,
                device=device,
                num_object=num_object,
                threshold=threshold,
            )
            if not input_points:
                raise ValueError(f'no objects detected with text prompt: "{text_prompt}"')
        if clip_idx == 0:
            visualize_detected_objects(
                frame=first_frame,
                input_boxes=input_boxes,
                obj_ids=obj_ids,
                output_path=output_path / 'detection_visualization.jpg',
            )
    else:
        obj_ids = list(prev_clip_last_frame_tracking['track_ids'])
        bboxes = prev_clip_last_frame_tracking['bounding_boxes']
        input_boxes = [[bboxes[tid] for tid in obj_ids if bboxes[tid] is not None]]
        valid = [i for i, box in enumerate(input_boxes[0]) if box is not None]
        obj_ids = [obj_ids[i] for i in valid]
        input_boxes = [[input_boxes[0][i] for i in valid]]
        if not obj_ids:
            raise ValueError('no valid tracking results from previous clip')
        _logger.info(f'clip {clip_idx + 1}: initializing from {len(obj_ids)} previous objects')
    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=obj_ids,
        input_boxes=input_boxes,
    )
    if track_colors is None:
        track_colors = {}
    for track_id in inference_session.obj_ids:
        if track_id not in track_colors:
            hue = (track_id * 0.618) % 1.0
            color = hsv_to_rgb([hue, 0.8, 0.9])
            track_colors[track_id] = np.array([int(c * 255) for c in color])
    outputs = model(inference_session=inference_session, frame_idx=ann_frame_idx)
    video_res_masks = processor.post_process_masks(
        [outputs.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
    )[0]
    cap_info = cv2.VideoCapture(video_path)
    fps_orig = int(cap_info.get(cv2.CAP_PROP_FPS))
    width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()
    clip_video_path = clips_dir / f'clip_{clip_idx:04d}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(clip_video_path), fourcc, fps_orig // downsample_ratio, (width, height),
    )
    if not out.isOpened():
        raise RuntimeError(f'failed to open video writer for clip {clip_idx}')
    processed_frame_count = 0
    last_frame_tracking: dict | None = None
    skip_first_frame = clip_idx > 0
    try:
        if not skip_first_frame:
            scores_dict = (
                {
                    oid: outputs.object_score_logits[i]
                    for i, oid in enumerate(inference_session.obj_ids)
                }
                if hasattr(outputs, 'object_score_logits')
                else {oid: 1.0 for oid in inference_session.obj_ids}
            )
            frame_output = {
                'frame_idx': ann_frame_idx,
                'object_ids': inference_session.obj_ids,
                'masks': {
                    oid: video_res_masks[i]
                    for i, oid in enumerate(inference_session.obj_ids)
                },
                'scores': scores_dict,
            }
            tracked_objects = process_sam3_video_outputs(
                frame_output, ann_frame_idx, inference_session.obj_ids,
            )
            orig_frame_idx = global_frame_offset + ann_frame_idx * downsample_ratio
            frame_output_json = _build_frame_json(orig_frame_idx, tracked_objects)
            with open(outputs_dir / f'frame_{orig_frame_idx:06d}.json', 'w') as f:
                json.dump(frame_output_json, f, indent=2)
            _save_first_mask(tracked_objects, masks_dir, orig_frame_idx)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                out.write(visualize_tracks(frame, tracked_objects, track_colors))
                processed_frame_count += 1
            del frame_output, tracked_objects, video_res_masks, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for sam3_out in model.propagate_in_video_iterator(inference_session):
            frame_idx = sam3_out.frame_idx
            if skip_first_frame and frame_idx == 0:
                continue
            video_res_masks = processor.post_process_masks(
                [sam3_out.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
            )[0]
            frame_output = {
                'frame_idx': frame_idx,
                'object_ids': inference_session.obj_ids,
                'masks': {
                    oid: video_res_masks[i]
                    for i, oid in enumerate(inference_session.obj_ids)
                },
                'scores': {
                    oid: sam3_out['object_score_logits'][i]
                    for i, oid in enumerate(inference_session.obj_ids)
                },
            }
            tracked_objects = process_sam3_video_outputs(
                frame_output, frame_idx, inference_session.obj_ids,
            )
            actual_frame_idx = (frame_idx - 1) if skip_first_frame else frame_idx
            orig_frame_idx = global_frame_offset + actual_frame_idx * downsample_ratio
            frame_output_json = _build_frame_json(orig_frame_idx, tracked_objects)
            with open(outputs_dir / f'frame_{orig_frame_idx:06d}.json', 'w') as f:
                json.dump(frame_output_json, f, indent=2)
            _save_first_mask(tracked_objects, masks_dir, orig_frame_idx)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                out.write(visualize_tracks(frame, tracked_objects, track_colors))
                processed_frame_count += 1
            if frame_idx == len(clip_frames) - 1:
                last_frame_tracking = frame_output_json
            del frame_output, tracked_objects, video_res_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (frame_idx + 1) % 50 == 0:
                _logger.info(
                    f'clip {clip_idx + 1}: {frame_idx + 1}/{len(clip_frames)} frames done'
                )
    finally:
        out.release()
        del clip_frames, inference_session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    _logger.info(f'clip {clip_idx + 1} complete: {processed_frame_count} frames')
    return track_colors, last_frame_tracking, processed_frame_count


def _build_frame_json(orig_frame_idx: int, tracked_objects: dict[int, dict]) -> dict:
    return {
        'frame_idx': orig_frame_idx,
        'num_objects': len(tracked_objects),
        'track_ids': list(tracked_objects.keys()),
        'scores': {tid: float(obj['score']) for tid, obj in tracked_objects.items()},
        'bounding_boxes': {
            tid: obj['box'].tolist() if obj.get('box') is not None else None
            for tid, obj in tracked_objects.items()
        },
    }


def _save_first_mask(
    tracked_objects: dict[int, dict],
    masks_dir: Path,
    orig_frame_idx: int,
) -> None:
    """Save the first tracked object's mask as a PNG."""
    for obj_info in tracked_objects.values():
        mask = obj_info.get('mask')
        if mask is None:
            continue
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().float().numpy() if mask.dtype == torch.bfloat16 \
                else mask.cpu().numpy()
        if mask.max() > 1.0:
            mask = mask.astype(np.uint8)
        else:
            mask = (mask * 255).astype(np.uint8)
        if len(mask.shape) > 2:
            mask = mask.squeeze()
            if len(mask.shape) > 2:
                mask = mask[0]
        cv2.imwrite(str(masks_dir / f'mask{orig_frame_idx:08d}.png'), mask)
        return  # only save the first object's mask


@torch.no_grad()
def process_video(
    video_path: str,
    output_dir: str,
    text_prompt: str | None = None,
    num_object: int | None = None,
    threshold: float = 0.5,
    max_frames: int | None = None,
    downsample_ratio: int = 1,
    clip_size: int = 512,
    start_time: float | None = None,
    end_time: float | None = None,
    initial_boxes: list[list[float]] | None = None,
) -> None:
    """Run SAM3 video tracking on a video, producing per-frame segmentation masks.

    Videos are processed in clips of clip_size frames. The first clip uses text
    prompt detection; subsequent clips re-initialise from the previous clip's
    last-frame tracking state.

    Parameters
    ----------
    video_path: path to input video
    output_dir: root directory for all outputs (masks/, outputs/, clips/)
    text_prompt: text query for first-clip object detection
    num_object: maximum objects to track; None tracks all detected
    threshold: confidence threshold for text-prompt detection
    max_frames: cap on total frames to process; None processes all
    downsample_ratio: temporal stride — process every Nth source frame
    clip_size: frames per processing chunk
    start_time: start offset in seconds; None starts at the beginning
    end_time: end offset in seconds; None runs to the end
    initial_boxes: pre-defined [[x1, y1, x2, y2], ...] boxes for clip 0;
        skips text detection when provided
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_info = get_video_stats(video_path)
    total_video_frames = video_info['total_frames']
    fps = video_info['fps']
    start_frame_idx = max(0, min(int(start_time * fps), total_video_frames)) \
        if start_time is not None else 0
    end_frame_idx = max(start_frame_idx, min(int(end_time * fps), total_video_frames)) \
        if end_time is not None else total_video_frames
    if max_frames is not None:
        end_frame_idx = min(end_frame_idx, start_frame_idx + max_frames)
    actual_max_frames = end_frame_idx - start_frame_idx
    _logger.info(
        f'video: {total_video_frames} frames @ {fps:.2f} fps — '
        f'processing frames {start_frame_idx}–{end_frame_idx} ({actual_max_frames} total)'
    )
    first_clip_idx = start_frame_idx // clip_size
    last_clip_idx = (end_frame_idx - 1) // clip_size
    _logger.info(
        f'clip size: {clip_size} — clips {first_clip_idx}–{last_clip_idx} '
        f'({last_clip_idx - first_clip_idx + 1} total)'
    )
    device = Accelerator().device
    _logger.info(f'using device: {device}')
    _logger.info('loading SAM3 tracker model and processor')
    model = Sam3TrackerVideoModel.from_pretrained('facebook/sam3').to(device, dtype=torch.bfloat16)
    processor = Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3')
    track_colors: dict[int, np.ndarray] | None = None
    prev_clip_last_frame_tracking: dict | None = None
    clip_video_paths: list[str] = []
    total_processed_frames = 0
    clips_dir = output_path / 'clips'
    for clip_idx in range(first_clip_idx, last_clip_idx + 1):
        clip_start_global = clip_idx * clip_size
        clip_end_global = min(clip_start_global + clip_size, total_video_frames)
        clip_start = max(clip_start_global, start_frame_idx)
        clip_end = min(clip_end_global, end_frame_idx)
        if clip_start >= clip_end:
            continue
        _logger.info(f'loading clip {clip_idx + 1} (frames {clip_start}–{clip_end - 1})')
        clip_idxs = np.arange(clip_start, clip_end, downsample_ratio)
        clip_frames = get_frames_from_idxs(video_path, clip_idxs).transpose(0, 2, 3, 1)
        if clip_idx > first_clip_idx and prev_clip_last_frame_tracking is not None:
            prev_frame = get_frames_from_idxs(
                video_path, np.array([clip_start - 1]),
            ).transpose(0, 2, 3, 1)
            clip_frames = np.concatenate([prev_frame, clip_frames], axis=0)
        track_colors, last_frame_tracking, processed_count = process_video_clip(
            clip_frames=clip_frames,
            clip_idx=clip_idx,
            global_frame_offset=clip_start,
            model=model,
            processor=processor,
            device=device,
            output_path=output_path,
            video_path=video_path,
            text_prompt=text_prompt if clip_idx == first_clip_idx else None,
            num_object=num_object,
            threshold=threshold,
            prev_clip_last_frame_tracking=prev_clip_last_frame_tracking,
            track_colors=track_colors,
            downsample_ratio=downsample_ratio,
            initial_boxes=initial_boxes if clip_idx == first_clip_idx else None,
        )
        prev_clip_last_frame_tracking = last_frame_tracking
        total_processed_frames += processed_count
        clip_video_paths.append(str(clips_dir / f'clip_{clip_idx:04d}.mp4'))
        del clip_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    output_video_path = output_path / 'tracking_visualization.mp4'
    merge_videos(clip_video_paths, str(output_video_path), int(fps) // downsample_ratio)
    _logger.info(
        f'segmentation complete — {total_processed_frames} frames, '
        f'{len(track_colors) if track_colors else 0} objects tracked'
    )
    _logger.info(f'masks saved to {output_path / "masks"}')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
