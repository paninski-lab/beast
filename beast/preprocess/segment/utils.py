"""Visualization and mask utilities for segmentation outputs."""

from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb


def visualize_detected_objects(
    frame: np.ndarray,
    input_boxes: list[list[list[float]]],
    obj_ids: list[int],
    output_path: Path | None = None,
) -> np.ndarray:
    """Draw detection bounding boxes and labels on a frame.

    Parameters
    ----------
    frame: (H, W, 3) RGB frame
    input_boxes: boxes in 3-level nested format [[[x1, y1, x2, y2], ...]]
    obj_ids: object IDs corresponding to each box
    output_path: optional path to save the annotated image

    Returns
    -------
    annotated frame (H, W, 3) in BGR format
    """
    vis_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    all_boxes = input_boxes[0]
    for i, obj_id in enumerate(obj_ids):
        if i >= len(all_boxes):
            continue
        x1, y1, x2, y2 = (int(c) for c in all_boxes[i])
        hue = (obj_id * 0.618) % 1.0
        color = hsv_to_rgb([hue, 0.8, 0.9])
        color_bgr = tuple(int(c * 255) for c in color[::-1])
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color_bgr, 3)
        label = f'ID:{obj_id}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y1, label_size[1] + 10)
        cv2.rectangle(
            vis_frame,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0] + 10, label_y + 5),
            color_bgr,
            -1,
        )
        cv2.putText(
            vis_frame, label, (x1 + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
    if output_path is not None:
        cv2.imwrite(str(output_path), vis_frame)
    return vis_frame


def visualize_tracks(
    frame: np.ndarray,
    tracked_objects: dict[int, dict],
    track_colors: dict[int, np.ndarray],
) -> np.ndarray:
    """Overlay masks, bounding boxes, track IDs, and scores on a frame.

    Parameters
    ----------
    frame: (H, W, 3) BGR frame
    tracked_objects: dict mapping track_id to info dict with keys mask, box, score
    track_colors: dict mapping track_id to RGB color array

    Returns
    -------
    annotated frame (H, W, 3) in BGR format
    """
    vis_frame = frame.copy()
    h, w = frame.shape[:2]
    for track_id, obj_info in tracked_objects.items():
        color = track_colors[track_id]
        color_bgr = tuple(int(c) for c in color[::-1])
        mask = obj_info.get('mask')
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().float().numpy() if mask.dtype == torch.bfloat16 \
                    else mask.cpu().numpy()
            if len(mask.shape) == 3:
                mask = mask[0]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = mask.astype(np.uint8)
            overlay = vis_frame.copy()
            overlay[mask > 0] = color_bgr
            vis_frame = cv2.addWeighted(vis_frame, 0.6, overlay, 0.4, 0)
        box = obj_info.get('box')
        if box is not None:
            if isinstance(box, torch.Tensor):
                box = box.cpu().float().numpy() if box.dtype == torch.bfloat16 \
                    else box.cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color_bgr, 2)
        score = obj_info.get('score', 0.0)
        if isinstance(score, torch.Tensor):
            score = score.cpu().item()
        label = f'ID:{track_id} ({score:.2f})'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        if box is not None:
            lx, ly = x1, max(y1, label_size[1] + 10)
        else:
            lx, ly = 10, label_size[1] + 10
        cv2.rectangle(
            vis_frame,
            (lx, ly - label_size[1] - 5),
            (lx + label_size[0] + 5, ly + 5),
            color_bgr,
            -1,
        )
        cv2.putText(
            vis_frame, label, (lx + 2, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )
    return vis_frame


def extract_bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Extract a bounding box from a binary mask in xyxy format.

    Parameters
    ----------
    mask: (H, W) mask array; float in [0, 1] or uint8

    Returns
    -------
    [x1, y1, x2, y2] int32 array, or None if mask is empty
    """
    if mask is None or mask.size == 0:
        return None
    if mask.dtype != np.uint8:
        mask_binary = (mask / 255.0 if mask.max() > 1.0 else mask) > 0.5
        mask_binary = mask_binary.astype(np.uint8)
    else:
        mask_binary = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask
    if mask_binary.sum() == 0:
        return None
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    if x1 >= x2 or y1 >= y2:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.int32)
