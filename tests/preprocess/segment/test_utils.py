"""Tests for beast/preprocess/segment/utils.py."""

from pathlib import Path

import numpy as np
import torch

from beast.preprocess.segment.utils import (
    extract_bbox_from_mask,
    visualize_detected_objects,
    visualize_tracks,
)


def _make_frame(h: int = 20, w: int = 20) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _rect_mask(
    h: int = 20, w: int = 20,
    r1: int = 4, c1: int = 6, r2: int = 14, c2: int = 16,
    dtype: type = np.float32,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=dtype)
    mask[r1:r2, c1:c2] = 255 if dtype == np.uint8 else 1.0
    return mask


# ---------------------------------------------------------------------------
# TestExtractBboxFromMask
# ---------------------------------------------------------------------------

class TestExtractBboxFromMask:
    """Test the extract_bbox_from_mask function."""

    def test_none_input_returns_none(self) -> None:
        assert extract_bbox_from_mask(None) is None  # type: ignore[arg-type]

    def test_empty_array_returns_none(self) -> None:
        assert extract_bbox_from_mask(np.array([])) is None

    def test_all_zeros_float_returns_none(self) -> None:
        assert extract_bbox_from_mask(np.zeros((10, 10), dtype=np.float32)) is None

    def test_all_zeros_uint8_returns_none(self) -> None:
        assert extract_bbox_from_mask(np.zeros((10, 10), dtype=np.uint8)) is None

    def test_float_unit_mask_returns_xyxy(self) -> None:
        # mask[4:14, 6:16] = 1.0 → rows 4-13, cols 6-15
        mask = _rect_mask(20, 20, r1=4, c1=6, r2=14, c2=16)
        result = extract_bbox_from_mask(mask)
        assert result is not None
        assert result[0] == 6   # x1
        assert result[1] == 4   # y1
        assert result[2] == 15  # x2 (inclusive last col)
        assert result[3] == 13  # y2 (inclusive last row)

    def test_float_255_scaled_mask_returns_bbox(self) -> None:
        # values > 1.0 are divided by 255 before thresholding
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:8, 3:9] = 200.0
        result = extract_bbox_from_mask(mask)
        assert result is not None
        assert result[0] == 3   # x1
        assert result[1] == 2   # y1

    def test_uint8_255_mask_returns_bbox(self) -> None:
        mask = _rect_mask(10, 10, r1=1, c1=2, r2=5, c2=8, dtype=np.uint8)
        result = extract_bbox_from_mask(mask)
        assert result is not None
        assert result[0] == 2   # x1
        assert result[1] == 1   # y1

    def test_uint8_binary_mask_returns_bbox(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:5, 2:8] = 1
        result = extract_bbox_from_mask(mask)
        assert result is not None
        assert result[0] == 2

    def test_single_column_mask_returns_none(self) -> None:
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:, 5] = 1.0   # x1 == x2 == 5 → degenerate
        assert extract_bbox_from_mask(mask) is None

    def test_single_row_mask_returns_none(self) -> None:
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[5, :] = 1.0   # y1 == y2 == 5 → degenerate
        assert extract_bbox_from_mask(mask) is None

    def test_return_dtype_is_int32(self) -> None:
        result = extract_bbox_from_mask(_rect_mask())
        assert result is not None
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# TestVisualizeDetectedObjects
# ---------------------------------------------------------------------------

class TestVisualizeDetectedObjects:
    """Test the visualize_detected_objects function."""

    def _boxes(self, *boxes: list) -> list:
        return [list(boxes)]

    def test_returns_uint8_bgr_frame_same_shape(self) -> None:
        frame = _make_frame(30, 40)
        result = visualize_detected_objects(frame, self._boxes([5, 5, 20, 20]), [1])
        assert result.shape == (30, 40, 3)
        assert result.dtype == np.uint8

    def test_saves_file_when_output_path_given(self, tmp_path: Path) -> None:
        out = tmp_path / 'vis.jpg'
        frame = _make_frame()
        visualize_detected_objects(frame, self._boxes([2, 2, 10, 10]), [1], output_path=out)
        assert out.exists()

    def test_no_output_path_does_not_write_file(self, tmp_path: Path) -> None:
        visualize_detected_objects(_make_frame(), self._boxes([2, 2, 10, 10]), [1])
        assert list(tmp_path.iterdir()) == []

    def test_more_ids_than_boxes_skips_extras(self) -> None:
        # 3 ids but only 1 box: extras silently skipped
        result = visualize_detected_objects(_make_frame(), self._boxes([2, 2, 10, 10]), [1, 2, 3])
        assert result.shape == _make_frame().shape

    def test_empty_obj_ids_returns_plain_bgr_frame(self) -> None:
        frame = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = visualize_detected_objects(frame, [[]], [])
        assert result.shape == (20, 20, 3)


# ---------------------------------------------------------------------------
# TestVisualizeTracks
# ---------------------------------------------------------------------------

class TestVisualizeTracks:
    """Test the visualize_tracks function."""

    def _colors(self, track_id: int = 1) -> dict:
        return {track_id: np.array([255, 0, 0], dtype=np.uint8)}

    def test_returns_frame_same_shape(self) -> None:
        result = visualize_tracks(_make_frame(30, 40), {}, {})
        assert result.shape == (30, 40, 3)

    def test_empty_tracked_objects_returns_copy(self) -> None:
        frame = np.full((20, 20, 3), 42, dtype=np.uint8)
        result = visualize_tracks(frame, {}, {})
        np.testing.assert_array_equal(result, frame)

    def test_object_with_2d_float_mask_applied(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[5:15, 5:15] = 1.0
        tracked = {1: {'mask': mask, 'box': None, 'score': 0.9}}
        result = visualize_tracks(frame, tracked, self._colors())
        assert result.shape == (20, 20, 3)

    def test_object_with_3d_mask_squeezed(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.ones((1, 20, 20), dtype=np.uint8)
        tracked = {1: {'mask': mask, 'box': None, 'score': 0.0}}
        result = visualize_tracks(frame, tracked, self._colors())
        assert result.shape == (20, 20, 3)

    def test_mask_resized_when_shape_mismatch(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8)   # smaller than frame
        tracked = {1: {'mask': mask, 'box': None, 'score': 0.5}}
        result = visualize_tracks(frame, tracked, self._colors())
        assert result.shape == (20, 20, 3)

    def test_object_with_box_drawn(self) -> None:
        frame = np.zeros((30, 30, 3), dtype=np.uint8)
        box = np.array([2, 2, 20, 20], dtype=np.float32)
        tracked = {1: {'mask': None, 'box': box, 'score': 1.0}}
        result = visualize_tracks(frame, tracked, self._colors())
        assert result.shape == (30, 30, 3)

    def test_torch_tensor_mask_converted(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = torch.ones(20, 20, dtype=torch.float32)
        tracked = {1: {'mask': mask, 'box': None, 'score': 0.5}}
        result = visualize_tracks(frame, tracked, self._colors())
        assert result.shape == (20, 20, 3)

    def test_torch_tensor_score_extracted(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        score = torch.tensor(0.75)
        tracked = {1: {'mask': None, 'box': None, 'score': score}}
        result = visualize_tracks(frame, tracked, self._colors())
        assert result.shape == (20, 20, 3)

    def test_no_box_uses_top_left_label_position(self) -> None:
        # object has no mask and no box — label falls back to (10, label_h+10)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        tracked = {1: {'mask': None, 'box': None, 'score': 0.5}}
        result = visualize_tracks(frame, tracked, self._colors())
        assert result.shape == (50, 50, 3)
