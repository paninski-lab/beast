"""Tests for beast/preprocess/segment/sam3.py.

conftest.py in this directory mocks `accelerate` before this module is imported.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from beast.preprocess.segment.sam3 import (
    _build_frame_json,
    _precache_sam3_models,
    _save_first_mask,
    process_sam3_video_outputs,
)

_SAM3 = 'beast.preprocess.segment.sam3'


# ---------------------------------------------------------------------------
# TestPrecacheSam3Models
# ---------------------------------------------------------------------------

class TestPrecacheSam3Models:
    """Test the _precache_sam3_models function."""

    def _make_model_mocks(self):
        return {
            f'{_SAM3}.Sam3TrackerVideoModel': MagicMock(),
            f'{_SAM3}.Sam3TrackerVideoProcessor': MagicMock(),
            f'{_SAM3}.Sam3Model': MagicMock(),
            f'{_SAM3}.Sam3Processor': MagicMock(),
        }

    def test_downloads_all_four_models_on_success(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != 'HF_HUB_DOWNLOAD_TIMEOUT'}
        with patch.dict(os.environ, env, clear=True):
            with patch(f'{_SAM3}.Sam3TrackerVideoModel') as m_tracker:
                with patch(f'{_SAM3}.Sam3TrackerVideoProcessor') as m_tracker_proc:
                    with patch(f'{_SAM3}.Sam3Model') as m_model:
                        with patch(f'{_SAM3}.Sam3Processor') as m_proc:
                            _precache_sam3_models()
        m_tracker.from_pretrained.assert_called_once_with('facebook/sam3')
        m_tracker_proc.from_pretrained.assert_called_once_with('facebook/sam3')
        m_model.from_pretrained.assert_called_once_with('facebook/sam3')
        m_proc.from_pretrained.assert_called_once_with('facebook/sam3')

    def test_restores_old_timeout_env_var(self) -> None:
        with patch.dict(os.environ, {'HF_HUB_DOWNLOAD_TIMEOUT': '60'}):
            with patch(f'{_SAM3}.Sam3TrackerVideoModel'):
                with patch(f'{_SAM3}.Sam3TrackerVideoProcessor'):
                    with patch(f'{_SAM3}.Sam3Model'):
                        with patch(f'{_SAM3}.Sam3Processor'):
                            _precache_sam3_models()
            # still inside patch.dict scope — env var should be restored to '60'
            assert os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT') == '60'

    def test_removes_timeout_env_var_when_not_originally_set(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != 'HF_HUB_DOWNLOAD_TIMEOUT'}
        with patch.dict(os.environ, env, clear=True):
            with patch(f'{_SAM3}.Sam3TrackerVideoModel'):
                with patch(f'{_SAM3}.Sam3TrackerVideoProcessor'):
                    with patch(f'{_SAM3}.Sam3Model'):
                        with patch(f'{_SAM3}.Sam3Processor'):
                            _precache_sam3_models()
        assert 'HF_HUB_DOWNLOAD_TIMEOUT' not in os.environ

    def test_download_failure_tries_local_cache(self) -> None:
        err = RuntimeError('network error')
        # side_effect on the class mock only fires on instantiation; set it on from_pretrained
        mock_tracker = MagicMock()
        mock_tracker.from_pretrained.side_effect = err
        env = {k: v for k, v in os.environ.items() if k != 'HF_HUB_DOWNLOAD_TIMEOUT'}
        with patch.dict(os.environ, env, clear=True):
            with patch(f'{_SAM3}.Sam3TrackerVideoModel', mock_tracker):
                with patch(f'{_SAM3}.Sam3TrackerVideoProcessor'):
                    with patch(f'{_SAM3}.Sam3Model') as m_model:
                        with patch(f'{_SAM3}.Sam3Processor') as m_proc:
                            _precache_sam3_models()  # should not raise
        m_model.from_pretrained.assert_called_with('facebook/sam3', local_files_only=True)
        m_proc.from_pretrained.assert_called_with('facebook/sam3', local_files_only=True)

    def test_both_download_and_local_cache_fail_does_not_raise(self) -> None:
        err = RuntimeError('network error')
        env = {k: v for k, v in os.environ.items() if k != 'HF_HUB_DOWNLOAD_TIMEOUT'}
        with patch.dict(os.environ, env, clear=True):
            with patch(f'{_SAM3}.Sam3TrackerVideoModel', side_effect=err):
                with patch(f'{_SAM3}.Sam3TrackerVideoProcessor'):
                    with patch(f'{_SAM3}.Sam3Model', side_effect=err):
                        with patch(f'{_SAM3}.Sam3Processor', side_effect=err):
                            _precache_sam3_models()  # warning only, no exception


# ---------------------------------------------------------------------------
# TestProcessSam3VideoOutputs
# ---------------------------------------------------------------------------

class TestProcessSam3VideoOutputs:
    """Test the process_sam3_video_outputs function."""

    def _float_mask(self, h: int = 10, w: int = 10) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        mask[2:8, 2:8] = 1.0
        return mask

    def test_empty_outputs_returns_empty_dict(self) -> None:
        assert process_sam3_video_outputs({}, frame_idx=0) == {}

    def test_new_format_returns_tracked_object(self) -> None:
        mask = self._float_mask()
        outputs = {'masks': {1: mask}, 'scores': {1: 0.9}, 'object_ids': [1]}
        result = process_sam3_video_outputs(outputs, frame_idx=0)
        assert 1 in result
        assert result[1]['score'] == pytest.approx(0.9)
        assert result[1]['mask'] is mask

    def test_new_format_selected_object_ids_filters(self) -> None:
        mask = self._float_mask()
        outputs = {'masks': {1: mask, 2: mask}, 'object_ids': [1, 2]}
        result = process_sam3_video_outputs(outputs, frame_idx=0, selected_object_ids=[2])
        assert 1 not in result
        assert 2 in result

    def test_new_format_bfloat16_tensor_mask_converted(self) -> None:
        mask = torch.ones(10, 10, dtype=torch.bfloat16)
        outputs = {'masks': {1: mask}, 'object_ids': [1]}
        result = process_sam3_video_outputs(outputs, frame_idx=0)
        assert 1 in result
        assert isinstance(result[1]['mask'], np.ndarray)

    def test_new_format_tensor_score_extracted(self) -> None:
        mask = self._float_mask()
        score_tensor = torch.tensor(0.75)
        outputs = {'masks': {1: mask}, 'scores': {1: score_tensor}, 'object_ids': [1]}
        result = process_sam3_video_outputs(outputs, frame_idx=0)
        assert result[1]['score'] == pytest.approx(0.75)

    def test_new_format_object_ids_defaults_to_mask_keys(self) -> None:
        mask = self._float_mask()
        outputs = {'masks': {3: mask}}  # no explicit object_ids
        result = process_sam3_video_outputs(outputs, frame_idx=0)
        assert 3 in result

    def test_legacy_format_returns_tracked_objects(self) -> None:
        masks = np.zeros((1, 10, 10), dtype=np.float32)
        masks[0, 2:8, 2:8] = 1.0
        boxes = np.array([[2, 2, 7, 7]], dtype=np.float32)
        outputs = {
            'object_ids': np.array([1]),
            'masks': masks,
            'scores': np.array([0.8]),
            'boxes': boxes,
        }
        result = process_sam3_video_outputs(outputs, frame_idx=0)
        assert 1 in result
        assert result[1]['score'] == pytest.approx(0.8)

    def test_legacy_format_selected_object_ids_filters(self) -> None:
        masks = np.zeros((2, 10, 10), dtype=np.float32)
        outputs = {
            'object_ids': np.array([1, 2]),
            'masks': masks,
            'scores': np.array([0.9, 0.5]),
            'boxes': None,
        }
        result = process_sam3_video_outputs(outputs, frame_idx=0, selected_object_ids=[1])
        assert 1 in result
        assert 2 not in result


# ---------------------------------------------------------------------------
# TestBuildFrameJson
# ---------------------------------------------------------------------------

class TestBuildFrameJson:
    """Test the _build_frame_json function."""

    def test_empty_tracked_objects(self) -> None:
        result = _build_frame_json(42, {})
        assert result['frame_idx'] == 42
        assert result['num_objects'] == 0
        assert result['track_ids'] == []

    def test_single_object_numpy_box_converted_to_list(self) -> None:
        box = np.array([1, 2, 10, 20], dtype=np.int32)
        tracked = {1: {'score': 0.9, 'box': box}}
        result = _build_frame_json(0, tracked)
        assert result['bounding_boxes'][1] == [1, 2, 10, 20]
        assert isinstance(result['bounding_boxes'][1], list)

    def test_none_box_remains_none(self) -> None:
        tracked = {1: {'score': 0.5, 'box': None}}
        result = _build_frame_json(0, tracked)
        assert result['bounding_boxes'][1] is None

    def test_all_track_ids_present(self) -> None:
        tracked = {
            1: {'score': 0.9, 'box': None},
            2: {'score': 0.7, 'box': None},
        }
        result = _build_frame_json(5, tracked)
        assert set(result['track_ids']) == {1, 2}
        assert result['num_objects'] == 2
        assert result['frame_idx'] == 5


# ---------------------------------------------------------------------------
# TestSaveFirstMask
# ---------------------------------------------------------------------------

class TestSaveFirstMask:
    """Test the _save_first_mask function."""

    def test_no_mask_writes_no_file(self, tmp_path: Path) -> None:
        masks_dir = tmp_path / 'masks'
        masks_dir.mkdir()
        _save_first_mask({1: {'mask': None}}, masks_dir, orig_frame_idx=0)
        assert list(masks_dir.iterdir()) == []

    def test_float_unit_mask_saved_as_uint8(self, tmp_path: Path) -> None:
        masks_dir = tmp_path / 'masks'
        masks_dir.mkdir()
        mask = np.full((10, 10), 1.0, dtype=np.float32)
        _save_first_mask({1: {'mask': mask}}, masks_dir, orig_frame_idx=7)
        saved = masks_dir / 'mask00000007.png'
        assert saved.exists()

    def test_float_255_mask_cast_to_uint8(self, tmp_path: Path) -> None:
        masks_dir = tmp_path / 'masks'
        masks_dir.mkdir()
        mask = np.full((10, 10), 200.0, dtype=np.float32)
        _save_first_mask({1: {'mask': mask}}, masks_dir, orig_frame_idx=3)
        assert (masks_dir / 'mask00000003.png').exists()

    def test_3d_mask_squeezed_before_save(self, tmp_path: Path) -> None:
        masks_dir = tmp_path / 'masks'
        masks_dir.mkdir()
        mask = np.ones((1, 10, 10), dtype=np.float32)
        _save_first_mask({1: {'mask': mask}}, masks_dir, orig_frame_idx=1)
        assert (masks_dir / 'mask00000001.png').exists()

    def test_only_first_object_mask_saved(self, tmp_path: Path) -> None:
        masks_dir = tmp_path / 'masks'
        masks_dir.mkdir()
        mask = np.ones((10, 10), dtype=np.float32)
        _save_first_mask(
            {1: {'mask': mask}, 2: {'mask': mask}},
            masks_dir,
            orig_frame_idx=0,
        )
        assert len(list(masks_dir.iterdir())) == 1

    def test_bfloat16_tensor_mask_converted(self, tmp_path: Path) -> None:
        masks_dir = tmp_path / 'masks'
        masks_dir.mkdir()
        mask = torch.ones(10, 10, dtype=torch.bfloat16)
        _save_first_mask({1: {'mask': mask}}, masks_dir, orig_frame_idx=2)
        assert (masks_dir / 'mask00000002.png').exists()
