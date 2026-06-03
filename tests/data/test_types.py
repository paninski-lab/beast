"""Tests for beast.data.types."""

import torch

from beast.data.types import ExampleDict, MultiViewExampleDict


class TestExampleDict:
    """Test the ExampleDict TypedDict."""

    def test_expected_keys(self) -> None:
        assert set(ExampleDict.__annotations__) == {'image', 'video', 'idx', 'image_path'}

    def test_valid_single_item(self) -> None:
        item: ExampleDict = {
            'image': torch.zeros(3, 224, 224),
            'video': 'video.mp4',
            'idx': 0,
            'image_path': 'img000.png',
        }
        assert item['idx'] == 0
        assert item['image'].shape == (3, 224, 224)

    def test_valid_batched_item(self) -> None:
        item: ExampleDict = {
            'image': torch.zeros(3, 224, 224),
            'video': ['video.mp4', 'video.mp4'],
            'idx': [0, 1],
            'image_path': ['img000.png', 'img001.png'],
        }
        assert isinstance(item['video'], list)
        assert isinstance(item['idx'], list)


class TestMultiViewExampleDict:
    """Test the MultiViewExampleDict TypedDict."""

    def test_required_keys(self) -> None:
        annotations = MultiViewExampleDict.__annotations__
        assert {'image', 'view_names', 'video_id', 'frame_id'}.issubset(set(annotations))

    def test_optional_keys(self) -> None:
        # c2w, fxfycxcy, input_mask must be present as annotations (even if NotRequired)
        annotations = MultiViewExampleDict.__annotations__
        assert 'c2w' in annotations
        assert 'fxfycxcy' in annotations
        assert 'input_mask' in annotations

    def test_valid_minimal_item(self) -> None:
        # Arrange / Act — only required fields
        item: MultiViewExampleDict = {
            'image': torch.zeros(3, 3, 64, 64),
            'view_names': ['camera1', 'camera2', 'camera3'],
            'video_id': 's1-d1',
            'frame_id': 'img00000001.png',
        }
        # Assert
        assert item['image'].shape == (3, 3, 64, 64)
        assert len(item['view_names']) == 3

    def test_valid_full_item(self) -> None:
        # Arrange / Act — all fields including optional ones
        item: MultiViewExampleDict = {
            'image': torch.zeros(3, 3, 64, 64),
            'view_names': ['camera1', 'camera2', 'camera3'],
            'video_id': 's1-d1',
            'frame_id': 'img00000001.png',
            'c2w': torch.eye(4).unsqueeze(0).expand(3, -1, -1),
            'fxfycxcy': torch.ones(3, 4),
            'input_mask': torch.zeros(3, 1, 64, 64),
        }
        # Assert
        assert item['c2w'].shape == (3, 4, 4)
        assert item['fxfycxcy'].shape == (3, 4)
        assert item['input_mask'].shape == (3, 1, 64, 64)
