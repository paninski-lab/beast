"""Tests for beast.data.types."""

import torch

from beast.data.types import ExampleDict


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
