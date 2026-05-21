from pathlib import Path

import pytest
import torch

from beast.data.datasets import BaseDataset


class TestBaseDataset:

    def test_rgb(self, base_dataset):

        # check stored object properties
        assert base_dataset.data_dir.is_dir()
        assert len(base_dataset.image_list) > 0

        # check batch properties
        idx = 3
        example = base_dataset[idx]
        assert example['image'].shape == (3, 224, 224)
        assert isinstance(example['image'], torch.Tensor)
        assert example['video'] is not None
        assert example['idx'] == idx
        assert isinstance(example['image_path'], str)
        assert Path(example['image_path']).is_file()

    def test_grayscale(self, data_dir):
        from beast.data.datasets import _IMAGENET_MEAN, _IMAGENET_STD
        dataset = BaseDataset(data_dir=data_dir, imgaug_pipeline=None, num_channels=1)
        example = dataset[0]
        assert isinstance(example, dict)
        image = example['image']
        assert image.shape == (3, 224, 224)
        # undo per-channel ImageNet normalization before comparing channels
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        denorm = image * std + mean
        assert torch.allclose(denorm[0], denorm[1], atol=1e-5)
        assert torch.allclose(denorm[1], denorm[2], atol=1e-5)

    def test_invalid_num_channels(self, data_dir):
        with pytest.raises(ValueError, match='num_channels must be 1 or 3'):
            BaseDataset(data_dir=data_dir, imgaug_pipeline=None, num_channels=2)
