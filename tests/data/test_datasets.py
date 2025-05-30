from pathlib import Path

import torch


def test_base_dataset(base_dataset):
    
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
