from pathlib import Path
from typing import Callable

import pytest

from beast.data.augmentations import expand_imgaug_str_to_dict, imgaug_pipeline
from beast.data.datamodules import BaseDataModule
from beast.data.datasets import BaseDataset

ROOT = Path(__file__).parent.parent


@pytest.fixture
def data_dir() -> Path:
    return ROOT.joinpath('tests/testing_data/data_dir')


@pytest.fixture
def video_file() -> Path:
    return ROOT.joinpath('tests/testing_data/test_vid.mp4')


@pytest.fixture
def config_ae_path() -> Path:
    return ROOT.joinpath('configs/resnet_ae.yaml')


@pytest.fixture
def config_ae(config_ae_path) -> dict:
    from beast.io import load_config
    return load_config(config_ae_path)


@pytest.fixture
def aug_pipeline() -> Callable:
    params_dict = expand_imgaug_str_to_dict('default')
    pipeline = imgaug_pipeline(params_dict)
    return pipeline


@pytest.fixture
def base_dataset(data_dir, aug_pipeline) -> BaseDataset:
    dataset = BaseDataset(data_dir=data_dir, imgaug_pipeline=aug_pipeline)
    return dataset


@pytest.fixture
def base_datamodule(base_dataset) -> BaseDataModule:
    datamodule = BaseDataModule(
        dataset=base_dataset,
        train_batch_size=10,
        val_batch_size=10,
        test_batch_size=10,
        train_probability=0.8,
        val_probability=0.1,
        test_probability=0.1,
    )
    return datamodule
