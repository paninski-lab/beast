"""Tests for data module splitting and loading."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from beast.data.datamodules import BaseDataModule, split_sizes_from_probabilities
from beast.data.datasets import BaseDataset
from beast.data.samplers import ContrastBatchSampler


class TestBaseDataModule:
    """Test the BaseDataModule class."""

    def test_train_val_test_dataloaders(self, base_datamodule) -> None:
        train_size = base_datamodule.train_batch_size
        val_size = base_datamodule.val_batch_size
        test_size = base_datamodule.test_batch_size

        # check train batch properties
        train_dataloader = base_datamodule.train_dataloader()
        assert isinstance(train_dataloader.sampler, RandomSampler)
        batch = next(iter(train_dataloader))
        assert batch['image'].shape == (train_size, 3, 224, 224)
        # check imgaug pipeline makes non-repeatable data
        base_datamodule.train_dataset.dataset.imgaug_pipeline.seed_(0)
        b1 = base_datamodule.train_dataset[0]
        base_datamodule.train_dataset.dataset.imgaug_pipeline.seed_(1)
        b2 = base_datamodule.train_dataset[0]
        assert not np.allclose(b1['image'], b2['image'])

        # check val batch properties
        val_dataloader = base_datamodule.val_dataloader()
        batch = next(iter(val_dataloader))
        assert not isinstance(val_dataloader.sampler, RandomSampler)
        assert not isinstance(val_dataloader.sampler, ContrastBatchSampler)
        assert batch['image'].shape[1:] == (3, 224, 224)
        assert batch['image'].shape[0] <= val_size
        b1 = base_datamodule.val_dataset[0]
        b2 = base_datamodule.val_dataset[0]
        assert np.allclose(b1['image'], b2['image'], rtol=1e-3)

        test_dataloader = base_datamodule.test_dataloader()
        batch = next(iter(test_dataloader))
        assert not isinstance(test_dataloader.sampler, RandomSampler)
        assert not isinstance(val_dataloader.sampler, ContrastBatchSampler)
        assert batch['image'].shape[1:] == (3, 224, 224)
        assert batch['image'].shape[0] <= test_size
        b1 = base_datamodule.test_dataset[0]
        b2 = base_datamodule.test_dataset[0]
        assert np.allclose(b1['image'], b2['image'], rtol=1e-3)

    def test_full_labeled_dataloader(self, base_datamodule) -> None:
        # Arrange / Act
        loader = base_datamodule.full_labeled_dataloader()
        # Assert
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert 'image' in batch
        assert batch['image'].shape[1:] == (3, 224, 224)

    def test_setup_without_augmentations(self, data_dir) -> None:
        # Arrange — dataset with no augmentation pipeline → random_split path

        dataset = BaseDataset(data_dir=data_dir, imgaug_pipeline=None)
        dm = BaseDataModule(dataset=dataset, train_probability=0.8)
        # Act
        dm.setup()
        # Assert — all three splits are populated
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None

    def test_use_sampler_without_augmentations_raises(self, data_dir) -> None:
        # Arrange — sampler requires an augmentation pipeline; None pipeline should assert

        dataset = BaseDataset(data_dir=data_dir, imgaug_pipeline=None)
        dm = BaseDataModule(dataset=dataset, train_probability=0.8, use_sampler=True)
        # Act / Assert
        with pytest.raises(AssertionError, match='Sampler cannot be used without augmentations'):
            dm.setup()

    def test_slurm_env_var_sets_num_workers(self, data_dir, monkeypatch) -> None:
        # Arrange — SLURM_CPUS_PER_TASK present; num_workers should be read from it

        monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
        dataset = BaseDataset(data_dir=data_dir, imgaug_pipeline=None)
        # Act
        dm = BaseDataModule(dataset=dataset)
        # Assert
        assert dm.num_workers == 4


class TestBaseDataModuleContrastive:
    """Test BaseDataModule with contrastive (sampler-based) configuration."""

    def test_contrastive_datamodule_properties(self, base_datamodule_contrastive) -> None:
        assert base_datamodule_contrastive.use_sampler is True
        assert base_datamodule_contrastive.train_batch_size % 2 == 0

        np.random.seed(0)
        train_dataloader = base_datamodule_contrastive.train_dataloader()
        assert isinstance(train_dataloader.sampler, ContrastBatchSampler)

        sampler = train_dataloader.sampler
        assert sampler.batch_size == base_datamodule_contrastive.train_batch_size
        assert sampler.num_samples == len(base_datamodule_contrastive.train_dataset)
        assert sampler.batch_size % 2 == 0

        batch = next(iter(train_dataloader))
        assert isinstance(batch, dict)
        assert 'image' in batch
        assert 'idx' in batch

        expected_batch_size = base_datamodule_contrastive.train_batch_size
        num_pairs = expected_batch_size // 2
        ref_indices = batch['idx'][:num_pairs]
        pos_indices = batch['idx'][num_pairs:]
        assert torch.all(ref_indices >= 0)
        assert torch.all(pos_indices >= 0)

    def test_all_batches_have_correct_shape(self, base_datamodule_contrastive) -> None:
        np.random.seed(1)
        expected_batch_size = base_datamodule_contrastive.train_batch_size
        train_dataloader = base_datamodule_contrastive.train_dataloader()
        batch_count = 0
        for batch in train_dataloader:
            assert batch['image'].shape == (expected_batch_size, 3, 224, 224)
            assert batch['idx'].shape == (expected_batch_size,)
            assert torch.all(batch['idx'] >= 0)
            unique_indices = torch.unique(batch['idx'][::2])
            assert len(unique_indices) == len(batch['idx']) // 2
            batch_count += 1
        assert batch_count > 0


class TestSplitSizesFromProbabilities:
    """Test the split_sizes_from_probabilities function."""

    def test_basic_splits(self) -> None:

        out = split_sizes_from_probabilities(100, train_probability=0.8)
        assert out[0] == 80 and out[1] == 10 and out[2] == 10

    def test_explicit_val_probability(self) -> None:

        out = split_sizes_from_probabilities(100, train_probability=0.8, val_probability=0.1)
        assert out[0] == 80 and out[1] == 10 and out[2] == 10

    def test_all_three_probabilities(self) -> None:

        out = split_sizes_from_probabilities(
            100, train_probability=0.8, val_probability=0.1, test_probability=0.1,
        )
        assert out[0] == 80 and out[1] == 10 and out[2] == 10

    def test_leftover_samples_go_to_test(self) -> None:

        out = split_sizes_from_probabilities(101, train_probability=0.7)
        assert out[0] == 70 and out[1] == 15 and out[2] == 16

    def test_minimum_val_sample(self) -> None:

        out = split_sizes_from_probabilities(10, train_probability=0.95, val_probability=0.05)
        assert sum(out) == 10
        assert out[0] == 9
        assert out[1] == 1
        assert out[2] == 0

    def test_too_few_samples_raises(self) -> None:

        with pytest.raises(ValueError):
            split_sizes_from_probabilities(1, train_probability=0.95)
