"""Tests for data module splitting and loading."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from beast.data.datamodules import (
    BaseDataModule,
    MultiViewDataModule,
    split_sizes_from_probabilities,
)
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
        with pytest.raises(ValueError, match='Sampler cannot be used without augmentations'):
            dm.setup()

    def test_train_dataloader_without_setup_raises(self, base_dataset) -> None:
        # Arrange
        dm = BaseDataModule(dataset=base_dataset, train_probability=0.8)
        # Act / Assert
        with pytest.raises(RuntimeError, match='call setup()'):
            dm.train_dataloader()

    def test_val_dataloader_without_setup_raises(self, base_dataset) -> None:
        # Arrange
        dm = BaseDataModule(dataset=base_dataset, train_probability=0.8)
        # Act / Assert
        with pytest.raises(RuntimeError, match='call setup()'):
            dm.val_dataloader()

    def test_test_dataloader_without_setup_raises(self, base_dataset) -> None:
        # Arrange
        dm = BaseDataModule(dataset=base_dataset, train_probability=0.8)
        # Act / Assert
        with pytest.raises(RuntimeError, match='call setup()'):
            dm.test_dataloader()

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

    def test_probabilities_not_summing_to_one_raises(self) -> None:

        with pytest.raises(ValueError, match='must sum to 1.0'):
            split_sizes_from_probabilities(
                100,
                train_probability=0.8,
                val_probability=0.1,
                test_probability=0.2,
            )


# ---------------------------------------------------------------------------
# MultiViewDataModule tests
# ---------------------------------------------------------------------------

_IMAGE_SIZE = 64
_N_VIEWS = 3
_N_TOTAL_FRAMES = 20  # 2 sessions × 10 frames


class TestMultiViewDataModule:
    """Test the MultiViewDataModule class."""

    def test_setup_creates_splits(self, multiview_datamodule) -> None:
        assert multiview_datamodule.train_dataset is not None
        assert multiview_datamodule.val_dataset is not None
        assert (
            len(multiview_datamodule.train_dataset) + len(multiview_datamodule.val_dataset)
            == _N_TOTAL_FRAMES
        )

    def test_split_sizes(self, multiview_datamodule) -> None:
        assert len(multiview_datamodule.train_dataset) == 16
        assert len(multiview_datamodule.val_dataset) == 4

    def test_train_batch_shape(self, multiview_datamodule) -> None:
        batch = next(iter(multiview_datamodule.train_dataloader()))
        assert batch['image'].shape == (2, _N_VIEWS, 3, _IMAGE_SIZE, _IMAGE_SIZE)
        assert batch['c2w'].shape == (2, _N_VIEWS, 4, 4)
        assert batch['fxfycxcy'].shape == (2, _N_VIEWS, 4)

    def test_val_batch_shape(self, multiview_datamodule) -> None:
        batch = next(iter(multiview_datamodule.val_dataloader()))
        assert batch['image'].shape == (2, _N_VIEWS, 3, _IMAGE_SIZE, _IMAGE_SIZE)

    def test_train_mode_shuffles_views_val_does_not(self, multiview_datamodule) -> None:
        assert multiview_datamodule.train_dataset.mode == 'train'
        assert multiview_datamodule.val_dataset.mode == 'test'

    def test_sequential_split_no_overlap(self, multiview_datamodule) -> None:
        train_ids = set(multiview_datamodule.train_dataset.unique_frame_ids)
        val_ids = set(multiview_datamodule.val_dataset.unique_frame_ids)
        assert train_ids.isdisjoint(val_ids)

    def test_mask_propagated_to_batch(self, multiview_data_dir) -> None:
        dm = MultiViewDataModule(
            data_dir=multiview_data_dir, image_size=_IMAGE_SIZE,
            train_batch_size=2, use_mask=True, num_workers=0,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert 'input_mask' in batch
        assert batch['input_mask'].shape == (2, _N_VIEWS, 1, _IMAGE_SIZE, _IMAGE_SIZE)

    def test_val_at_least_one_frame(self, multiview_data_dir) -> None:
        # train_fraction=1.0 should still give at least one val frame
        dm = MultiViewDataModule(
            data_dir=multiview_data_dir, image_size=_IMAGE_SIZE, train_fraction=1.0,
        )
        dm.setup()
        assert len(dm.val_dataset) >= 1

    def test_invalid_train_fraction_raises(self, multiview_data_dir) -> None:
        with pytest.raises(ValueError, match='train_fraction'):
            MultiViewDataModule(
                data_dir=multiview_data_dir, image_size=_IMAGE_SIZE, train_fraction=0.0,
            )

    def test_setup_required_before_dataloader(self, multiview_data_dir) -> None:
        dm = MultiViewDataModule(data_dir=multiview_data_dir, image_size=_IMAGE_SIZE)
        with pytest.raises(RuntimeError, match='call setup()'):
            dm.train_dataloader()
        with pytest.raises(RuntimeError, match='call setup()'):
            dm.val_dataloader()

    def test_slurm_env_var_sets_num_workers(self, multiview_data_dir, monkeypatch) -> None:
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
        dm = MultiViewDataModule(data_dir=multiview_data_dir, image_size=_IMAGE_SIZE)
        assert dm.num_workers == 4
