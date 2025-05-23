"""Data modules split a dataset into train, val, and test modules."""

import copy
import os
from typeguard import typechecked

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split


@typechecked
class BaseDataModule(pl.LightningDataModule):
    """Splits a labeled dataset into train, val, and test data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 16,
        num_workers: int | None = None,
        train_probability: float = 0.8,
        val_probability: float | None = None,
        test_probability: float | None = None,
        seed: int = 42,
    ) -> None:
        """Data module splits a dataset into train, val, and test data loaders.

        Args:
            dataset: base dataset to be split into train/val/test
            train_batch_size: number of samples of training batches
            val_batch_size: number of samples in validation batches
            test_batch_size: number of samples in test batches
            num_workers: number of threads used for prefetching data
            train_probability: fraction of full dataset used for training
            val_probability: fraction of full dataset used for validation
            test_probability: fraction of full dataset used for testing
            train_frames: if integer, select this number of training frames
                from the initially selected train frames (defined by
                `train_probability`); if float, must be between 0 and 1
                (exclusive) and defines the fraction of the initially selected
                train frames
            seed: control data splits

        """
        super().__init__()
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        if num_workers is not None:
            self.num_workers = num_workers
        else:
            slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
            if slurm_cpus:
                self.num_workers = int(slurm_cpus)
            else:
                # Fallback to os.cpu_count()
                self.num_workers = os.cpu_count()
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.test_probability = test_probability
        self.train_dataset = None  # populated by self.setup()
        self.val_dataset = None  # populated by self.setup()
        self.test_dataset = None  # populated by self.setup()
        self.seed = seed
        self._setup()

    def _setup(self) -> None:

        datalen = self.dataset.__len__()
        print(f'Number of images in the full dataset (train+val+test): {datalen}')

        # split data based on provided probabilities
        data_splits_list = split_sizes_from_probabilities(
            datalen,
            train_probability=self.train_probability,
            val_probability=self.val_probability,
            test_probability=self.test_probability,
        )

        if self.dataset.imgaug_pipeline is None:
            # no augmentations in the pipeline; subsets can share same underlying dataset
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                data_splits_list,
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            # augmentations in the pipeline; we want validation and test datasets that only resize
            # we can't simply change the imgaug pipeline in the datasets after they've been split
            # because the subsets actually point to the same underlying dataset, so we create
            # separate datasets here
            train_idxs, val_idxs, test_idxs = random_split(
                range(len(self.dataset)),
                data_splits_list,
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.train_dataset = Subset(copy.deepcopy(self.dataset), indices=list(train_idxs))
            self.val_dataset = Subset(copy.deepcopy(self.dataset), indices=list(val_idxs))
            self.test_dataset = Subset(copy.deepcopy(self.dataset), indices=list(test_idxs))

            self.val_dataset.dataset.imgaug_pipeline = None
            self.test_dataset.dataset.imgaug_pipeline = None

        print(
            f'Dataset splits -- '
            f'train: {len(self.train_dataset)}, '
            f'val: {len(self.val_dataset)}, '
            f'test: {len(self.test_dataset)}'
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def full_labeled_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )


@typechecked
def split_sizes_from_probabilities(
    total_number: int,
    train_probability: float,
    val_probability: float | None = None,
    test_probability: float | None = None,
) -> list[int]:
    """Returns the number of examples for train, val and test given split probs.

    Args:
        total_number: total number of examples in dataset
        train_probability: fraction of examples used for training
        val_probability: fraction of examples used for validation
        test_probability: fraction of examples used for test. Defaults to None. Can be computed
            as the remaining examples.

    Returns:
        [num training examples, num validation examples, num test examples]

    """

    if test_probability is None and val_probability is None:
        remaining_probability = 1.0 - train_probability
        # round each to 5 decimal places (issue with floating point precision)
        val_probability = round(remaining_probability / 2, 5)
        test_probability = round(remaining_probability / 2, 5)
    elif test_probability is None:
        test_probability = 1.0 - train_probability - val_probability

    # probabilities should add to one
    assert test_probability + train_probability + val_probability == 1.0

    # compute numbers from probabilities
    train_number = int(np.floor(train_probability * total_number))
    val_number = int(np.floor(val_probability * total_number))

    # if we lose extra examples by flooring, send these to train_number or test_number, depending
    leftover = total_number - train_number - val_number
    if leftover < 5:
        # very few samples, let's bulk up train
        train_number += leftover
        test_number = 0
    else:
        test_number = leftover

    # make sure that we have at least one validation sample
    if val_number == 0:
        train_number -= 1
        val_number += 1
        if train_number < 1:
            raise ValueError('Must have at least two labeled frames, one train and one validation')

    # assert that we're using all datapoints
    assert train_number + test_number + val_number == total_number

    return [train_number, val_number, test_number]
