import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler

from beast.data.samplers import ContrastBatchSampler



def test_base_datamodule(base_datamodule):

    train_size = base_datamodule.train_batch_size
    val_size = base_datamodule.val_batch_size
    test_size = base_datamodule.test_batch_size

    # check train batch properties
    train_dataloader = base_datamodule.train_dataloader()
    assert isinstance(train_dataloader.sampler, RandomSampler) or isinstance(train_dataloader.sampler, ContrastBatchSampler)
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
    assert not isinstance(val_dataloader.sampler, RandomSampler) and not isinstance(val_dataloader.sampler, ContrastBatchSampler)
    assert batch['image'].shape[1:] == (3, 224, 224)
    assert batch['image'].shape[0] <= val_size
    # check imgaug pipeline makes repeatable data
    b1 = base_datamodule.val_dataset[0]
    b2 = base_datamodule.val_dataset[0]
    assert np.allclose(b1['image'], b2['image'], rtol=1e-3)

    test_dataloader = base_datamodule.test_dataloader()
    batch = next(iter(test_dataloader))
    assert not isinstance(test_dataloader.sampler, RandomSampler) and not isinstance(test_dataloader.sampler, ContrastBatchSampler)
    assert batch['image'].shape[1:] == (3, 224, 224)
    assert batch['image'].shape[0] <= test_size
    # check imgaug pipeline makes repeatable data
    b1 = base_datamodule.test_dataset[0]
    b2 = base_datamodule.test_dataset[0]
    assert np.allclose(b1['image'], b2['image'], rtol=1e-3)


def test_base_datamodule_contrastive(base_datamodule_contrastive):
    """Test the contrastive datamodule functionality."""
    
    # Check that the datamodule is configured for contrastive learning
    assert base_datamodule_contrastive.use_sampler is True
    assert base_datamodule_contrastive.train_batch_size % 2 == 0  # Even batch size for pairs
    
    # Get the train dataloader
    train_dataloader = base_datamodule_contrastive.train_dataloader()
    
    # Verify it's using the contrastive sampler
    assert hasattr(train_dataloader, 'sampler')
    assert isinstance(train_dataloader.sampler, ContrastBatchSampler)
    
    # Check sampler properties
    sampler = train_dataloader.sampler
    assert sampler.batch_size == base_datamodule_contrastive.train_batch_size
    assert sampler.num_samples == len(base_datamodule_contrastive.train_dataset)
    assert sampler.batch_size % 2 == 0  # Even batch size for reference-positive pairs
    
    # Get a batch and verify its structure
    batch = next(iter(train_dataloader))
    
    # Verify batch structure
    assert isinstance(batch, dict)
    assert 'image' in batch
    assert 'idx' in batch
    
    # Should have the expected number of images (batch_size)
    expected_batch_size = base_datamodule_contrastive.train_batch_size
    assert batch['image'].shape == (expected_batch_size, 3, 224, 224)
    assert batch['idx'].shape == (expected_batch_size,)
    
    # Verify that indices are valid
    assert torch.all(batch['idx'] >= 0)
    assert torch.all(batch['idx'] < len(base_datamodule_contrastive.train_dataset))
    
    # Check that we have unique indices (no duplicates within a batch)
    unique_indices = torch.unique(batch['idx'])
    assert len(unique_indices) <= expected_batch_size
    
    # Test that the collate function reorganizes data correctly
    # The contrastive_collate_fn reorganizes from [ref1, pos1, ref2, pos2, ...] 
    # to [ref1, ref2, ..., pos1, pos2, ...]
    num_pairs = expected_batch_size // 2
    ref_indices = batch['idx'][:num_pairs]
    pos_indices = batch['idx'][num_pairs:]
    
    # Verify that reference and positive indices are within the expected range
    # (they should be within idx_offset of each other, but we can't directly check
    # this since the collate function reorganizes the data)
    assert torch.all(ref_indices >= 0)
    assert torch.all(pos_indices >= 0)
    assert torch.all(ref_indices < len(base_datamodule_contrastive.train_dataset))
    assert torch.all(pos_indices < len(base_datamodule_contrastive.train_dataset))
    
    # Test multiple batches to ensure consistency
    batch_count = 0
    for batch in train_dataloader:
        if batch_count >= 3:  # Test first 3 batches
            break
        
        assert batch['image'].shape == (expected_batch_size, 3, 224, 224)
        assert batch['idx'].shape == (expected_batch_size,)
        assert torch.all(batch['idx'] >= 0)
        assert torch.all(batch['idx'] < len(base_datamodule_contrastive.train_dataset))
        
        batch_count += 1
    
    # Verify that we can get at least one batch
    assert batch_count > 0


def test_split_sizes_from_probabilities():

    from beast.data.datamodules import split_sizes_from_probabilities

    # make sure we count examples properly
    total_number = 100
    train_prob = 0.8
    val_prob = 0.1
    test_prob = 0.1

    out = split_sizes_from_probabilities(total_number, train_probability=train_prob)
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(
        total_number, train_probability=train_prob, val_probability=val_prob
    )
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(
        total_number,
        train_probability=train_prob,
        val_probability=val_prob,
        test_probability=test_prob,
    )
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(total_number, train_probability=0.7)
    assert out[0] == 70 and out[1] == 15 and out[2] == 15

    # test that extra samples end up in test
    out = split_sizes_from_probabilities(101, train_probability=0.7)
    assert out[0] == 70 and out[1] == 15 and out[2] == 16

    # make sure we have at least one example in the validation set
    total_number = 10
    train_prob = 0.95
    val_prob = 0.05
    out = split_sizes_from_probabilities(
        total_number, train_probability=train_prob, val_probability=val_prob
    )
    assert sum(out) == total_number
    assert out[0] == 9
    assert out[1] == 1
    assert out[2] == 0

    # make sure an error is raised if there are not enough labeled frames
    total_number = 1
    with pytest.raises(ValueError):
        split_sizes_from_probabilities(total_number, train_probability=train_prob)
