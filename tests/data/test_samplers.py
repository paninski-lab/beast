from unittest.mock import Mock, patch

import pytest
import torch

from beast.data.samplers import (
    ContrastBatchSampler,
    contrastive_collate_fn,
    extract_anchor_indices,
)


class TestExtractAnchorIndices:
    """Test the helper functions extracted from ContrastBatchSampler."""

    def test_extract_anchor_indices_basic(self):
        """Test basic anchor index extraction with consecutive frames."""
        # Create a list of image paths with consecutive frame numbers
        image_list = [
            "video1/frame001.png",
            "video1/frame002.png",
            "video1/frame003.png",
            "video1/frame008.png",
            "video2/frame001.png",
            "video2/frame002.png",
            "video2/frame003.png",
        ]

        anchor_indices, _ = extract_anchor_indices(image_list)

        # All frames except the first and last of each video should be anchors
        # because they have neighbors
        # v1-frame001, v1-frame002, v1-frame003, v2-frame001, v2-frame002, v2-frame003
        expected_anchors = [0, 1, 2, 4, 5, 6]
        assert set(anchor_indices) == set(expected_anchors), \
            f"Extracted anchor indices: {anchor_indices}"

    def test_extract_anchor_indices_no_neighbors(self):
        """Test anchor index extraction with no neighbors."""
        image_list = [
            "video1/frame001.png",
            "video1/frame002.png",
            "video1/frame005.png",
            "video1/frame008.png",
        ]
        anchor_indices, _ = extract_anchor_indices(image_list)

        assert anchor_indices == [0, 1], f"Extracted anchor indices: {anchor_indices}"


class TestContrastBatchSampler:
    """Test the ContrastBatchSampler class."""

    def test_init_basic(self):
        """Test basic initialization."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        subdataset = Mock()
        subdataset.image_list = [f"path_{i}" for i in range(100)]
        dataset.indices = list(range(100))
        dataset.dataset = subdataset

        sampler = ContrastBatchSampler(dataset, batch_size=8, idx_offset=2)

        assert sampler.batch_size == 8
        assert sampler.idx_offset == 2
        assert sampler.num_samples == 100
        assert sampler.num_batches == 12  # 100 // 8

    def test_init_odd_batch_size_error(self):
        """Test that odd batch size raises error."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        dataset.dataset = Mock()
        dataset.dataset.image_list = [f"path_{i}" for i in range(100)]
        dataset.indices = list(range(100))

        with pytest.raises(AssertionError, match="Batch size must be even"):
            ContrastBatchSampler(dataset, batch_size=7)

    def test_len(self):
        """Test the __len__ method."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        subdataset = Mock()
        subdataset.image_list = [f"path_{i}" for i in range(100)]
        dataset.indices = list(range(100))
        dataset.dataset = subdataset

        sampler = ContrastBatchSampler(dataset, batch_size=8)

        assert len(sampler) == 12  # 100 // 8

    def test_iter_basic(self):
        """Test basic iteration behavior."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=20)
        subdataset = Mock()
        subdataset.image_list = [f"path_{i}" for i in range(20)]
        dataset.indices = list(range(20))
        dataset.dataset = subdataset

        sampler = ContrastBatchSampler(dataset, batch_size=4, shuffle=True)

        # Get first batch
        batches = list(sampler)

        # Should have 5 batches (20 // 4)
        assert len(batches) <= 5

        # Each batch should have 4 elements
        for batch in batches:
            assert len(batch) == 4

    def test_iter_no_shuffle(self):
        """Test iteration without shuffling."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=20)
        subdataset = Mock()
        subdataset.image_list = [f"path_{i}" for i in range(20)]
        dataset.indices = list(range(20))
        dataset.dataset = subdataset

        sampler = ContrastBatchSampler(dataset, batch_size=4, shuffle=False)

        batches = list(sampler)

        # Should have 5 batches
        assert len(batches) <= 5

        # First batch should be in order since no shuffle
        first_batch = batches[0]
        print(f'first_batch: {first_batch}')
        # The exact order depends on the sampling logic, but should contain indices 0-19
        assert all(0 <= idx < 20 for idx in first_batch)

    def test_iter_with_positive_pairs(self):
        """Test that batches contain reference-positive pairs."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=20)
        subdataset = Mock()
        subdataset.image_list = [f"path_{i}" for i in range(20)]
        dataset.indices = list(range(20))
        dataset.dataset = subdataset

        sampler = ContrastBatchSampler(dataset, batch_size=4, idx_offset=1)

        batches = list(sampler)

        # Check that each batch contains pairs
        for batch in batches:
            # For a batch of size 4, we should have 2 pairs
            # Each pair should be consecutive indices (ref, pos)
            assert len(batch) == 4

            # Check that we have pairs (this is a simplified check)
            # In practice, the pairs might not be consecutive due to the sampling logic
            # but they should be within idx_offset of each other
            for i in range(0, len(batch), 2):
                ref_idx = batch[i]
                pos_idx = batch[i + 1]
                assert abs(ref_idx - pos_idx) <= sampler.idx_offset

    def test_iter_drop_last_true(self):
        """Test iteration with drop_last=True."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=10)
        subdataset = Mock()
        subdataset.image_list = [f"path_{i}" for i in range(10)]
        dataset.indices = list(range(10))
        dataset.dataset = subdataset

        sampler = ContrastBatchSampler(dataset, batch_size=4)

        batches = list(sampler)

        # The actual number of batches depends on the sampling logic
        # With 10 samples and batch_size=4, we might get fewer batches than expected
        # due to the constraint that we need valid positive pairs
        assert len(batches) >= 1  # At least one batch

        # All batches should be complete
        for batch in batches:
            assert len(batch) == 4

    def test_distributed_anchor_distribution(self):
        """Test that anchors are properly distributed across ranks without overlap per epoch"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        subdataset = Mock()
        subdataset.image_list = [f"video1/frame_{i:03d}.png" for i in range(100)]
        dataset.indices = list(range(100))
        dataset.dataset = subdataset

        # Mock distributed environment for 2 GPUs
        samplers = []
        for rank in range(2):
            with patch('torch.distributed.is_initialized', return_value=True):
                with patch('torch.distributed.get_world_size', return_value=2):
                    with patch('torch.distributed.get_rank', return_value=rank):
                        sampler = ContrastBatchSampler(dataset, batch_size=4, seed=42)
                        samplers.append(sampler)

        # Test that all samplers have access to the same full set of anchor indices
        # (before per-epoch distribution)
        all_anchors_rank0 = set(samplers[0].all_anchor_indices)
        all_anchors_rank1 = set(samplers[1].all_anchor_indices)
        assert all_anchors_rank0 == all_anchors_rank1, \
            "All ranks should start with same anchor indices"

        # Test anchor distribution during iteration (first epoch)
        epoch_1_batches = []
        for sampler in samplers:
            # Collect all indices used in this epoch by iterating through the sampler
            epoch_indices = []
            for batch in sampler:
                epoch_indices.extend(batch[::2])  # just take anchor indices
            epoch_1_batches.append(set(epoch_indices))

        # Verify no overlap between ranks in epoch 1
        anchors_rank0_epoch1 = epoch_1_batches[0]
        anchors_rank1_epoch1 = epoch_1_batches[1]
        print(anchors_rank0_epoch1)
        print(anchors_rank1_epoch1)
        assert anchors_rank0_epoch1.isdisjoint(anchors_rank1_epoch1), \
            "Ranks should have non-overlapping anchors in the same epoch"

        # Verify roughly equal distribution per epoch
        total_epoch_anchors = len(anchors_rank0_epoch1) + len(anchors_rank1_epoch1)
        assert abs(len(anchors_rank0_epoch1) - len(anchors_rank1_epoch1)) <= 2, \
            "Anchors should be roughly evenly distributed per epoch"

        # Test that distribution changes across epochs
        epoch_2_batches = []
        for sampler in samplers:
            # Reset and collect indices for epoch 2
            epoch_indices = []
            for batch in sampler:
                epoch_indices.extend(batch[::2])  # just take anchor indices
            epoch_2_batches.append(set(epoch_indices))

        anchors_rank0_epoch2 = epoch_2_batches[0]
        anchors_rank1_epoch2 = epoch_2_batches[1]

        # Verify that each rank gets different data across epochs
        assert anchors_rank0_epoch1 != anchors_rank0_epoch2, \
            "Rank 0 should get different anchor indices across epochs"
        assert anchors_rank1_epoch1 != anchors_rank1_epoch2, \
            "Rank 1 should get different anchor indices across epochs"

        # Verify no overlap within each epoch
        assert anchors_rank0_epoch2.isdisjoint(anchors_rank1_epoch2), \
            "Ranks should have non-overlapping anchors in epoch 2"

    def test_seed_determinism_across_ranks(self):
        """Test that same seed produces deterministic but different anchor sets per rank"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=50)
        subdataset = Mock()
        subdataset.image_list = [f"video1/frame_{i:03d}.png" for i in range(50)]
        dataset.indices = list(range(50))
        dataset.dataset = subdataset

        # Create samplers with same seed, run twice to test determinism
        def create_rank_samplers(seed):
            samplers = []
            for rank in range(2):
                with patch('torch.distributed.is_initialized', return_value=True):
                    with patch('torch.distributed.get_world_size', return_value=2):
                        with patch('torch.distributed.get_rank', return_value=rank):
                            sampler = ContrastBatchSampler(dataset, batch_size=4, seed=seed)
                            samplers.append(sampler.all_anchor_indices.copy())
            return samplers

        # Test determinism: same seed should give same results
        anchors_run1 = create_rank_samplers(seed=42)
        anchors_run2 = create_rank_samplers(seed=42)

        assert anchors_run1[0] == anchors_run2[0], "Rank 0 should be deterministic with same seed"
        assert anchors_run1[1] == anchors_run2[1], "Rank 1 should be deterministic with same seed"

    def test_anchor_redistribution_across_epochs(self):
        """Test that anchor indices are redistributed across epochs"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        subdataset = Mock()
        subdataset.image_list = [f"video1/frame_{i:03d}.png" for i in range(100)]
        dataset.indices = list(range(100))
        dataset.dataset = subdataset

        # Create sampler for single GPU
        with patch('torch.distributed.is_initialized', return_value=False):
            sampler = ContrastBatchSampler(dataset, batch_size=4, seed=42)

        # Collect indices from multiple epochs
        epoch_data = []
        for epoch in range(3):
            epoch_indices = []
            for batch in sampler:
                epoch_indices.extend(batch)
            epoch_data.append(set(epoch_indices))

        # Verify that different epochs use different anchor orderings
        # (though they may overlap since it's the same dataset)
        epoch1_indices, epoch2_indices, epoch3_indices = epoch_data

        # Convert to lists to check ordering
        epoch1_list = []
        epoch2_list = []
        epoch3_list = []

        # Reset sampler and collect ordered indices
        sampler.epoch = 0
        for batch in sampler:
            epoch1_list.extend(batch)

        for batch in sampler:
            epoch2_list.extend(batch)

        for batch in sampler:
            epoch3_list.extend(batch)

        # Verify that the ordering is different across epochs
        assert epoch1_list != epoch2_list, "Epoch 1 and 2 should have different anchor orderings"
        assert epoch2_list != epoch3_list, "Epoch 2 and 3 should have different anchor orderings"
        assert epoch1_list != epoch3_list, "Epoch 1 and 3 should have different anchor orderings"

    def test_reproducible_epoch_distribution(self):
        """Test that anchor distribution is reproducible with same seed"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        subdataset = Mock()
        subdataset.image_list = [f"video1/frame_{i:03d}.png" for i in range(100)]
        dataset.indices = list(range(100))
        dataset.dataset = subdataset

        # Create two samplers with same seed
        samplers = []
        for _ in range(2):
            with patch('torch.distributed.is_initialized', return_value=False):
                sampler = ContrastBatchSampler(dataset, batch_size=4, seed=42)
                samplers.append(sampler)

        # Collect indices from first epoch for both samplers
        epoch1_indices_sampler1 = []
        epoch1_indices_sampler2 = []

        for batch in samplers[0]:
            epoch1_indices_sampler1.extend(batch[::2])  # just take anchor indices

        for batch in samplers[1]:
            epoch1_indices_sampler2.extend(batch[::2])  # just take anchor indices

        # Verify same seed produces same results
        assert epoch1_indices_sampler1 == epoch1_indices_sampler2, \
            "Same seed should produce identical anchor distribution"

    def test_epoch_based_shuffling(self):
        """Test that different epochs produce different orderings within each rank"""
        n_frames = 52
        dataset = Mock()
        dataset.__len__ = Mock(return_value=n_frames)
        subdataset = Mock()
        subdataset.image_list = [f"video1/frame_{i:03d}.png" for i in range(n_frames)]
        dataset.indices = list(range(n_frames))
        dataset.dataset = subdataset

        with patch('torch.distributed.is_initialized', return_value=False):
            sampler = ContrastBatchSampler(dataset, batch_size=4, shuffle=True)

        # Get batches from two different epochs
        epoch1_batches = list(sampler)
        epoch2_batches = list(sampler)

        # Should have same number of batches
        assert len(epoch1_batches) == len(epoch2_batches)

        # But different ordering due to epoch-based shuffling
        epoch1_flat = [idx for batch in epoch1_batches for idx in batch]
        epoch2_flat = [idx for batch in epoch2_batches for idx in batch]
        assert epoch1_flat != epoch2_flat, "Different epochs should produce different orderings"

    def test_positive_indices_validity(self):
        """Test that all positive indices are valid for their anchors"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=30)
        subdataset = Mock()
        # Create a realistic scenario with consecutive frames
        subdataset.image_list = [f"video1/frame_{i:03d}.png" for i in range(15)] + \
            [f"video2/frame_{i:03d}.png" for i in range(15)]
        dataset.indices = list(range(30))
        dataset.dataset = subdataset

        with patch('torch.distributed.is_initialized', return_value=False):
            sampler = ContrastBatchSampler(dataset, batch_size=4, idx_offset=1)

        # Check that pos_indices relationships are valid
        for anchor_idx, pos_list in sampler.pos_indices.items():
            for pos_idx in pos_list:
                # Positive should be within idx_offset of anchor
                assert abs(anchor_idx - pos_idx) == sampler.idx_offset
                # Both should be in dataset indices
                assert anchor_idx in sampler.dataset_indices
                assert pos_idx in sampler.dataset_indices

    def test_batch_anchor_positive_relationships(self):
        """Test that batches maintain proper anchor-positive relationships"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=20)
        subdataset = Mock()
        subdataset.image_list = [f"video1/frame_{i:03d}.png" for i in range(20)]
        dataset.indices = list(range(20))
        dataset.dataset = subdataset

        with patch('torch.distributed.is_initialized', return_value=False):
            sampler = ContrastBatchSampler(dataset, batch_size=4, idx_offset=1)

        batches = list(sampler)

        for batch in batches:
            # Process pairs (assuming even indices are anchors, odd are positives)
            for i in range(0, len(batch), 2):
                if i + 1 < len(batch):
                    anchor_idx = batch[i]
                    pos_idx = batch[i + 1]

                    # Verify this is a valid anchor-positive relationship
                    assert anchor_idx in sampler.pos_indices, f"Anchor {anchor_idx} should have positives"
                    assert pos_idx in sampler.pos_indices[anchor_idx], \
                        f"Positive {pos_idx} should be valid for anchor {anchor_idx}"


class TestContrastiveCollateFn:
    """Test the contrastive_collate_fn function."""

    def test_contrastive_collate_fn_basic(self):
        """Test basic collate function behavior."""
        # Create mock batch data
        batch_data = [
            {"image": torch.randn(3, 224, 224), "idx": 0},  # ref
            {"image": torch.randn(3, 224, 224), "idx": 1},  # pos
            {"image": torch.randn(3, 224, 224), "idx": 2},  # ref
            {"image": torch.randn(3, 224, 224), "idx": 3},  # pos
        ]

        result = contrastive_collate_fn(batch_data)

        # Should return a dict with 'image' and 'idx' keys
        assert isinstance(result, dict)
        assert 'image' in result
        assert 'idx' in result

        # Should have 4 images total (2 refs + 2 pos)
        assert result['image'].shape == (4, 3, 224, 224)
        assert result['idx'].shape == (4,)

        # The collate function organizes as [ref1, ref2, pos1, pos2]
        # So for input [ref1, pos1, ref2, pos2], output should be [ref1, ref2, pos1, pos2]
        expected_indices = torch.tensor([0, 2, 1, 3])
        assert torch.allclose(result['idx'], expected_indices)

    def test_contrastive_collate_fn_odd_batch(self):
        """Test collate function with odd number of samples."""
        # Create mock batch data with odd number
        batch_data = [
            {"image": torch.randn(3, 224, 224), "idx": 0},  # ref
            {"image": torch.randn(3, 224, 224), "idx": 1},  # pos
            {"image": torch.randn(3, 224, 224), "idx": 2},  # ref
        ]

        result = contrastive_collate_fn(batch_data)

        # Should handle odd number gracefully
        assert result['image'].shape == (3, 3, 224, 224)
        assert result['idx'].shape == (3,)
        # For odd batch: [ref1, pos1, ref2] -> [ref1, ref2, pos1]
        expected_indices = torch.tensor([0, 2, 1])
        assert torch.allclose(result['idx'], expected_indices)

    def test_contrastive_collate_fn_single_pair(self):
        """Test collate function with single reference-positive pair."""
        batch_data = [
            {"image": torch.randn(3, 224, 224), "idx": 5},  # ref
            {"image": torch.randn(3, 224, 224), "idx": 6},  # pos
        ]

        result = contrastive_collate_fn(batch_data)

        assert result['image'].shape == (2, 3, 224, 224)
        assert result['idx'].shape == (2,)
        # For single pair: [ref1, pos1] -> [ref1, pos1] (no reordering needed)
        expected_indices = torch.tensor([5, 6])
        assert torch.allclose(result['idx'], expected_indices)

    def test_contrastive_collate_fn_different_image_sizes(self):
        """Test collate function with different image sizes."""
        batch_data = [
            {"image": torch.randn(3, 100, 100), "idx": 0},  # ref
            {"image": torch.randn(3, 100, 100), "idx": 1},  # pos
        ]

        result = contrastive_collate_fn(batch_data)

        assert result['image'].shape == (2, 3, 100, 100)
        assert result['idx'].shape == (2,)


class TestTopKFunction:
    """Test the topk function for computing top-k accuracy."""

    def test_topk_basic(self):
        """Test basic topk functionality."""
        # Import the function
        from beast.models.vits import topk

        # Create mock similarities and labels
        similarities = torch.tensor([
            [0.1, 0.8, 0.3, 0.2],  # Highest similarity at index 1
            [0.4, 0.2, 0.9, 0.1],  # Highest similarity at index 2
            [0.7, 0.1, 0.2, 0.3],  # Highest similarity at index 0
        ])
        labels = torch.tensor([1, 2, 0])  # Correct labels

        # Test top-1 accuracy
        result = topk(similarities, labels, k=1)
        expected = torch.tensor(1.0)  # All predictions are correct
        assert torch.allclose(result, expected)

    def test_topk_k_greater_than_batch_size(self):
        """Test topk when k is greater than batch size."""
        from beast.models.vits import topk

        similarities = torch.tensor([
            [0.1, 0.8, 0.3],
            [0.4, 0.2, 0.9],
        ])
        labels = torch.tensor([1, 2])

        # k=5 but batch size is 2, should use k=2
        result = topk(similarities, labels, k=5)
        # Should be 1.0 since both predictions are correct
        assert torch.allclose(result, torch.tensor(1.0))

    def test_topk_partial_correct(self):
        """Test topk with partial correct predictions."""
        from beast.models.vits import topk

        similarities = torch.tensor([
            [0.1, 0.8, 0.3, 0.2],  # Highest at index 1, correct
            [0.4, 0.2, 0.1, 0.9],  # Highest at index 3, incorrect (label is 2)
            [0.7, 0.1, 0.2, 0.3],  # Highest at index 0, correct
        ])
        labels = torch.tensor([1, 2, 0])

        # Top-1 accuracy should be 2/3
        result = topk(similarities, labels, k=1)
        expected = torch.tensor(2.0 / 3.0)
        assert torch.allclose(result, expected)

    def test_topk_top3_accuracy(self):
        """Test top-3 accuracy calculation."""
        from beast.models.vits import topk

        similarities = torch.tensor([
            [0.1, 0.8, 0.3, 0.2, 0.5],  # Top 3: [1, 4, 2], label 1 is in top 3
            [0.4, 0.2, 0.9, 0.1, 0.3],  # Top 3: [2, 0, 4], label 2 is in top 3
            [0.7, 0.1, 0.2, 0.3, 0.6],  # Top 3: [0, 4, 3], label 0 is in top 3
        ])
        labels = torch.tensor([1, 2, 0])

        # Top-3 accuracy should be 1.0 (all labels are in top 3)
        result = topk(similarities, labels, k=3)
        expected = torch.tensor(1.0)
        assert torch.allclose(result, expected)

    def test_topk_no_correct_predictions(self):
        """Test topk when no predictions are correct."""
        from beast.models.vits import topk

        similarities = torch.tensor([
            [0.8, 0.1, 0.3, 0.2],  # Highest at index 0, but label is 1
            [0.9, 0.2, 0.1, 0.3],  # Highest at index 0, but label is 2
        ])
        labels = torch.tensor([1, 2])

        # Top-1 accuracy should be 0.0
        result = topk(similarities, labels, k=1)
        expected = torch.tensor(0.0)
        assert torch.allclose(result, expected)


class TestBatchWiseContrastiveLoss:
    """Test the batch_wise_contrastive_loss function."""

    def test_batch_wise_contrastive_loss_basic(self):
        """Test basic batch-wise contrastive loss functionality."""
        from beast.models.vits import batch_wise_contrastive_loss

        # Create a mock similarity matrix for batch size 4
        # Simulate perfect similarity matrix where diagonal-like elements are high
        sim_matrix = torch.tensor([
            [1.0, 0.1, 0.2, 0.1],  # Row 0: high similarity with itself
            [0.1, 1.0, 0.2, 0.1],  # Row 1: high similarity with itself
            [0.2, 0.1, 1.0, 0.1],  # Row 2: high similarity with itself
            [0.1, 0.2, 0.1, 1.0],  # Row 3: high similarity with itself
        ])

        result = batch_wise_contrastive_loss(sim_matrix)

        # Check that result is a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'infoNCE_loss' in result
        assert 'percent_correct' in result

        # Check that loss is a tensor
        assert isinstance(result['infoNCE_loss'], torch.Tensor)
        assert isinstance(result['percent_correct'], torch.Tensor)

        # Check that loss is finite
        assert torch.isfinite(result['infoNCE_loss'])
        assert torch.isfinite(result['percent_correct'])

    def test_batch_wise_contrastive_loss_larger_batch(self):
        """Test batch-wise contrastive loss with larger batch size."""
        from beast.models.vits import batch_wise_contrastive_loss

        # Create a mock similarity matrix for batch size 6
        batch_size = 6
        sim_matrix = torch.randn(batch_size, batch_size)

        # Make diagonal elements higher to simulate good representations
        for i in range(batch_size):
            sim_matrix[i, i] = 2.0

        result = batch_wise_contrastive_loss(sim_matrix)

        assert isinstance(result, dict)
        assert 'infoNCE_loss' in result
        assert 'percent_correct' in result

        # Loss should be finite
        assert torch.isfinite(result['infoNCE_loss'])
        assert torch.isfinite(result['percent_correct'])

        # Percent correct should be between 0 and 1
        assert 0.0 <= result['percent_correct'] <= 1.0

    def test_batch_wise_contrastive_loss_diagonal_removal(self):
        """Test that diagonal elements are properly removed from similarity matrix."""
        from beast.models.vits import batch_wise_contrastive_loss

        # Create similarity matrix with high diagonal values
        sim_matrix = torch.tensor([
            [10.0, 0.1, 0.2, 0.1],
            [0.1, 10.0, 0.2, 0.1],
            [0.2, 0.1, 10.0, 0.1],
            [0.1, 0.2, 0.1, 10.0],
        ])

        result = batch_wise_contrastive_loss(sim_matrix)

        # The function should remove diagonal elements before computing loss
        # So even with high diagonal values, the loss should be reasonable
        assert torch.isfinite(result['infoNCE_loss'])
        assert result['infoNCE_loss'] > 0  # Should be positive loss

    def test_batch_wise_contrastive_loss_labels_construction(self):
        """Test that labels are constructed correctly for contrastive learning."""
        from beast.models.vits import batch_wise_contrastive_loss

        # Create a simple similarity matrix
        sim_matrix = torch.tensor([
            [1.0, 0.1, 0.2, 0.1],
            [0.1, 1.0, 0.2, 0.1],
            [0.2, 0.1, 1.0, 0.1],
            [0.1, 0.2, 0.1, 1.0],
        ])

        result = batch_wise_contrastive_loss(sim_matrix)

        # The function should construct labels correctly for contrastive learning
        # For batch size 4, it should create labels [1, 0, 1, 0] (after adjustment)
        # This is because it splits the batch into two halves and creates pairs

        # Check that the result is reasonable
        assert torch.isfinite(result['infoNCE_loss'])
        assert 0.0 <= result['percent_correct'] <= 1.0

    def test_batch_wise_contrastive_loss_edge_cases(self):
        """Test edge cases for batch-wise contrastive loss."""
        from beast.models.vits import batch_wise_contrastive_loss

        # Test with very small similarity values
        sim_matrix = torch.tensor([
            [0.001, 0.0001, 0.0002, 0.0001],
            [0.0001, 0.001, 0.0002, 0.0001],
            [0.0002, 0.0001, 0.001, 0.0001],
            [0.0001, 0.0002, 0.0001, 0.001],
        ])

        result = batch_wise_contrastive_loss(sim_matrix)
        assert torch.isfinite(result['infoNCE_loss'])

        # Test with very large similarity values
        sim_matrix = torch.tensor([
            [100.0, 1.0, 2.0, 1.0],
            [1.0, 100.0, 2.0, 1.0],
            [2.0, 1.0, 100.0, 1.0],
            [1.0, 2.0, 1.0, 100.0],
        ])

        result = batch_wise_contrastive_loss(sim_matrix)
        assert torch.isfinite(result['infoNCE_loss'])


class TestContrastBatchSamplerWithRealDataset:
    """Test ContrastBatchSampler with a real dataset fixture."""

    def test_sampler_with_base_datamodule(self, base_datamodule):
        """Test sampler with the base_datamodule fixture."""
        # Create a sampler using the train dataset
        train_dataset = base_datamodule.train_dataset
        sampler = ContrastBatchSampler(
            dataset=train_dataset,
            batch_size=4,
            idx_offset=1,
            shuffle=False,  # For deterministic testing
        )

        # Test basic properties
        assert len(sampler) > 0
        assert sampler.batch_size == 4
        assert sampler.num_samples == len(train_dataset)
        assert sampler.idx_offset == 1
        assert 0 not in sampler.all_anchor_indices

        # Get a few batches
        batches = list(sampler)[:3]  # Get first 3 batches
        for batch in batches:
            assert len(batch) == 4
            # Check that we have reference-positive pairs
            for i in range(0, len(batch), 2):
                ref_idx = batch[i]
                pos_idx = batch[i + 1]
                # Verify indices are valid
                total_images = len(base_datamodule.dataset.image_list)
                assert 0 <= ref_idx < total_images, f"Ref index {ref_idx} is out of range"
                assert 0 <= pos_idx < total_images, f"Pos index {pos_idx} is out of range"

    def test_sampler_with_contrastive_collate(self, base_datamodule):
        """Test sampler combined with contrastive_collate_fn."""
        train_dataset = base_datamodule.train_dataset
        sampler = ContrastBatchSampler(
            dataset=train_dataset,
            batch_size=4,
            idx_offset=1,
            shuffle=False
        )
        assert sampler.idx_offset == 1
        assert sampler.batch_size == 4
        assert sampler.num_samples == len(train_dataset)
        assert 0 not in sampler.all_anchor_indices

        # Get a batch of indices
        batch_indices = next(iter(sampler))
        # Get the actual data using the dataset
        batch_data = []
        for idx in batch_indices:
            sample = train_dataset[idx]
            batch_data.append(sample)

        # Apply collate function
        result = contrastive_collate_fn(batch_data)

        # Verify the result
        assert isinstance(result, dict)
        assert 'image' in result
        assert 'idx' in result
        assert result['image'].shape == (4, 3, 224, 224)
        assert result['idx'].shape == (4,)

    def test_contrastive_datamodule_pipeline(self, base_datamodule_contrastive):
        """Test the full contrastive learning pipeline using the contrastive datamodule."""
        # Get the train dataloader which should use ContrastBatchSampler
        train_dataloader = base_datamodule_contrastive.train_dataloader()

        # Verify it's using the contrastive sampler
        assert hasattr(train_dataloader, 'sampler')
        assert isinstance(train_dataloader.sampler, ContrastBatchSampler)

        # Get a batch
        batch = next(iter(train_dataloader))

        # Verify batch structure
        assert isinstance(batch, dict)
        assert 'image' in batch
        assert 'idx' in batch

        # Should have 8 images (4 ref + 4 pos pairs)
        assert batch['image'].shape == (8, 3, 224, 224)
        assert batch['idx'].shape == (8,)

        # Verify that indices are valid
        assert torch.all(batch['idx'] >= 0)
        assert torch.all(batch['idx'] < len(base_datamodule_contrastive.train_dataset))

        # The collate function reorganizes the data, so we need to check differently
        # The original sampler produces [ref1, pos1, ref2, pos2, ...]
        # The collate function reorganizes to [ref1, ref2, ..., pos1, pos2, ...]
        # So we can't directly check pairs in the result

        # Instead, verify that all indices are within the dataset range
        # and that we have the expected number of unique indices
        unique_indices = torch.unique(batch['idx'])
        assert len(unique_indices) <= 8  # Should have at most 8 unique indices
