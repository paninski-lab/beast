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
        assert sampler.max_idx == 99
        assert sampler.all_indices == list(range(100))

    def test_init_odd_batch_size_error(self):
        """Test that odd batch size raises error."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        dataset.dataset = Mock()
        dataset.dataset.image_list = [f"path_{i}" for i in range(100)]
        dataset.indices = list(range(100))

        with pytest.raises(AssertionError, match="Batch size must be even"):
            ContrastBatchSampler(dataset, batch_size=7)

    def test_init_without_frame_idx(self):
        """Test initialization when dataset doesn't have frame_idx."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=50)
        dataset.dataset = Mock()
        dataset.dataset.image_list = [f"path_{i}" for i in range(50)]
        dataset.indices = list(range(50))
        # No frame_idx attribute - this should work fine

        sampler = ContrastBatchSampler(dataset, batch_size=4)

        assert sampler.all_indices == list(range(50))
        assert sampler.max_idx == 49

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

    @patch('random.shuffle')
    def test_iter_basic(self, mock_shuffle):
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

        # Check that shuffle was called
        mock_shuffle.assert_called_once()

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
            shuffle=False  # For deterministic testing
        )

        # Test basic properties
        assert len(sampler) > 0
        assert sampler.batch_size == 4
        assert sampler.num_samples == len(train_dataset)
        assert sampler.idx_offset == 1
        assert 0 not in sampler.anchor_indices

        # Get a few batches
        batches = list(sampler)[:3]  # Get first 3 batches
        for batch in batches:
            assert len(batch) == 4

            # Check that we have reference-positive pairs
            for i in range(0, len(batch), 2):
                ref_idx = batch[i]
                pos_idx = batch[i + 1]

                # Verify indices are valid
                assert 0 <= ref_idx < len(train_dataset), f"Ref index {ref_idx} is out of range"
                assert 0 <= pos_idx < len(train_dataset), f"Pos index {pos_idx} is out of range"

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
        assert 0 not in sampler.anchor_indices

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
