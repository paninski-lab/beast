import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def extract_anchor_indices(image_list, idx_offset=1):
    """
    Extract anchor indices from image paths that have valid neighboring frames.

    Args:
        image_list: List of image paths

    Returns:
        List of indices that can serve as anchors (have valid neighbors)
    """
    anchor_indices = []
    pos_indices = {}

    # Parse each image path to extract video and frame information
    frame_info = []
    for idx, img_path in enumerate(image_list):
        path = Path(img_path)
        # Extract video name (parent directory) and frame number from filename
        video_name = path.parent.name
        # Try to extract frame number from filename
        # (assuming format like frame001.png, 001.png, etc.)
        frame_match = re.search(r'(\d+)', path.stem)
        if frame_match:
            frame_num = int(frame_match.group(1))
            frame_info.append({
                'idx': idx,
                'video': video_name,
                'frame_num': frame_num,
                'path': path
            })

    # Sort by video and frame number
    frame_info.sort(key=lambda x: (x['video'], x['frame_num']))

    # Find frames that have valid neighbors (same video, consecutive frame numbers)
    for i, frame in enumerate(frame_info):
        # Check if previous frame exists and is from same video
        has_prev = (
            i > 0 and
            frame_info[i-1]['video'] == frame['video'] and
            frame_info[i-1]['frame_num'] == frame['frame_num'] - idx_offset
        )

        # Check if next frame exists and is from same video
        has_next = (
            i < len(frame_info) - 1 and
            frame_info[i+1]['video'] == frame['video'] and
            frame_info[i+1]['frame_num'] == frame['frame_num'] + idx_offset
        )

        # Add to anchor indices if it has at least one valid neighbor
        if has_prev or has_next:
            anchor_indices.append(frame['idx'])
            pos = []
            if has_prev:
                pos.append(frame_info[i-1]['idx'])
            if has_next:
                pos.append(frame_info[i+1]['idx'])
            if len(pos) > 0:
                pos_indices[frame['idx']] = pos
    # return the indices in sorted order
    anchor_indices.sort()
    return anchor_indices, pos_indices


class ContrastBatchSampler(Sampler):
    """
    Custom batch sampler:
      - Each batch is exactly `batch_size = N` clips.
      - For each reference index i, we pick exactly one i_p from i +/- idx_offset
        (and skip if none are valid).
      - The rest of the clips in that batch are chosen from "far away" indices.
    The __len__ of this sampler is #batches, i.e. total_clips // batch_size.
    """

    def __init__(self, dataset, batch_size, idx_offset=1, shuffle=True, seed=42):

        super().__init__()

        # Get distributed training info
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        assert batch_size % 2 == 0, (
            "Batch size must be even to form (ref, pos) pairs."
        )

        self.dataset = dataset
        self.batch_size = batch_size
        self.idx_offset = idx_offset
        self.shuffle = shuffle
        self.num_samples = len(dataset)  # total number of clips

        # Calculate samples per replica
        self.samples_per_replica = self.num_samples // self.num_replicas
        self.total_samples = self.samples_per_replica * self.num_replicas
        
        # Calculate batches for this replica
        self.num_batches = self.samples_per_replica // self.batch_size

        # Extract anchors only from the subset's image list (subset = train/val/test)
        self.dataset_indices = sorted(dataset.indices)
        subset_image_list = [dataset.dataset.image_list[i] for i in self.dataset_indices]
        self.anchor_indices, self.pos_indices = extract_anchor_indices(
            subset_image_list, idx_offset=self.idx_offset,
        )

        # CRITICAL: Shuffle anchor indices with fixed seed across all GPUs
        # This ensures all GPUs see the same shuffled order before chunking
        rng = np.random.RandomState(seed)  # Use numpy for deterministic shuffling
        rng.shuffle(self.anchor_indices)

        # Distribute shuffled anchor indices across replicas
        indices_per_replica = len(self.anchor_indices) // self.num_replicas
        start_idx = self.rank * indices_per_replica
        end_idx = start_idx + indices_per_replica
        
        # Handle remainder indices for the last replica
        if self.rank == self.num_replicas - 1:
            end_idx = len(self.anchor_indices)
            
        self.anchor_indices = self.anchor_indices[start_idx:end_idx]

        self.epoch = 0
        self.seed = seed

    def __iter__(self):

        self.epoch += 1

        # Set random seed for reproducible shuffling across replicas
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + hash(self.rank))
            indices = torch.randperm(len(self.anchor_indices), generator=g).tolist()
            anchor_indices = [self.anchor_indices[i] for i in indices]
        else:
            anchor_indices = self.anchor_indices.copy()

        used = set()
        batches_returned = 0
        idx_cursor = 0

        while batches_returned < self.num_batches:
            batch = []
            # Keep pairing up references and positives until we have batch_size
            while len(batch) < self.batch_size:
                # Find next unused anchor
                while (
                    idx_cursor < len(anchor_indices)
                    and anchor_indices[idx_cursor] in used
                ):
                    idx_cursor += 1

                if idx_cursor >= len(anchor_indices):
                    break

                i = anchor_indices[idx_cursor]

                # Find valid positive indices
                valid_positives = [
                    p for p in self.pos_indices[i] 
                    if p in self.dataset_indices and p not in used
                ]
                
                if not valid_positives:
                    used.add(i)  # Mark this anchor as used even if no valid positives
                    idx_cursor += 1
                    continue
                    
                # Choose random positive
                i_p = np.random.choice(valid_positives)

                batch.extend([i, i_p])
                used.update(self.pos_indices[i])
                used.add(i)

                idx_cursor += 1
                if idx_cursor >= len(anchor_indices):
                    break

            # If we failed to get a full batch size, then drop or return partial
            if len(batch) < self.batch_size:
                break

            # return the batch
            yield batch
            batches_returned += 1

    def __len__(self):
        return self.num_batches


def contrastive_collate_fn(batch_of_dicts):
    """
    Splits a batch of dictionaries into separate lists for refs and pos.

    Args:
        batch_of_dicts (list): List of dictionaries returned by __getitem__.

    Returns:
        refs_data (tensor): Tensor of shape (N, ...) containing N references.
        pos_data (tensor): Tensor of shape (N, ...) containing N positives.
    Batch size affects the sampling very little, as we always sample exactly
    """
    refs = []
    pos = []

    ref_idx = []
    pos_idx = []
    for i, sample in enumerate(batch_of_dicts):
        if i % 2 == 0:
            refs.append(sample["image"])
            ref_idx.append(sample["idx"])
        else:
            # Assuming 'ref' key is used for both since __getitem__ returns only 'ref'
            pos.append(sample["image"])
            pos_idx.append(sample["idx"])

    all_data = torch.cat([torch.stack(refs), torch.stack(pos)], dim=0)
    all_idx = torch.cat([torch.tensor(ref_idx), torch.tensor(pos_idx)], dim=0)
    return {'image': all_data, 'idx': all_idx}
