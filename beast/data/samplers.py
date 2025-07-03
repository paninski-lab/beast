import random

import numpy as np
import torch
from torch.utils.data import Sampler


def find_positive_candidates(ref_idx, idx_offset, max_idx, all_indices_set, used_indices):
    """
    Find valid positive candidates for a given reference index.
    
    Args:
        ref_idx: Reference index
        idx_offset: Offset for finding positive pairs
        max_idx: Maximum valid index
        all_indices_set: Set of all valid indices
        used_indices: Set of already used indices
        
    Returns:
        List of valid positive candidate indices
    """
    i_p_candidates = []
    for cand in [ref_idx - idx_offset, ref_idx + idx_offset]:
        if 0 <= cand < max_idx:
            if cand not in used_indices:
                if cand in all_indices_set:
                    i_p_candidates.append(cand)
    return i_p_candidates


def get_neighbor_indices(ref_idx, idx_offset):
    """
    Get all neighbor indices within the offset range.
    
    Args:
        ref_idx: Reference index
        idx_offset: Offset range
        
    Returns:
        List of neighbor indices
    """
    return [ref_idx + j for j in range(-idx_offset, idx_offset + 1)]

class ContrastBatchSampler(Sampler):
    """
    Custom batch sampler:
      - Each batch is exactly `batch_size = N` clips.
      - For each reference index i, we pick exactly one i_p from i +/- idx_offset 
        (and skip if none are valid).
      - The rest of the clips in that batch are chosen from "far away" indices.
    The __len__ of this sampler is #batches, i.e. total_clips // batch_size.
    """
    def __init__(self, dataset, batch_size, idx_offset=1, shuffle=True, drop_last=True):
        assert batch_size % 2 == 0, (
            "Batch size must be even to form (ref, pos) pairs."
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.idx_offset = idx_offset
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)  # total number of clips
        self.num_batches = self.num_samples // self.batch_size
        if not drop_last and self.num_samples % self.batch_size != 0:
            self.num_batches += 1  # if you want to allow incomplete batch
        # if dataset.frame_idx exists, use it to get the number of clip
        if hasattr(dataset, 'image_list') and dataset.frame_idx is not None:
            # frame_idx is a dict contain index: path; get the list of index
            try:
                self.all_indices = list(dataset.image_list.keys())
                self.max_idx = max(self.all_indices)
            except (TypeError, AttributeError):
                # If frame_idx exists but is not iterable (e.g., mock object), fall back to range
                self.all_indices = list(range(self.num_samples))
                self.max_idx = self.num_samples - 1
        else:
            # No frame_idx, use sequential indices
            self.all_indices = list(range(self.num_samples))
            self.max_idx = self.num_samples - 1
        # self.used_indices_freq = {i: 0 for i in self.all_indices}
        # self.used_indices = set()
        self.epoch = 0
        self.all_indices_set = set(self.all_indices)
        # self.idx_offset = 16 # 16 for vic-mae
    
    def __iter__(self):
        self.epoch += 1
        
        if self.shuffle:
            random.shuffle(self.all_indices)
        
        used = set()
        batches_returned = 0
        
        # We'll keep sampling until we form all possible batches
        idx_cursor = 0
        
        while batches_returned < self.num_batches:
            batch = []
            # Keep pairing up references and positives until we have batch_size
            while len(batch) < self.batch_size:
                
                # If we run out of "unused" indices, we break early
                # (especially if drop_last == True)
                while idx_cursor < self.num_samples and self.all_indices[idx_cursor] in used:
                    idx_cursor += 1
                if idx_cursor >= self.num_samples:
                    break
                
                i = self.all_indices[idx_cursor]
                
                # Use helper function to find positive candidates
                i_p_candidates = find_positive_candidates(
                    i, self.idx_offset, self.max_idx, self.all_indices_set, used
                )
                
                if not i_p_candidates:
                    # if no valid positives, skip this ref
                    # print(f"Skipping ref {i} as no valid positives found.")
                    idx_cursor += 1
                    continue
                
                # choose a random positive
                pos_idx = int(np.random.uniform(0, len(i_p_candidates)))
                # uniform sampling to choose a positive
                i_p = i_p_candidates[pos_idx]
                
                # Now we have a reference i, a positive i_p
                # Mark them as used
                # Use helper function to get neighbor indices
                neighbors = get_neighbor_indices(i, self.idx_offset)
                used.update(neighbors)
                batch.extend([i, i_p])
                
                idx_cursor += 1
                if idx_cursor >= self.num_samples:
                    break
            
            # If we failed to get a full batch size, then drop or return partial
            if len(batch) < self.batch_size:
                if self.drop_last:
                    break  # discard partial batch
                # else fill the remainder randomly from unused "far" indices
                needed = self.batch_size - len(batch)
                far_candidates = [x for x in self.all_indices if x not in used]
                if len(far_candidates) < needed:
                    # can't fill
                    break
                chosen = random.sample(far_candidates, needed)
                used.update(chosen)
                batch.extend(chosen)
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
            pos.append(sample["image"])  # Assuming 'ref' key is used for both
                                       # since __getitem__ returns only 'ref'
            pos_idx.append(sample["idx"])

    all_data = torch.cat([torch.stack(refs), torch.stack(pos)], dim=0)
    all_idx = torch.cat([torch.tensor(ref_idx), torch.tensor(pos_idx)], dim=0)
    return {'image': all_data, 'idx': all_idx}
