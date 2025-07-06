import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def extract_anchor_indices(image_list):
    """
    Extract anchor indices from image paths that have valid neighboring frames.
    
    Args:
        image_list: List of image paths
        
    Returns:
        List of indices that can serve as anchors (have valid neighbors)
    """
    anchor_indices = []
    
    # Parse each image path to extract video and frame information
    frame_info = []
    for idx, img_path in enumerate(image_list):
        path = Path(img_path)
        # Extract video name (parent directory) and frame number from filename
        video_name = path.parent.name
        # Try to extract frame number from filename (assuming format like frame001.png, 001.png, etc.)
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
        has_prev = (i > 0 and 
                   frame_info[i-1]['video'] == frame['video'] and 
                   frame_info[i-1]['frame_num'] == frame['frame_num'] - 1)
        
        # Check if next frame exists and is from same video
        has_next = (i < len(frame_info) - 1 and 
                   frame_info[i+1]['video'] == frame['video'] and 
                   frame_info[i+1]['frame_num'] == frame['frame_num'] + 1)
        
        # Add to anchor indices if it has at least one valid neighbor
        if has_prev or has_next:
            anchor_indices.append(frame['idx'])
    
    return anchor_indices


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
        
        # Use sequential indices for all samples
        self.all_indices = list(range(self.num_samples))
        self.max_idx = self.num_samples - 1
        

        image_list = dataset.dataset.image_list
        self.anchor_indices = extract_anchor_indices(image_list)
        # only remain anchor indices that are in the dataset.indices
        self.anchor_indices = [i for i in self.anchor_indices if i in dataset.indices[idx_offset:-idx_offset]]
        
        self.epoch = 0
        self.all_indices_set = set(self.all_indices)
    
    def __iter__(self):
        self.epoch += 1
        
        if self.shuffle:
            random.shuffle(self.anchor_indices)
        
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
                while idx_cursor < len(self.anchor_indices) and self.anchor_indices[idx_cursor] in used:
                    idx_cursor += 1
                if idx_cursor >= len(self.anchor_indices):
                    break
                
                i = self.anchor_indices[idx_cursor]
                
                # choose a random positive
                i_p_offset = np.random.choice([-1, 1])
                i_p = i + i_p_offset
                
                # Now we have a reference i, a positive i_p
                # Mark them as used
                neighbor_indices = get_neighbor_indices(i, self.idx_offset)
                used.update(neighbor_indices)
                batch.extend([i, i_p])
                
                idx_cursor += 1
                if idx_cursor >= len(self.anchor_indices):
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
