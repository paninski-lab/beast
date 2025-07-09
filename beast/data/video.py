from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from typeguard import typechecked

from beast.data.datasets import _IMAGENET_MEAN, _IMAGENET_STD
from beast.video import get_frames_from_idxs


@typechecked
class VideoFrameIterator:
    """Iterator that yields batches of video frames sequentially."""

    def __init__(
        self,
        video_file: str | Path,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize the video frame iterator.

        Parameters
        ----------
        video_file: absolute path to the video file
        batch_size: number of frames per batch

        """
        self.video_file = str(video_file)
        self.batch_size = batch_size

        # send image to tensor, resize to canonical dimensions, and normalize
        pytorch_transform_list = [
            # transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

        # open video capture
        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            raise ValueError(f'Cannot open video file: {self.video_file}')

        # get video metadata
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # initialize frame counter
        self.current_frame = 0

    def __iter__(self):
        """Return the iterator object."""
        return self

    def __next__(self) -> dict:
        """Get the next batch of frames.

        Returns
        -------
        dict
            'image': Batch of frames with shape (B, C, H, W)
            'video': path to video file
            'idx': indices of frames in batch
            'image_paths': null field

        Raises
        ------
        StopIteration
            When all frames have been processed

        """
        if self.current_frame >= self.total_frames:
            raise StopIteration

        # calculate batch indices
        start_idx = self.current_frame
        end_idx = min(self.current_frame + self.batch_size, self.total_frames)
        batch_indices = np.arange(start_idx, end_idx)

        # load frames for this batch
        frames = get_frames_from_idxs(video_file=None, idxs=batch_indices, cap=self.cap)

        # update current frame counter
        self.current_frame = end_idx

        # apply transforms
        batch_tensor = self.pytorch_transform(torch.from_numpy(frames).float() / 255.)

        # construct batch to pass to model.predict_step
        batch_dict = {
            'image': batch_tensor,
            'video': self.video_file,
            'idx': batch_indices,
            'image_path': None,
        }
        return batch_dict

    def __len__(self):
        """Return the number of batches."""
        return (self.total_frames + self.batch_size - 1) // self.batch_size

    def __del__(self):
        """Clean up the video capture object."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def reset(self):
        """Reset the iterator to the beginning of the video."""
        self.current_frame = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
