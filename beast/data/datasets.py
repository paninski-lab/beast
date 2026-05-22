"""Dataset objects store images and augmentation pipeline."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import imgaug.augmenters.size as _iaa_size
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from beast.data.types import ExampleDict
from beast.logging import log_step

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _patched_prevent(axis_size: int, crop_start: int, crop_end: int) -> tuple[int, ...]:
    """Monkey patch to fix imaug 0.4.2 compatability issue with numpy 2.x"""
    result = _iaa_size._prevent_zero_sizes_after_crops_(
        np.array([axis_size], dtype=np.int32),
        np.array([crop_start], dtype=np.int32),
        np.array([crop_end], dtype=np.int32),
    )
    return tuple(int(np.asarray(v).flat[0]) for v in result)


#  monkey patch to fix imaug 0.4.2 compatability issue with numpy 2.x
_iaa_size._prevent_zero_size_after_crop_ = _patched_prevent


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset that contains images."""

    def __init__(
        self,
        data_dir: str | Path,
        imgaug_pipeline: Callable | None,
        num_channels: int = 3,
    ) -> None:
        """Initialize a dataset for autoencoder models.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_pipeline: imgaug transform pipeline to apply to images
        num_channels: number of output channels; 1 loads as grayscale then converts to RGB,
            3 loads directly as RGB

        """
        if num_channels not in (1, 3):
            raise ValueError(f'num_channels must be 1 or 3, got {num_channels}')
        self.num_channels = num_channels
        log_step(f"BaseDataset.__init__ called with data_dir: {data_dir}", level='debug')
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')
        log_step(f"Data directory exists: {self.data_dir}", level='debug')

        self.imgaug_pipeline = imgaug_pipeline
        # collect ALL png files in data_dir
        scan_start = time.time()
        try:
            log_step(
                f"Starting to scan for PNG files in {self.data_dir}"
                ' (this may take a while for large directories)...',
                level='debug',
            )
            self.image_list = sorted(list(self.data_dir.rglob('*.png')))
            scan_duration = time.time() - scan_start
            log_step(
                f"Finished scanning. Found {len(self.image_list)} PNG files"
                f' in {scan_duration:.2f} seconds',
                level='debug',
            )
        except Exception as e:
            log_step(f"ERROR during file scanning: {e}", level='error')
            raise
        if len(self.image_list) == 0:
            raise ValueError(f'{self.data_dir} does not contain image data in png format')
        log_step(
            f"BaseDataset initialization complete with {len(self.image_list)} images",
            level='debug',
        )

        # send image to tensor, resize to canonical dimensions, and normalize
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

    def __len__(self) -> int:
        """Return number of images in the dataset."""
        return len(self.image_list)

    def __getitem__(self, idx: int | list) -> ExampleDict | list[ExampleDict]:
        """Get item(s) from dataset.

        Parameters
        ----------
        idx: single index or list  indices

        Returns
        -------
        Single ExampleDict or list of ExampleDict objects

        """
        # Handle batch of indices
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            # Handle single index
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> ExampleDict:
        """Get a single item from the dataset."""
        img_path = self.image_list[idx]

        # read image from file and apply transformations (if any)
        if self.num_channels == 1:
            image = Image.open(img_path).convert('L').convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
        if self.imgaug_pipeline is not None:
            # expands add batch dim for imgaug
            transformed_images = self.imgaug_pipeline(
                images=np.expand_dims(np.asarray(image), axis=0)
            )
            # get rid of the batch dim
            transformed_images = transformed_images[0]
        else:
            transformed_images = image

        transformed_tensor = cast(torch.Tensor, self.pytorch_transform(transformed_images))

        return ExampleDict(
            image=transformed_tensor,  # shape (3, img_height, img_width)
            video=img_path.parts[-2],
            idx=idx,
            image_path=str(img_path),
        )
