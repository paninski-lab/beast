"""Dataset objects store images and augmentation pipeline."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typeguard import typechecked

from beast.data.types import ExampleDict

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@typechecked
class BaseDataset(torch.utils.data.Dataset):
    """Base dataset that contains images."""

    def __init__(self, data_dir: str | Path, imgaug_pipeline: Callable | None) -> None:
        """Initialize a dataset for autoencoder models.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_transform: imgaug transform pipeline to apply to images

        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')

        self.imgaug_pipeline = imgaug_pipeline
        # collect ALL png files in data_dir
        self.image_list = sorted(list(self.data_dir.rglob('*.png')))
        if len(self.image_list) == 0:
            raise ValueError(f'{self.data_dir} does not contain image data in png format')

        # send image to tensor, resize to canonical dimensions, and normalize
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int | list) -> ExampleDict | list[ExampleDict]:
        """Get item(s) from dataset.
        
        Parameters
        ----------
        idx: single index or list of indices
        
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
        # if 1 color channel, change to 3.
        image = Image.open(img_path).convert('RGB')
        if self.imgaug_pipeline is not None:
            # expands add batch dim for imgaug
            transformed_images = self.imgaug_pipeline(images=np.expand_dims(image, axis=0))
            # get rid of the batch dim
            transformed_images = transformed_images[0]
        else:
            transformed_images = image

        transformed_images = self.pytorch_transform(transformed_images)

        return ExampleDict(
            image=transformed_images,  # shape (3, img_height, img_width)
            video=img_path.parts[-2],
            idx=idx,
            image_path=str(img_path),
        )
