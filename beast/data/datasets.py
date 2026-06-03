"""Dataset objects store images and augmentation pipeline."""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import imgaug.augmenters.size as _iaa_size
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from beast.data.types import ExampleDict, MultiViewExampleDict
from beast.geometry.camera import (
    intrinsics_to_fxfycxcy,
    normalize_camera_sequence,
    scale_intrinsics,
    w2c_to_c2w,
)
from beast.logging import log_step

_logger = logging.getLogger(__name__)

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


class MultiViewDataset(torch.utils.data.Dataset):
    """Multi-view dataset for BEAST3D training.

    Each item is a single time point from one session, containing synchronized
    images from all available cameras along with their camera parameters.

    The dataset reads from the output directory produced by ``beast extract_3d``.
    Each item returns:

    - ``image``: float32 tensor of shape ``(V, 3, H, W)`` in ``[0, 1]``.
    - ``c2w``: float32 tensor of shape ``(V, 4, 4)`` camera-to-world matrices.
    - ``fxfycxcy``: float32 tensor of shape ``(V, 4)`` intrinsics at ``image_size``
      resolution in absolute pixel units.
    - ``view_names``: list of V camera name strings.
    - ``video_id``, ``frame_id``: metadata strings.
    - ``input_mask`` (optional): float32 tensor of shape ``(V, 1, H, W)`` binary
      foreground masks, only present when ``use_mask=True``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int,
        mode: str = 'train',
        use_mask: bool = False,
        normalize_cameras: bool = True,
        frame_ids: list[str] | None = None,
    ) -> None:
        """Initialize the multi-view dataset.

        Parameters
        ----------
        data_dir: path to the ``dataset/`` directory produced by ``beast extract_3d``.
        image_size: square size to resize images to (both height and width).
        mode: ``'train'`` shuffles view order randomly; ``'test'`` uses sorted order.
        use_mask: if True, load binary segmentation masks alongside images.
        normalize_cameras: if True, re-center cameras so camera 0 is at the origin
            and scale translations so the mean camera distance is 1.
        frame_ids: optional pre-selected list of ``'{video_id}/{frame_filename}'``
            strings; if None, all frames from all sessions are used.

        Raises
        ------
        ValueError
            if data_dir does not exist, contains no sessions, or use_mask is True
            but a mask file is missing for any frame.

        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')

        self.image_size = image_size
        self.mode = mode
        self.use_mask = use_mask
        self.normalize_cameras = normalize_cameras

        info_path = self.data_dir / 'info.json'
        if not info_path.exists():
            raise ValueError(f'info.json not found in {self.data_dir}')
        with open(info_path) as f:
            info = json.load(f)
        self.available_views: list[str] = sorted(info['available_views'])

        if frame_ids is not None:
            self.unique_frame_ids = frame_ids
        else:
            csv_files = sorted(self.data_dir.rglob('selected_frames.csv'))
            if not csv_files:
                raise ValueError(f'No selected_frames.csv files found under {self.data_dir}')
            ids: list[str] = []
            for csv_path in csv_files:
                video_id = csv_path.parent.name
                frames = csv_path.read_text().splitlines()
                ids.extend(f'{video_id}/{f}' for f in frames if f)
            self.unique_frame_ids = sorted(set(ids))

        if not self.unique_frame_ids:
            raise ValueError(f'{self.data_dir} contains no multi-view frame data')

        self.img_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

        _logger.info(
            f'MultiViewDataset: {len(self.unique_frame_ids)} frames × '
            f'{len(self.available_views)} views  (mode={mode}, use_mask={use_mask})'
        )

    def __len__(self) -> int:
        """Return number of frames in the dataset."""
        return len(self.unique_frame_ids)

    def __getitem__(self, idx: int) -> MultiViewExampleDict:
        """Return all camera views for a single time point.

        Parameters
        ----------
        idx: index into unique_frame_ids.

        Returns
        -------
        MultiViewExampleDict with stacked tensors for all V cameras.

        """
        return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> MultiViewExampleDict:
        """Load images and camera params for one (session, frame) pair."""
        unique_frame_id = self.unique_frame_ids[idx]
        video_id, frame_id = unique_frame_id.split('/', 1)

        images: list[torch.Tensor] = []
        extrinsics: list[torch.Tensor] = []
        fxfycxcy_list: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for view in self.available_views:
            img_path = self.data_dir / video_id / view / frame_id
            cam_path = img_path.with_suffix('.npy')

            image = Image.open(img_path).convert('RGB')
            images.append(self.img_transform(image))

            if not cam_path.exists():
                raise FileNotFoundError(
                    f'camera file missing for {img_path.name} '
                    f'(expected {cam_path})'
                )
            cam_info = np.load(cam_path, allow_pickle=True).item()
            K = torch.from_numpy(cam_info['intrinsics']).float()
            scale_w = self.image_size / cam_info['width']
            scale_h = self.image_size / cam_info['height']
            K = scale_intrinsics(K, scale_w, scale_h)
            fxfycxcy_list.append(intrinsics_to_fxfycxcy(K))
            extrinsics.append(torch.from_numpy(cam_info['extrinsics']).float())

            if self.use_mask:
                mask_filename = frame_id.replace('img', 'mask', 1)
                mask_path = self.data_dir / video_id / view / mask_filename
                mask = Image.open(mask_path).convert('L')
                mask_tensor = self.mask_transform(mask)
                masks.append((mask_tensor > 0.5).float())

        images_t = torch.stack(images)           # (V, 3, H, W)
        w2c_t = torch.stack(extrinsics)          # (V, 4, 4)
        fxfycxcy_t = torch.stack(fxfycxcy_list)  # (V, 4)

        if self.normalize_cameras:
            c2w_t = normalize_camera_sequence(w2c_t)
        else:
            c2w_t = w2c_to_c2w(w2c_t)

        view_names = list(self.available_views)

        if self.mode == 'train':
            perm = torch.randperm(len(self.available_views))
            images_t = images_t[perm]
            c2w_t = c2w_t[perm]
            fxfycxcy_t = fxfycxcy_t[perm]
            view_names = [view_names[i] for i in perm.tolist()]
            if self.use_mask:
                masks_t = torch.stack(masks)[perm]
        else:
            if self.use_mask:
                masks_t = torch.stack(masks)

        result: MultiViewExampleDict = {
            'image': images_t,
            'c2w': c2w_t,
            'fxfycxcy': fxfycxcy_t,
            'view_names': view_names,
            'video_id': video_id,
            'frame_id': frame_id,
        }
        if self.use_mask:
            result['input_mask'] = masks_t
        return result
