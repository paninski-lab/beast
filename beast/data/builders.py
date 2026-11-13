"""Shared builders that construct datamodules from a full experiment config dict.

Model packages call these from their ``build_datamodule`` function (see
``beast/models/<name>/<name>_train.py``), passing any model-specific options.
"""

from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only

from beast.data.augmentations import expand_imgaug_str_to_dict, imgaug_pipeline
from beast.data.datamodules import BaseDataModule, MultiViewDataModule
from beast.data.datasets import BaseDataset
from beast.logging import log_step


def build_multiview_datamodule(
    config: dict,
    use_mask: bool = False,
) -> pl.LightningDataModule:
    """Build the multi-view datamodule used by the erayzer / beast3d model classes.

    Parameters
    ----------
    config: full experiment configuration dict
    use_mask: whether to load ground-truth foreground masks (supervises the rendered alpha)

    Returns
    -------
    set-up datamodule

    """
    if rank_zero_only.rank == 0:
        log_step('Creating MultiViewDataModule', level='debug')
    datamodule = MultiViewDataModule(
        data_dir=config['data']['data_dir'],
        image_size=config['model']['image_tokenizer']['image_size'],
        train_batch_size=config['training']['train_batch_size'],
        val_batch_size=config['training']['val_batch_size'],
        train_fraction=config['training']['train_probability'],
        train_frames=config['training']['train_frames'],
        num_workers=config['training']['num_workers'],
        seed=config['training']['seed'],
        use_mask=use_mask,
    )
    datamodule.setup()
    if rank_zero_only.rank == 0:
        log_step('MultiViewDataModule created', level='debug')
    return datamodule


def build_singleview_datamodule(
    config: dict,
    sampler_kind: Literal['none', 'contrastive', 'triplet'] | None = None,
) -> pl.LightningDataModule:
    """Build the single-view image datamodule.

    Parameters
    ----------
    config: full experiment configuration dict
    sampler_kind: batch sampler to use; if None, 'contrastive' when
        ``model_params.use_infoNCE`` is set and 'none' otherwise

    Returns
    -------
    datamodule

    """
    # imgaug transform
    if rank_zero_only.rank == 0:
        log_step('Setting up imgaug pipeline', level='debug')
    pipe_params = config['training']['imgaug']
    if isinstance(pipe_params, str):
        pipe_params = expand_imgaug_str_to_dict(pipe_params)  # type: ignore[arg-type]
    imgaug_pipeline_ = imgaug_pipeline(pipe_params)
    if rank_zero_only.rank == 0:
        log_step('Imgaug pipeline created', level='debug')

    # dataset
    dataset = BaseDataset(
        data_dir=config['data']['data_dir'],
        imgaug_pipeline=imgaug_pipeline_,
        num_channels=config['model']['model_params']['num_channels'],
    )

    # datamodule; breaks up dataset into train/val/test
    if rank_zero_only.rank == 0:
        log_step('Creating BaseDataModule', level='debug')
    if sampler_kind is None:
        use_contrastive = config['model']['model_params'].get('use_infoNCE', False)
        sampler_kind = 'contrastive' if use_contrastive else 'none'
    positive_window = 1000  # only used by the triplet sampler
    if sampler_kind == 'triplet':
        positive_window = config['model']['model_params']['positive_window']
    datamodule = BaseDataModule(
        dataset=dataset,
        train_batch_size=config['training']['train_batch_size'],
        val_batch_size=config['training']['val_batch_size'],
        test_batch_size=config['training']['test_batch_size'],
        sampler_kind=sampler_kind,
        positive_window=positive_window,
        num_workers=config['training']['num_workers'],
        train_probability=config['training']['train_probability'],
        val_probability=config['training']['val_probability'],
        train_frames=config['training']['train_frames'],
        seed=config['training']['seed'],
    )
    if rank_zero_only.rank == 0:
        log_step('BaseDataModule created', level='debug')

    return datamodule
