"""Training loop and callback setup for BEAST models."""

import logging
import os
import random
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only

import beast
from beast.data.augmentations import imgaug_pipeline
from beast.data.datamodules import BaseDataModule
from beast.data.datasets import BaseDataset
from beast.logging import log_step
from beast.models.base import BaseLightningModel

_logger = logging.getLogger(__name__)


def reset_seeds(seed: int = 0) -> None:
    """Reset all random seeds for reproducible training.

    Parameters
    ----------
    seed: seed value for all random number generators

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@rank_zero_only
def pretty_print_config(config: dict) -> None:
    """Log the config dict section by section."""
    _logger.info('config file:')
    for key, val in config.items():
        _logger.info('--------------------')
        _logger.info(f'{key} parameters')
        _logger.info('--------------------')
        if isinstance(val, dict):
            for k, v in val.items():
                _logger.info(f'{k}: {v}')
        else:
            _logger.info(f'{val}')
    _logger.info('')


def train(config: dict, model: BaseLightningModel, output_dir: str | Path) -> BaseLightningModel:
    """Set up data, trainer, and callbacks, then train the model.

    Parameters
    ----------
    config: full experiment configuration dict
    model: initialized Lightning model to train
    output_dir: directory where config, checkpoints, and logs are saved

    Returns
    -------
    trained model

    """
    output_dir = Path(output_dir)

    # Only print from rank 0
    if rank_zero_only.rank == 0:
        log_step("Entering train() function", level='debug')
        _logger.info(f'output directory: {output_dir}')
        _logger.info(f'model type: {type(model)}')

    # reset all seeds
    if rank_zero_only.rank == 0:
        log_step("Resetting seeds", level='debug')
    reset_seeds(seed=0)

    # record beast version
    if rank_zero_only.rank == 0:
        log_step("Recording beast version", level='debug')
    config['model']['beast_version'] = beast.version

    if rank_zero_only.rank == 0:
        log_step("Printing config", level='debug')
    pretty_print_config(config)

    # ----------------------------------------------------------------------------------
    # Set up data objects
    # ----------------------------------------------------------------------------------

    # imgaug transform
    if rank_zero_only.rank == 0:
        log_step("Setting up imgaug pipeline", level='debug')
    pipe_params = config.get('training', {}).get('imgaug', 'none')
    if isinstance(pipe_params, str):
        from beast.data.augmentations import expand_imgaug_str_to_dict
        pipe_params = expand_imgaug_str_to_dict(pipe_params)  # type: ignore[arg-type]
    imgaug_pipeline_ = imgaug_pipeline(pipe_params)
    if rank_zero_only.rank == 0:
        log_step("Imgaug pipeline created", level='debug')

    # dataset
    dataset = BaseDataset(
        data_dir=config['data']['data_dir'],
        imgaug_pipeline=imgaug_pipeline_,
        num_channels=config['model']['model_params'].get('num_channels', 3),
    )

    # datamodule; breaks up dataset into train/val/test
    if rank_zero_only.rank == 0:
        log_step("Creating BaseDataModule", level='debug')
    datamodule = BaseDataModule(
        dataset=dataset,
        train_batch_size=config['training']['train_batch_size'],
        val_batch_size=config['training']['val_batch_size'],
        test_batch_size=config['training']['test_batch_size'],
        use_sampler=config['model']['model_params'].get('use_infoNCE', False),
        num_workers=config['training']['num_workers'],
        train_probability=config['training'].get('train_probability', 0.95),
        val_probability=config['training'].get('val_probability', 0.05),
        seed=config['training']['seed'],
    )
    if rank_zero_only.rank == 0:
        log_step("BaseDataModule created", level='debug')

    # update number of training steps (for learning rate scheduler with step information)
    if rank_zero_only.rank == 0:
        log_step("Calculating training steps", level='debug')
    num_epochs = config['training']['num_epochs']
    train_size = int(np.floor(config['training'].get('train_probability', 0.95) * len(dataset)))
    steps_per_epoch = int(np.ceil(
        train_size
        / config['training']['train_batch_size']
        / config['training']['num_gpus']
        / config['training']['num_nodes']
    ))
    model.config['optimizer']['steps_per_epoch'] = steps_per_epoch
    model.config['optimizer']['total_steps'] = steps_per_epoch * num_epochs
    if rank_zero_only.rank == 0:
        log_step(
            f"Training steps calculated: {steps_per_epoch} steps/epoch, {num_epochs} epochs",
            level='debug',
        )

    # ----------------------------------------------------------------------------------
    # Save configuration in output directory
    # ----------------------------------------------------------------------------------
    # Done before training; files will exist even if script dies prematurely.

    # save config file
    if rank_zero_only.rank == 0:
        log_step(f"Saving config to {output_dir}", level='debug')
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_config_file = Path(output_dir) / 'config.yaml'
    with open(dest_config_file, 'w') as file:
        yaml.dump(config, file)
    if rank_zero_only.rank == 0:
        log_step("Config saved", level='debug')

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    if rank_zero_only.rank == 0:
        log_step("Creating TensorBoardLogger", level='debug')
    logger = pl_loggers.TensorBoardLogger('tb_logs', name='')
    if rank_zero_only.rank == 0:
        log_step("TensorBoardLogger created", level='debug')

    # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
    if rank_zero_only.rank == 0:
        log_step("Setting up callbacks", level='debug')
    callbacks = get_callbacks(
        lr_monitor=True,
        ckpt_every_n_epochs=config['training'].get('ckpt_every_n_epochs', None),
    )
    if rank_zero_only.rank == 0:
        log_step(f"Callbacks created: {len(callbacks)} callbacks", level='debug')

    # initialize to Trainer defaults. Note max_steps defaults to -1.
    min_epochs = config['training']['num_epochs']
    max_epochs = min_epochs

    # our custom sampler does not play nice with DDP
    if config['model']['model_params'].get('use_infoNCE', False):
        use_distributed_sampler = False
    else:
        use_distributed_sampler = True

    if rank_zero_only.rank == 0:
        log_step("Creating PyTorch Lightning Trainer", level='debug')
        log_step("  - accelerator: gpu", level='debug')
        log_step(f"  - devices: {config['training']['num_gpus']}", level='debug')
        log_step(f"  - num_nodes: {config['training']['num_nodes']}", level='debug')
        log_step(f"  - max_epochs: {max_epochs}", level='debug')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config['training']['num_gpus'],
        num_nodes=config['training']['num_nodes'],
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        check_val_every_n_epoch=config['training'].get('check_val_every_n_epoch', 1),
        log_every_n_steps=config['training'].get('log_every_n_steps', 10),
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=config['optimizer'].get('accumulate_grad_batches', 1),
        sync_batchnorm=True,
        use_distributed_sampler=use_distributed_sampler,
    )
    if rank_zero_only.rank == 0:
        log_step("Trainer created", level='debug')

    # train model!
    if rank_zero_only.rank == 0:
        log_step(
            "About to call trainer.fit() - this may hang here if there are issues"
            ' with data loading or GPU setup',
            level='debug',
        )
    trainer.fit(model=model, datamodule=datamodule)
    if rank_zero_only.rank == 0:
        log_step("trainer.fit() completed", level='debug')

    # when devices > 0, lightning creates a process per device.
    # kill processes other than the main process, otherwise they all go forward.
    if not trainer.is_global_zero:
        sys.exit(0)

    # return trained model
    return model


def get_callbacks(
    checkpointing: bool = True,
    lr_monitor: bool = True,
    ckpt_every_n_epochs: int | None = None,
) -> list:
    """Build list of Lightning callbacks for training.

    Parameters
    ----------
    checkpointing: whether to save the best checkpoint by validation loss
    lr_monitor: whether to log learning rate each epoch
    ckpt_every_n_epochs: if set, also save a checkpoint every n epochs

    Returns
    -------
    list of configured Lightning callbacks

    """
    callbacks = []

    if lr_monitor:
        lr_monitor_cb = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor_cb)

    # always save out best model
    if checkpointing:
        ckpt_best_callback = pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filename='{epoch}-{step}-best',
        )
        callbacks.append(ckpt_best_callback)

    if ckpt_every_n_epochs:
        # if ckpt_every_n_epochs is not None, save separate checkpoint files
        ckpt_callback = pl_callbacks.ModelCheckpoint(
            monitor=None,
            every_n_epochs=ckpt_every_n_epochs,
            save_top_k=-1,
        )
        callbacks.append(ckpt_callback)

    return callbacks
