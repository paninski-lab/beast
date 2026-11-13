"""Training loop and callback setup for BEAST models."""

import os
import random
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only

import beast
from beast.callbacks.factory import get_callbacks
from beast.logging import log_step
from beast.models.base import BaseLightningModel


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
    log_step('config file:')
    for key, val in config.items():
        log_step('--------------------')
        log_step(f'{key} parameters')
        log_step('--------------------')
        if isinstance(val, dict):
            for k, v in val.items():
                log_step(f'{k}: {v}')
        else:
            log_step(f'{val}')
    log_step('')


def train(
    config: dict,
    model: BaseLightningModel,
    datamodule: pl.LightningDataModule,
    output_dir: str | Path,
) -> BaseLightningModel:
    """Set up the trainer and callbacks, then train the model.

    Parameters
    ----------
    config: full experiment configuration dict
    model: initialized Lightning model to train
    datamodule: set-up datamodule providing the train and val data loaders; must expose a
        ``use_distributed_sampler`` attribute
    output_dir: directory where config, checkpoints, and logs are saved

    Returns
    -------
    trained model

    """
    output_dir = Path(output_dir)

    # Only print from rank 0
    if rank_zero_only.rank == 0:
        log_step("Entering train() function", level='debug')
        log_step(f'output directory: {output_dir}')
        log_step(f'model type: {type(model)}')

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
    # Save configuration in output directory
    # ----------------------------------------------------------------------------------
    # Done before training; files will exist even if script dies prematurely.

    # save config file
    if rank_zero_only.rank == 0:
        log_step(f'Saving config to {output_dir}', level='debug')
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_config_file = Path(output_dir) / 'config.yaml'
    with open(dest_config_file, 'w') as file:
        yaml.dump(config, file)
    if rank_zero_only.rank == 0:
        log_step('Config saved', level='debug')
    # stashed after saving so it is not written to config.yaml; read by the models'
    # configure_callbacks (e.g. to locate where validation visuals are saved)
    model.config['training']['output_dir'] = str(output_dir)

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    if rank_zero_only.rank == 0:
        log_step('Creating TensorBoardLogger', level='debug')
    logger = pl_loggers.TensorBoardLogger('tb_logs', name='')
    if rank_zero_only.rank == 0:
        log_step('TensorBoardLogger created', level='debug')

    # learning rate monitoring and model checkpointing; model-specific callbacks come from
    # model.configure_callbacks
    if rank_zero_only.rank == 0:
        log_step('Setting up callbacks', level='debug')
    callbacks = get_callbacks(
        lr_monitor=True,
        ckpt_every_n_epochs=config['training']['ckpt_every_n_epochs'],
    )
    if rank_zero_only.rank == 0:
        log_step(f'Callbacks created: {len(callbacks)} callbacks', level='debug')

    if rank_zero_only.rank == 0:
        log_step('Creating PyTorch Lightning Trainer', level='debug')
        log_step('  - accelerator: gpu', level='debug')
        log_step(f"  - devices: {config['training']['num_gpus']}", level='debug')
        log_step(f"  - num_nodes: {config['training']['num_nodes']}", level='debug')
        log_step(f"  - num_epochs: {config['training']['num_epochs']}", level='debug')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config['training']['num_gpus'],
        num_nodes=config['training']['num_nodes'],
        precision=config['training']['precision'],
        check_val_every_n_epoch=config['training']['check_val_every_n_epoch'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=config['optimizer']['accumulate_grad_batches'],
        sync_batchnorm=True,
        use_distributed_sampler=datamodule.use_distributed_sampler,  # type: ignore[attr-defined]
        max_epochs=config['training']['num_epochs'],
        min_epochs=config['training']['num_epochs'],
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
