import os
import random
import sys
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.utilities import rank_zero_only
from typeguard import typechecked

import beast
from beast.data.augmentations import imgaug_pipeline
from beast.data.datamodules import BaseDataModule
from beast.data.datasets import BaseDataset


def _debug_log(msg: str, flush: bool = True):
    """Debug logging function with timestamp."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] DEBUG: {msg}", flush=flush)


@typechecked
def reset_seeds(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@rank_zero_only
@typechecked
def pretty_print_config(config: dict) -> None:
    print('config file:')
    for key, val in config.items():
        print('--------------------')
        print(f'{key} parameters')
        print('--------------------')
        if isinstance(val, dict):
            for k, v in val.items():
                print(f'{k}: {v}')
        else:
            print(f'{val}')
        print()
    print('\n\n')


@typechecked
def train(config: dict, model, output_dir: str | Path):

    # Only print from rank 0
    if rank_zero_only.rank == 0:
        _debug_log("Entering train() function")
        print(f'output directory: {output_dir}')
        print(f'model type: {type(model)}')

    # reset all seeds
    if rank_zero_only.rank == 0:
        _debug_log("Resetting seeds")
    reset_seeds(seed=0)

    # record beast version
    if rank_zero_only.rank == 0:
        _debug_log("Recording beast version")
    config['model']['beast_version'] = beast.version

    if rank_zero_only.rank == 0:
        _debug_log("Printing config")
    pretty_print_config(config)

    # ----------------------------------------------------------------------------------
    # Set up data objects
    # ----------------------------------------------------------------------------------

    # imgaug transform
    if rank_zero_only.rank == 0:
        _debug_log("Setting up imgaug pipeline")
    pipe_params = config.get('training', {}).get('imgaug', 'none')
    if isinstance(pipe_params, str):
        from beast.data.augmentations import expand_imgaug_str_to_dict
        pipe_params = expand_imgaug_str_to_dict(pipe_params)
    imgaug_pipeline_ = imgaug_pipeline(pipe_params)
    if rank_zero_only.rank == 0:
        _debug_log("Imgaug pipeline created")

    # dataset
    if rank_zero_only.rank == 0:
        _debug_log(f"Creating BaseDataset with data_dir: {config['data']['data_dir']}")
        _debug_log("WARNING: This may take a long time if data directory is large (scanning for PNG files)")
    dataset = BaseDataset(
        data_dir=config['data']['data_dir'],
        imgaug_pipeline=imgaug_pipeline_,
    )
    if rank_zero_only.rank == 0:
        _debug_log(f"BaseDataset created. Found {len(dataset)} images")

    # datamodule; breaks up dataset into train/val/test
    if rank_zero_only.rank == 0:
        _debug_log("Creating BaseDataModule")
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
        _debug_log("BaseDataModule created")

    # update number of training steps (for learning rate scheduler with step information)
    if rank_zero_only.rank == 0:
        _debug_log("Calculating training steps")
    num_epochs = config['training']['num_epochs']
    steps_per_epoch = int(np.ceil(
        len(datamodule.train_dataset)
        / config['training']['train_batch_size']
        / config['training']['num_gpus']
        / config['training']['num_nodes']
    ))
    model.config['optimizer']['steps_per_epoch'] = steps_per_epoch
    model.config['optimizer']['total_steps'] = steps_per_epoch * num_epochs
    if rank_zero_only.rank == 0:
        _debug_log(f"Training steps calculated: {steps_per_epoch} steps/epoch, {num_epochs} epochs")

    # ----------------------------------------------------------------------------------
    # Save configuration in output directory
    # ----------------------------------------------------------------------------------
    # Done before training; files will exist even if script dies prematurely.

    # save config file
    if rank_zero_only.rank == 0:
        _debug_log(f"Saving config to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_config_file = Path(output_dir) / 'config.yaml'
    with open(dest_config_file, 'w') as file:
        yaml.dump(config, file)
    if rank_zero_only.rank == 0:
        _debug_log("Config saved")

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    if rank_zero_only.rank == 0:
        _debug_log("Creating TensorBoardLogger")
    logger = pl.loggers.TensorBoardLogger('tb_logs', name='')
    if rank_zero_only.rank == 0:
        _debug_log("TensorBoardLogger created")

    # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
    if rank_zero_only.rank == 0:
        _debug_log("Setting up callbacks")
    callbacks = get_callbacks(
        lr_monitor=True,
        ckpt_every_n_epochs=config['training'].get('ckpt_every_n_epochs', None),
    )
    if rank_zero_only.rank == 0:
        _debug_log(f"Callbacks created: {len(callbacks)} callbacks")

    # initialize to Trainer defaults. Note max_steps defaults to -1.
    min_epochs = config['training']['num_epochs']
    max_epochs = min_epochs

    # our custom sampler does not play nice with DDP
    if config['model']['model_params'].get('use_infoNCE', False):
        use_distributed_sampler = False
    else:
        use_distributed_sampler = True

    if rank_zero_only.rank == 0:
        _debug_log("Creating PyTorch Lightning Trainer")
        _debug_log(f"  - accelerator: gpu")
        _debug_log(f"  - devices: {config['training']['num_gpus']}")
        _debug_log(f"  - num_nodes: {config['training']['num_nodes']}")
        _debug_log(f"  - max_epochs: {max_epochs}")
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
        _debug_log("Trainer created")

    # train model!
    if rank_zero_only.rank == 0:
        _debug_log("About to call trainer.fit() - this may hang here if there are issues with data loading or GPU setup")
    trainer.fit(model=model, datamodule=datamodule)
    if rank_zero_only.rank == 0:
        _debug_log("trainer.fit() completed")

    # when devices > 0, lightning creates a process per device.
    # kill processes other than the main process, otherwise they all go forward.
    if not trainer.is_global_zero:
        sys.exit(0)

    # return trained model
    return model


@typechecked
def get_callbacks(
    checkpointing: bool = True,
    lr_monitor: bool = True,
    ckpt_every_n_epochs: int | None = None,
) -> list:

    callbacks = []

    if lr_monitor:
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

    # always save out best model
    if checkpointing:
        ckpt_best_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filename='{epoch}-{step}-best',
        )
        callbacks.append(ckpt_best_callback)

    if ckpt_every_n_epochs:
        # if ckpt_every_n_epochs is not None, save separate checkpoint files
        ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor=None,
            every_n_epochs=ckpt_every_n_epochs,
            save_top_k=-1,
        )
        callbacks.append(ckpt_callback)

    return callbacks
