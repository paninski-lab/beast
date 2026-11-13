"""Base Lightning module shared by all BEAST model architectures."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, OneCycleLR

from beast.logging import log_step


class BaseLightningModel(ABC, pl.LightningModule):
    """Base Lightning Module that specific model architectures will inherit from."""

    def __init__(self, config: dict) -> None:
        """Initialize with config."""

        super().__init__()

        if rank_zero_only.rank == 0:
            log_step(f'Initializing a {self._get_name()} instance.')

        self.config = config
        self.seed = config['model']['seed']
        torch.manual_seed(self.seed)
        self.return_reconstructions = True

        self.save_hyperparameters(config)
        # Child classes implement architecture setup

    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build and return the learning rate scheduler specified in config.

        Parameters
        ----------
        optimizer: the optimizer to schedule

        Returns
        -------
        configured learning rate scheduler

        """
        name = self.config['optimizer']['scheduler']
        scheduler: torch.optim.lr_scheduler.LRScheduler
        if name == 'step':
            # define a scheduler that reduces the base learning rate at predefined steps
            scheduler = MultiStepLR(
                optimizer=optimizer,
                milestones=self.config['optimizer']['steps'],
                gamma=self.config['optimizer']['gamma'],
            )
        elif name == 'cosine':
            # base learning rate is per 256 images of effective batch (linear scaling rule)
            max_lr = self.config['optimizer']['lr']
            if self.config['optimizer']['scale_lr_by_batch']:
                global_batch_size = (
                    self.config['training']['train_batch_size']
                    * self.config['training']['num_gpus']
                    * self.config['training']['num_nodes']
                    * self.config['optimizer']['accumulate_grad_batches']
                )
                max_lr = max_lr * global_batch_size / 256
            scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=max_lr,
                epochs=self.config['training']['num_epochs'],
                steps_per_epoch=1,  # update learning rate after every batch of data
                pct_start=self.config['optimizer']['warmup_pct'],
                anneal_strategy='cos',
                div_factor=self.config['optimizer']['div_factor'],
                final_div_factor=self.config['optimizer']['final_div_factor'],
            )
        elif name == 'constant':
            # fixed learning rate for the whole run; never scaled by batch size
            scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda _epoch: 1.0)
        else:
            raise NotImplementedError(f'{name} scheduler is not yet implemented')
        return scheduler

    def get_parameters(self) -> Iterator:
        """Return an iterator over trainable model parameters."""
        params = filter(lambda p: p.requires_grad, self.parameters())
        return params

    def configure_optimizers(self) -> dict:
        """Select optimizer, lr scheduler, and metric for monitoring."""

        # get trainable params
        params = self.get_parameters()

        # init optimizer
        optimizer = self.config['optimizer']['type']
        learning_rate = self.config['optimizer']['lr']
        betas = (self.config['optimizer']['beta1'], self.config['optimizer']['beta2'])
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, lr=learning_rate, betas=betas)
        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                betas=betas,
                weight_decay=self.config['optimizer']['wd'],
            )
        else:
            raise NotImplementedError(f'{optimizer} optimizer is not yet implemented')

        # get learning rate scheduler
        scheduler = self.get_scheduler(optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }

    def evaluate_batch(
        self,
        batch_dict: dict,
        stage: Literal['train', 'val', 'test'] | None = None,
    ) -> torch.Tensor:
        """Compute and log the losses on a batch of labeled data."""

        # forward pass; collected true and predicted heatmaps, keypoints
        data_dict = self.get_model_outputs(batch_dict=batch_dict)

        # compute and log loss on labeled data
        loss, log_list = self.compute_loss(stage=stage, **data_dict)

        if stage:
            # logging with sync_dist=True will average the metric across GPUs in
            # multi-GPU training. Performance overhead was found negligible.

            # log overall supervised loss
            # For training: log on step; for validation: log on epoch
            on_step = (stage == 'train')
            on_epoch = True  # Always log on epoch for both train and val
            self.log(
                f'{stage}_loss',
                loss,
                prog_bar=True,
                sync_dist=True,
                on_step=on_step,
                on_epoch=on_epoch
            )
            # log individual supervised losses
            for log_dict in log_list:
                self.log(
                    log_dict['name'],
                    log_dict['value'].to(self.device),
                    prog_bar=log_dict.get('prog_bar', False),
                    sync_dist=True,
                    on_step=on_step,
                    on_epoch=on_epoch
                )

        return loss

    def training_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Base training step, a wrapper around the `evaluate_batch` method."""
        loss = self.evaluate_batch(batch_dict, 'train')
        return {'loss': loss}

    def validation_step(self, batch_dict: dict, batch_idx: int) -> None:
        """Base validation step, a wrapper around the `evaluate_batch` method."""
        self.evaluate_batch(batch_dict, 'val')

    def test_step(self, batch_dict: dict, batch_idx: int) -> None:
        """Base test step, a wrapper around the `evaluate_batch` method."""
        self.evaluate_batch(batch_dict, 'test')

    def build_viz_callback(
        self,
        output_dir: str | Path,
        every_n_epochs: int,
        num_frames: int,
    ) -> pl.Callback:
        """Build the validation visualization callback; subclasses override to change the type.

        Parameters
        ----------
        output_dir: run directory where visuals are saved
        every_n_epochs: save every this many epochs
        num_frames: number of validation frames (or scenes) to visualize

        Returns
        -------
        configured visualization callback

        """
        # imported here: beast.callbacks.val_viz imports this module
        from beast.callbacks.val_viz import ValidationVizCallback
        return ValidationVizCallback(
            output_dir=output_dir,
            every_n_epochs=every_n_epochs,
            num_frames=num_frames,
        )

    def configure_callbacks(self) -> list[pl.Callback]:
        """Return model-specific callbacks; Lightning merges these with the trainer's.

        Adds the validation visualization callback when ``training.viz_every_n_epochs`` is set.
        Models needing other callbacks should extend this method. Do not return
        ``ModelCheckpoint`` instances here: Lightning replaces trainer callbacks of the same type.

        Returns
        -------
        list of callbacks

        Raises
        ------
        ValueError: if visualization is enabled but ``training.output_dir`` is not set

        """
        cfg = self.config['training']
        if not cfg['viz_every_n_epochs']:
            return []
        if not cfg.get('output_dir'):
            raise ValueError('training.output_dir is required when viz_every_n_epochs is set')
        return [self.build_viz_callback(
            output_dir=cfg['output_dir'],
            every_n_epochs=cfg['viz_every_n_epochs'],
            num_frames=cfg['viz_num_frames'],
        )]

    def get_viz_panels(self, batch_dict: dict) -> list[torch.Tensor]:
        """Return image panels for validation contact sheets; optional for subclasses.

        Models that implement this can be used with ``ValidationVizCallback``. Each returned
        tensor is one panel type (e.g. input, masked input, reconstruction) for the whole batch.

        Parameters
        ----------
        batch_dict: dict containing model inputs

        Returns
        -------
        list of float tensors of shape (batch, 3, height, width) with values in [0, 1];
        element k holds panel k of every frame, in a fixed order

        Raises
        ------
        NotImplementedError: if the model does not support contact-sheet visualization

        """
        raise NotImplementedError(
            f'{type(self).__name__} does not implement get_viz_panels'
        )

    # Required Lightning methods to be implemented by children
    @abstractmethod
    def get_model_outputs(self, batch_dict: dict) -> dict:
        """Run forward pass and return model outputs; implemented by subclasses.

        Parameters
        ----------
        batch_dict: dict containing model inputs

        Returns
        -------
        dict of model outputs

        """

    @abstractmethod
    def compute_loss(self, stage: str | None, **kwargs) -> tuple[torch.Tensor, list[dict]]:
        """Compute loss from model outputs; implemented by subclasses.

        Parameters
        ----------
        stage: training stage ('train', 'val', 'test', or None)
        **kwargs: model output dict entries

        Returns
        -------
        tuple of (total loss tensor, list of logging dicts)

        """

    @abstractmethod
    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Run inference on a single batch; implemented by subclasses.

        Parameters
        ----------
        batch_dict: dict containing model inputs
        batch_idx: index of the current batch

        Returns
        -------
        dict of predictions and metadata

        """
