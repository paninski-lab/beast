from typing import Iterator, Literal

import lightning.pytorch as pl
import torch
from typeguard import typechecked


@typechecked
class BaseLightningModel(pl.LightningModule):
    """Base Lightning Module that specific model architectures will inherit from."""

    def __init__(self, config: dict) -> None:
        """Initialize with config."""

        super().__init__()

        if self.local_rank == 0:
            print(f'\nInitializing a {self._get_name()} instance.')

        self.config = config
        self.seed = config['model']['seed']
        torch.manual_seed(self.seed)

        self.save_hyperparameters(config)
        # Child classes implement architecture setup

    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler = self.config['optimizer']['scheduler']
        if scheduler == 'step':
            # define a scheduler that reduces the base learning rate at predefined steps
            from torch.optim.lr_scheduler import MultiStepLR
            scheduler = MultiStepLR(
                optimizer=optimizer,
                milestones=self.config['optimizer']['steps'],
                gamma=self.config['optimizer']['gamma'],
            )
        elif scheduler == 'cosine':
            from torch.optim.lr_scheduler import OneCycleLR
            # compute max learning rate
            global_batch_size = (
                self.config['training']['train_batch_size']
                * self.config['training']['num_gpus']
                * self.config['training']['num_nodes']
            )
            max_lr = self.config['optimizer']['lr'] * global_batch_size / 256
            scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=max_lr,
                epochs=self.config['training']['num_epochs'],
                steps_per_epoch=1,  # update learning rate after every batch of data
                pct_start=self.config['optimizer']['warmup_pct'],
                anneal_strategy='cos',
                div_factor=self.config['optimizer']['div_factor'],
                final_div_factor=self.config['optimizer'].get('final_div_factor', 1),
            )
        else:
            raise NotImplementedError(f'{scheduler} scheduler is not yet implemented')
        return scheduler

    def get_parameters(self) -> Iterator:
        params = filter(lambda p: p.requires_grad, self.parameters())
        return params

    def configure_optimizers(self) -> dict:
        """Select optimizer, lr scheduler, and metric for monitoring."""

        # get trainable params
        params = self.get_parameters()

        # init optimizer
        optimizer = self.config['optimizer']['type']
        learning_rate = self.config['optimizer']['lr']
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, lr=learning_rate)
        elif self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=self.config['optmizer']['wd'],
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
    ) -> torch.tensor:
        """Compute and log the losses on a batch of labeled data."""

        # forward pass; collected true and predicted heatmaps, keypoints
        data_dict = self.get_model_outputs(batch_dict=batch_dict)

        # compute and log loss on labeled data
        loss, log_list = self.compute_loss(stage=stage, **data_dict)

        if stage:
            # logging with sync_dist=True will average the metric across GPUs in
            # multi-GPU training. Performance overhead was found negligible.

            # log overall supervised loss
            self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
            # log individual supervised losses
            for log_dict in log_list:
                self.log(
                    log_dict['name'],
                    log_dict['value'].to(self.device),
                    prog_bar=log_dict.get('prog_bar', False),
                    sync_dist=True,
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

    # Required Lightning methods to be implemented by children
    def get_model_outputs(self, batch_dict: dict) -> dict:
        raise NotImplementedError

    def compute_loss(self, stage: int, **kwargs) -> tuple[torch.tensor, list[dict]]:
        raise NotImplementedError

    def predict_step(self, batch_dict: dict, batch_idx: int) -> torch.tensor:
        raise NotImplementedError
