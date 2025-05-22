import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Union


class BaseLightningModel(pl.LightningModule):
    """Base Lightning Module that specific model architectures will inherit from."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        # Child classes implement architecture setup

    # Required Lightning methods to be implemented by children
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
