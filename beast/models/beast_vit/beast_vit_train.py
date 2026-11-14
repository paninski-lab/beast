"""Training entry point for the Vision Transformer autoencoder.

Delegates to the shared beast.train.train function. ``build_datamodule`` builds the
BaseDataset / BaseDataModule, with the contrastive sampler when ``use_infoNCE`` is set.
"""

import lightning.pytorch as pl

from beast.data.builders import build_singleview_datamodule
from beast.train import train


def build_datamodule(config: dict) -> pl.LightningDataModule:
    """Build the datamodule for the Vision Transformer autoencoder.

    Parameters
    ----------
    config: full experiment configuration dict

    Returns
    -------
    single-view datamodule

    """
    return build_singleview_datamodule(config)


__all__ = ['build_datamodule', 'train']
