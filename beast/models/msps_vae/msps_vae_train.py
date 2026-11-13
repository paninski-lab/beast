"""Training entry point for MSPS-VAE.

Delegates to the shared beast.train.train function. ``build_datamodule`` builds the
BaseDataModule with the triplet sampler.
"""

import lightning.pytorch as pl

from beast.data.builders import build_singleview_datamodule
from beast.train import train


def build_datamodule(config: dict) -> pl.LightningDataModule:
    """Build the datamodule for MSPS-VAE.

    Parameters
    ----------
    config: full experiment configuration dict

    Returns
    -------
    single-view datamodule using the triplet sampler

    """
    return build_singleview_datamodule(config, sampler_kind='triplet')


__all__ = ['build_datamodule', 'train']
