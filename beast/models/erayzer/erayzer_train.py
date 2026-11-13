"""Training entry point for ERayZer.

Delegates to the shared beast.train.train function. ``build_datamodule`` builds the
MultiViewDataModule.
"""

import lightning.pytorch as pl

from beast.data.builders import build_multiview_datamodule
from beast.train import train


def build_datamodule(config: dict) -> pl.LightningDataModule:
    """Build the datamodule for ERayZer.

    Parameters
    ----------
    config: full experiment configuration dict

    Returns
    -------
    multi-view datamodule

    """
    return build_multiview_datamodule(config)


__all__ = ['build_datamodule', 'train']
