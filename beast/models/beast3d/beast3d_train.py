"""Training entry point for BEAST3D.

Delegates to the shared beast.train.train function. ``build_datamodule`` builds the
MultiViewDataModule with ground-truth foreground masks, which supervise the rendered alpha.
"""

import lightning.pytorch as pl

from beast.data.builders import build_multiview_datamodule
from beast.train import train


def build_datamodule(config: dict) -> pl.LightningDataModule:
    """Build the datamodule for BEAST3D.

    Parameters
    ----------
    config: full experiment configuration dict

    Returns
    -------
    multi-view datamodule that also loads foreground masks

    """
    return build_multiview_datamodule(config, use_mask=True)


__all__ = ['build_datamodule', 'train']
