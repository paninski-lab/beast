"""Training entry point for BEAST3D.

Delegates to the shared beast.train.train function, which drives the
MultiViewDataModule and step-based training used by the erayzer / beast3d
model classes.
"""

from beast.train import train

__all__ = ['train']
