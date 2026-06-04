"""Training entry point for ERayZer.

Delegates to the shared beast.train.train function, which handles
MultiViewDataModule and step-based training for the erayzer model class.
"""

from beast.train import train

__all__ = ['train']
