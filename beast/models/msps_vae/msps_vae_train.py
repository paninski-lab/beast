"""Training entry point for MSPS-VAE.

Delegates to the shared beast.train.train function, which handles BaseDataModule
(routed to the triplet sampler for this model_class) and epoch-based training.
"""

from beast.train import train

__all__ = ['train']
