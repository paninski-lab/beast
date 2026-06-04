"""Training entry point for the ResNet autoencoder.

Delegates to the shared beast.train.train function, which handles
BaseDataset / BaseDataModule and epoch-based training.
"""

from beast.train import train

__all__ = ['train']
