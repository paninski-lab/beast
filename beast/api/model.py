import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
import yaml
from typeguard import typechecked

from beast.models.base import BaseLightningModel
from beast.models.resnets import ResnetAutoencoder
from beast.models.vits import VisionTransformer
from beast.train import train

_logger = logging.getLogger('BEAST.API.MODEL')


# TODO: Replace with contextlib.chdir in python 3.11.
@contextlib.contextmanager
def chdir(dir: str | Path):
    pwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(pwd)


@typechecked
class Model:
    """High-level API wrapper for BEAST models.

    This class manages both the model and the training/inference processes.
    """

    MODEL_REGISTRY = {
        'vit': VisionTransformer,
        'resnet': ResnetAutoencoder,
        # Add more models as needed
    }

    def __init__(self, model: BaseLightningModel, config: Dict[str, Any]):
        """Initialize with model and config."""
        self.model = model
        self.config = config

    @classmethod
    def from_dir(cls, model_dir: str | Path):
        """Load a model from a directory.

        Parameters
        ----------
        model_dir: Path to directory containing model checkpoint and config

        Returns
        -------
        Initialized model wrapper

        """
        config_path = os.path.join(model_dir, 'config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_type = config.get('model_type', '').lower()
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f'Unknown model type: {model_type}')

        # Initialize the LightningModule
        model_class = cls.MODEL_REGISTRY[model_type]
        model = model_class(config)

        # Load weights
        checkpoint_path = os.path.join(model_dir, 'model.ckpt')
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])

        return cls(model, config)

    @classmethod
    def from_config(cls, config_path: str | Path | dict):
        """Create a new model from a config file.

        Parameters
        ----------
        config_path: Path to config file or config dict

        Returns
        -------
        Initialized model wrapper

        """
        if not isinstance(config_path, dict):
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = config_path

        model_type = config['model'].get('model_class', '').lower()
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f'Unknown model type: {model_type}')

        # Initialize the LightningModule
        model_class = cls.MODEL_REGISTRY[model_type]
        model = model_class(config)

        _logger.info(f'Initialized a {model_class} model')

        return cls(model, config)

    def train(self, output_dir: str | Path = 'runs/default'):
        """Train the model using PyTorch Lightning.

        Parameters
        ----------
        output_dir: Directory to save checkpoints

        """
        with chdir(output_dir):
            self.model = train(self.config, self.model, output_dir=output_dir)

    # def predict_video(
    #     self,
    #     video_path: str,
    #     batch_size: int = 32,
    #     extract_layers: Optional[List[str]] = None
    # ) -> Dict[str, Any]:
    #     """Run inference on a video.
    #
    #     Parameters
    #     ----------
    #     video_path: Path to video file
    #     batch_size: Batch size for inference
    #     extract_layers: Optional layers to extract features from
    #
    #     Returns
    #     -------
    #     Predictions and intermediate features
    #
    #     """
    #     # Extract frames from video
    #     frames = self._extract_frames(video_path)
    #
    #     # Create DataLoader for frames
    #     dataloader = self._create_frame_dataloader(frames, batch_size=batch_size)
    #
    #     # Set up feature extraction hooks if needed
    #     if extract_layers:
    #         extracted_features = self._setup_feature_extraction(extract_layers)
    #
    #     # Run inference
    #     trainer = pl.Trainer(accelerator='auto', devices=1)
    #     predictions = trainer.predict(self.model, dataloaders=dataloader)
    #
    #     results = {'predictions': predictions}
    #
    #     # Add extracted features if available
    #     if extract_layers:
    #         results['features'] = extracted_features
    #
    #     return results

    # def save(self, output_dir: str) -> None:
    #     """Save model and config.
    #
    #     Parameters
    #     ----------
    #     output_dir: Directory to save model
    #
    #     """
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     # Save config
    #     with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
    #         yaml.dump(self.config, f, indent=2)
    #
    #     # Save model weights
    #     checkpoint_path = os.path.join(output_dir, 'model.ckpt')
    #     torch.save({'state_dict': self.model.state_dict()}, checkpoint_path)
