import contextlib
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from typeguard import typechecked

from beast.inference import predict_images, predict_video
from beast.models.base import BaseLightningModel
from beast.models.resnets import ResnetAutoencoder
from beast.models.vits import VisionTransformer
from beast.train import train


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

    def __init__(
        self,
        model: BaseLightningModel,
        config: dict[str, Any],
        model_dir: str | Path | None = None
    ) -> None:
        """Initialize with model and config."""
        self.model = model
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None

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

        model_dir = Path(model_dir)

        config_path = model_dir / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_type = config['model'].get('model_class', '').lower()
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f'Unknown model type: {model_type}')

        # Initialize the LightningModule
        model_class = cls.MODEL_REGISTRY[model_type]
        model = model_class(config)

        print(f'Loaded a {model_class} model')

        # Load best weights
        checkpoint_path = list(model_dir.rglob('*best.ckpt'))[0]
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])
        print(f'Loaded model weights from {checkpoint_path}')

        return cls(model, config, model_dir)

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

        print(f'Initialized a {model_class} model')

        return cls(model, config, model_dir=None)

    def train(self, output_dir: str | Path = 'runs/default'):
        """Train the model using PyTorch Lightning.

        Parameters
        ----------
        output_dir: Directory to save checkpoints

        """
        self.model_dir = Path(output_dir)
        with chdir(self.model_dir):
            self.model = train(self.config, self.model, output_dir=self.model_dir)

    def predict_images(
        self,
        image_dir: str | Path,
        output_dir: str | Path | None = None,
        batch_size: int = 32,
        save_latents: bool = True,
        save_reconstructions: bool = True,
    ) -> dict[str, Any]:
        """Run inference on a possibly nested directory of images.

        Parameters
        ----------
        image_dir: absolute path to possibly nested image directories
        output_dir: absolute path to directory where results are saved
        batch_size: batch size for inference
        save_latents: save latents for each image as a numpy file
        save_reconstructions: save reconstructed images

        Returns
        -------
        Predictions and latents

        """
        image_dir = Path(image_dir)
        outputs = predict_images(
            model=self.model,
            output_dir=output_dir or self.model_dir / 'image_predictions' / image_dir.stem,
            source_dir=image_dir,
            batch_size=batch_size,
            save_latents=save_latents,
            save_reconstructions=save_reconstructions,
        )
        return outputs

    def predict_video(
        self,
        video_file: str | Path,
        output_dir: str | Path | None = None,
        batch_size: int = 32,
        save_latents: bool = True,
        save_reconstructions: bool = True,
    ) -> None:
        """Run inference on a single video.

        Parameters
        ----------
        video_file: absolute path to video file (mp4 or avi)
        output_dir: absolute path to directory where results are saved
        batch_size: batch size for inference
        save_latents: save latents for each image as a numpy file
        save_reconstructions: save reconstructed images

        """
        video_file = Path(video_file)
        predict_video(
            model=self.model,
            output_dir=output_dir or self.model_dir / 'video_predictions',
            video_file=video_file,
            batch_size=batch_size,
            save_latents=save_latents,
            save_reconstructions=save_reconstructions,
        )
