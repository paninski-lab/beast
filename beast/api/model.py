import os
import yaml

from typing import Dict, Any, Optional, List, Union
import pytorch_lightning as pl
import torch

from beast.models.base import BaseLightningModel
from beast.models.resnets import ResnetAutoencoder
from beast.models.vits import VisionTransformer


class Model:
    """High-level API wrapper for BEAST models.

    This class manages both the model and the training/inference processes.
    """

    MODEL_REGISTRY = {
        "vit": VisionTransformer,
        "resnet": ResnetAutoencoder,
        # Add more models as needed
    }

    def __init__(self, model: BaseLightningModel, config: Dict[str, Any]):
        """Initialize with model and config."""
        self.model = model
        self.config = config

    @classmethod
    def from_dir(cls, model_dir: str) -> "Model":
        """Load a model from a directory.

        Args:
            model_dir: Path to directory containing model checkpoint and config

        Returns:
            BeastModel: Initialized model wrapper
        """
        config_path = os.path.join(model_dir, "config.yaml")
        with open(config_path) as f:
            config = yaml.load(f)

        model_type = config.get("model_type", "").lower()
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize the LightningModule
        model_class = cls.MODEL_REGISTRY[model_type]
        model = model_class(config)

        # Load weights
        checkpoint_path = os.path.join(model_dir, "model.ckpt")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict["state_dict"])

        return cls(model, config)

    @classmethod
    def from_config(cls, config_path: str) -> "Model":
        """Create a new model from a config file.

        Args:
            config_path: Path to config file

        Returns:
            BeastModel: Initialized model wrapper
        """
        with open(config_path) as f:
            config = yaml.load(f)

        model_type = config.get("model_type", "").lower()
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize the LightningModule
        model_class = cls.MODEL_REGISTRY[model_type]
        model = model_class(config)

        return cls(model, config)

    def train(
        self,
        output_dir: str = "runs/default",
        **trainer_kwargs
    ) -> Dict[str, Any]:
        """Train the model using PyTorch Lightning.

        Args:
            output_dir: Directory to save checkpoints
            **trainer_kwargs: Additional arguments to pass to Trainer

        Returns:
            Dict: Training history
        """
        # create datasets, dataloaders, callbacks, etc.

        # Set up Lightning callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=output_dir,
                filename="model-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                monitor="val_loss"
            ),
            pl.callbacks.LearningRateMonitor()
        ]

        # Set up trainer with defaults that can be overridden
        default_trainer_kwargs = {
            "max_epochs": 100,
            "accelerator": "auto",
            "devices": "auto",
            "logger": pl.loggers.TensorBoardLogger(output_dir),
            "callbacks": callbacks,
        }

        # Update with user-provided kwargs
        trainer_config = {**default_trainer_kwargs, **trainer_kwargs}
        trainer = pl.Trainer(**trainer_config)

        # Train the model
        trainer.fit(self.model)

        # Save final model and config
        self.save(output_dir)

        return {"trainer": trainer}

    def predict_video(self,
                      video_path: str,
                      batch_size: int = 32,
                      extract_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run inference on a video.

        Args:
            video_path: Path to video file
            batch_size: Batch size for inference
            extract_layers: Optional layers to extract features from

        Returns:
            Dict: Predictions and intermediate features
        """
        # Extract frames from video
        frames = self._extract_frames(video_path)

        # Create DataLoader for frames
        dataloader = self._create_frame_dataloader(frames, batch_size=batch_size)

        # Set up feature extraction hooks if needed
        if extract_layers:
            extracted_features = self._setup_feature_extraction(extract_layers)

        # Run inference
        trainer = pl.Trainer(accelerator="auto", devices=1)
        predictions = trainer.predict(self.model, dataloaders=dataloader)

        results = {"predictions": predictions}

        # Add extracted features if available
        if extract_layers:
            results["features"] = extracted_features

        return results

    def save(self, output_dir: str) -> None:
        """Save model and config.

        Args:
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f, indent=2)

        # Save model weights
        checkpoint_path = os.path.join(output_dir, "model.ckpt")
        torch.save({"state_dict": self.model.state_dict()}, checkpoint_path)
