from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from PIL import Image
from typeguard import typechecked

from beast.data.datasets import BaseDataset
from beast.models.base import BaseLightningModel


@typechecked
class PredictionHandler:
    """Handles saving predictions while preserving directory structure."""

    def __init__(self, output_dir: Path, source_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.source_dir = Path(source_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store metadata for each prediction
        self.metadata = []

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor (C, H, W) to PIL Image."""
        # Handle different tensor formats
        if tensor.dim() == 4:  # (B, C, H, W) - take first batch item
            tensor = tensor[0]

        # Convert from (C, H, W) to (H, W, C)
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)

        # Ensure values are in [0, 255] range
        if tensor.max() <= 1.0:
            tensor = tensor * 255

        # Convert to uint8 numpy array
        np_array = tensor.detach().cpu().numpy().astype(np.uint8)

        # Handle grayscale vs RGB
        if np_array.shape[2] == 1:
            np_array = np_array.squeeze(2)
            return Image.fromarray(np_array, mode='L')
        else:
            return Image.fromarray(np_array, mode='RGB')

    def save_reconstruction(
        self,
        reconstruction: torch.Tensor,
        video: str,
        idx: int,
        original_path: Path,
    ) -> Path:
        """Save a single reconstruction maintaining directory structure."""
        # Create output subdirectory matching source structure
        output_subdir = self.output_dir / video
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Get original filename
        original_filename = original_path.name
        output_path = output_subdir / original_filename

        # Convert tensor to image and save
        image = self.tensor_to_image(reconstruction)
        image.save(output_path)

        return output_path

    def save_latents(
        self,
        latents: torch.Tensor,
        video: str,
        idx: int,
        original_path: Path,
    ) -> Path:
        """Save latent representations as numpy arrays."""
        # Create latents subdirectory
        latents_dir = self.output_dir / 'latents' / video
        latents_dir.mkdir(parents=True, exist_ok=True)

        # Save as .npy file
        original_stem = original_path.stem
        latents_path = latents_dir / f'{original_stem}.npy'

        # Convert to numpy and save
        latents_np = latents.detach().cpu().numpy()
        np.save(latents_path, latents_np)

        return latents_path

    def process_batch_predictions(
        self,
        predictions: dict,
        batch_metadata: dict,
        save_reconstructions: bool = True,
        save_latents: bool = False
    ) -> dict[str, list]:
        """Process a batch of predictions and save them."""
        reconstructions = predictions['reconstructions']
        latents = predictions['latents']

        batch_size = reconstructions.shape[0]

        saved_files = {
            'reconstructions': [],
            'latents': [],
            'metadata': []
        }

        for i in range(batch_size):
            video = batch_metadata['video'][i]
            idx = batch_metadata['idx'][i].item()

            # Get original image path for this item
            original_path = batch_metadata['image_paths'][i]

            # Initialize metadata entry
            metadata_entry = {
                'original_path': str(original_path),
                'video': video,
                'idx': idx
            }

            # Save reconstruction if requested
            if save_reconstructions:
                recon_path = self.save_reconstruction(
                    reconstructions[i], video, idx, Path(original_path),
                )
                saved_files['reconstructions'].append(str(recon_path))
                metadata_entry['reconstruction_path'] = str(recon_path)

            # Save latents if requested
            if save_latents:
                latents_path = self.save_latents(latents[i], video, idx, Path(original_path))
                saved_files['latents'].append(str(latents_path))
                metadata_entry['latents_path'] = str(latents_path)

            saved_files['metadata'].append(metadata_entry)
            self.metadata.append(metadata_entry)

        return saved_files

    def process_predictions(
        self,
        predictions: list[dict],
        save_reconstructions: bool = True,
        save_latents: bool = False,
    ) -> dict[str, Any]:
        """Process all predictions from trainer.predict() and save results.

        Parameters
        ----------
        predictions: List of prediction dictionaries from trainer.predict()
        save_reconstructions: Whether to save reconstruction images
        save_latents: Whether to save latent representations

        Returns
        -------
        Dictionary with summary of saved files and metadata

        """
        all_saved_files = {
            'reconstructions': [],
            'latents': [],
            'metadata': []
        }

        for batch_predictions in predictions:
            # Extract metadata from predictions
            batch_metadata = batch_predictions['metadata']

            # Process this batch
            saved_files = self.process_batch_predictions(
                batch_predictions,
                batch_metadata,
                save_reconstructions=save_reconstructions,
                save_latents=save_latents
            )

            # Accumulate results
            all_saved_files['reconstructions'].extend(saved_files['reconstructions'])
            all_saved_files['latents'].extend(saved_files['latents'])
            all_saved_files['metadata'].extend(saved_files['metadata'])

        # Save metadata summary
        metadata_path = self.save_metadata_summary()

        # Create results summary
        results = {
            'output_dir': str(self.output_dir),
            'num_images_processed': len(all_saved_files['metadata']),
            'metadata_file': str(metadata_path)
        }

        if save_reconstructions:
            results['reconstructions_saved'] = len(all_saved_files['reconstructions'])
            results['reconstructions_dir'] = str(self.output_dir)

        if save_latents:
            results['latents_saved'] = len(all_saved_files['latents'])
            results['latents_dir'] = str(self.output_dir / "latents")

        # Print summary
        print(f"✓ Processed {results['num_images_processed']} images")
        if save_reconstructions:
            print(f"✓ Saved {results['reconstructions_saved']} reconstructions")
        if save_latents:
            print(f"✓ Saved {results['latents_saved']} latent representations")
        print(f"✓ Results saved to: {self.output_dir}")
        print(f"✓ Metadata saved to: {metadata_path}")

        return results

    def save_metadata_summary(self) -> Path:
        """Save complete metadata summary to YAML."""
        metadata_path = self.output_dir / 'prediction_metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.safe_dump(self.metadata, f)
        return metadata_path


@typechecked
def predict_images(
    model: BaseLightningModel,
    output_dir: str | Path,
    source_dir: str | Path,
    batch_size: int = 32,
    save_latents: bool = False,
    save_reconstructions: bool = True,
) -> dict[str, Any]:
    """Run inference on images using a trained model and save results.

    Processes all images in a directory (including nested subdirectories) through
    a trained PyTorch Lightning model, generating reconstructions and/or latent
    representations. Preserves the original directory structure in the output.

    Parameters
    ----------
    model: trained Beast model for inference
    output_dir: directory where results will be saved; creates subdirectories matching the source
        directory structure
    source_dir: directory containing input images; supports nested directory structures
    batch_size: number of images to process in each batch
    save_latents: whether to save latent representations as .npy files in a 'latents/' subdirectory
    save_reconstructions: whether to save reconstructed images as PNG files

    Returns
    -------
    Dictionary containing inference results with keys:
        - 'output_dir': Path to output directory
        - 'num_images_processed': Total number of images processed
        - 'metadata_file': Path to YAML metadata summary file
        - 'reconstructions_saved': Number of reconstructions saved (if enabled)
        - 'latents_saved': Number of latent files saved (if enabled)
        - 'reconstructions_dir': Path to reconstructions directory (if enabled)
        - 'latents_dir': Path to latents directory (if enabled)

    """

    output_dir = Path(output_dir)
    source_dir = Path(source_dir)

    # initialize prediction handler
    handler = PredictionHandler(output_dir, source_dir)

    # dataset
    dataset = BaseDataset(
        data_dir=source_dir,
        imgaug_pipeline=None,
    )

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
    )

    # run inference
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    predictions = trainer.predict(model, dataloaders=dataloader, return_predictions=True)

    # process outputs
    results = handler.process_predictions(
        predictions,
        save_reconstructions=save_reconstructions,
        save_latents=save_latents,
    )

    return results
