"""ResNet autoencoder Lightning model.

The encoder, decoder, and latent bottleneck live in ``beast.nn.resnet`` so that other models
(e.g. MSPS-VAE) can reuse them.
"""

import torch
import torch.nn as nn
from jaxtyping import Float

from beast.data.utils import denormalize_images
from beast.models.base import BaseLightningModel
from beast.nn.resnet import LatentMapping, ResNetDecoder, ResNetEncoder, get_configs


class ResnetAutoencoder(BaseLightningModel):
    """ResNet autoencoder implementation."""

    def __init__(self, config: dict) -> None:
        """Initialize encoder, decoder, and optional latent bottleneck from config."""
        super().__init__(config)

        resnet_config, bottleneck = get_configs(config['model']['model_params']['backbone'])
        self.encoder = ResNetEncoder(configs=resnet_config, bottleneck=bottleneck)
        self.decoder = ResNetDecoder(configs=resnet_config[::-1], bottleneck=bottleneck)

        # if 'num_latents' exists, create a linear bottleneck layer
        # otherwise, the intermediate bottleneck is a set of high-dimensional feature maps
        self.num_latents = config['model']['model_params'].get('num_latents')
        if self.num_latents:
            self.encoder_to_latents = LatentMapping(
                num_latents=self.num_latents, source='encoder', bottleneck=bottleneck,
            )
            self.latents_to_decoder = LatentMapping(
                num_latents=self.num_latents, source='latents', bottleneck=bottleneck,
            )

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> tuple[
        Float[torch.Tensor, 'batch channels img_height img_width'],         # reconstructions
        Float[torch.Tensor, 'batch features feat_height feat_width']        # latents
        | Float[torch.Tensor, 'batch num_latents'],
    ]:
        """Encode input image to latents and decode back to image space.

        Returns
        -------
        tuple of (reconstructed_images, latents)

        """
        features = self.encoder(x)
        if self.num_latents:
            z = self.encoder_to_latents(features)
            features = self.latents_to_decoder(z)
        else:
            z = features
        xhat = self.decoder(features)
        return xhat, z

    def get_model_outputs(
        self,
        batch_dict: dict,
        return_images: bool = True,
        return_reconstructions: bool = True,
    ) -> dict:
        """Run forward pass and return results dict with optional images and reconstructions.

        Parameters
        ----------
        batch_dict: dict containing 'image' tensor
        return_images: whether to include input images in results
        return_reconstructions: whether to include reconstructions in results

        Returns
        -------
        dict with 'latents', and optionally 'images' and 'reconstructions'

        """
        x = batch_dict['image']
        xhat, z = self.forward(x)
        results_dict = {
            'latents': z,
        }
        if return_images:
            results_dict['images'] = x
        if return_reconstructions:
            results_dict['reconstructions'] = xhat
        return results_dict

    def compute_loss(
        self,
        stage: str | None,
        images: Float[torch.Tensor, 'batch channels img_height img_width'],
        reconstructions: Float[torch.Tensor, 'batch channels img_height img_width'],
        latents: (
            Float[torch.Tensor, 'batch features feat_height feat_width']
            | Float[torch.Tensor, 'batch num_latents']
        ),
        **kwargs,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Compute MSE reconstruction loss between input images and reconstructions.

        Parameters
        ----------
        stage: training stage ('train', 'val', 'test', or None)
        images: original input images
        reconstructions: model reconstructions
        latents: encoder latent representations (unused; required by base class signature)
        **kwargs: additional keyword arguments (ignored)

        Returns
        -------
        tuple of (loss tensor, list of logging dicts)

        """
        mse_loss = nn.functional.mse_loss(images, reconstructions, reduction='mean')
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss}
        ]
        return mse_loss, log_list

    @torch.no_grad()
    def get_viz_panels(self, batch_dict: dict) -> list[torch.Tensor]:
        """Return input and reconstruction panels for contact sheets.

        Parameters
        ----------
        batch_dict: dict containing 'image' tensor

        Returns
        -------
        list of two (batch, 3, height, width) tensors in [0, 1]: input, reconstruction

        """
        out = self.get_model_outputs(batch_dict)
        return [denormalize_images(out['images']), denormalize_images(out['reconstructions'])]

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Run inference on a single batch and return latents with metadata.

        Parameters
        ----------
        batch_dict: dict containing 'image', 'video', 'idx', 'image_path'
        batch_idx: index of the current batch

        Returns
        -------
        dict with 'latents', optional 'reconstructions', and 'metadata'

        """
        results_dict = self.get_model_outputs(
            batch_dict,
            return_images=False,
            return_reconstructions=self.return_reconstructions,
        )
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict
