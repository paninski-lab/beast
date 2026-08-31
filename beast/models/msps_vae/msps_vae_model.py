"""MSPS-VAE: ResNet autoencoder with a structurally orthogonal latent split.

Reuses the ResNet encoder/decoder and linear bottleneck from `beast_resnet`, but splits
the latent vector into two subspaces via a fixed (non-trainable) random orthogonal
rotation instead of a single flat bottleneck: `z_u` (unsupervised/pattern) and `z_b`
(background/identity, shaped by a triplet loss keyed on video identity). See
cuttle-patterns/docs/msps_vae_implementation.md for the full design rationale, including
why orthogonality is enforced structurally rather than via a soft penalty loss.
"""

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float

from beast.models.base import BaseLightningModel
from beast.models.beast_resnet.beast_resnet_model import (
    LatentMapping,
    ResNetDecoder,
    ResNetEncoder,
    get_configs,
)


class OrthogonalSplit(nn.Module):
    """Splits a shared latent vector into two frozen, structurally orthogonal subspaces.

    A single Haar-random orthogonal matrix is drawn once at construction time; the
    unsupervised and background projections are frozen row-slices of it. Because the
    matrix's rows are orthonormal by construction, any row-partition of it is orthogonal
    by construction too — no orthogonality loss is needed, and the split can't drift
    during training (see the original PS-VAE implementation's `ConvAEMSPSEncoder`, whose
    equivalent soft `||UU^T - I||` penalty was dropped in favor of this fixed-matrix
    approach after proving unstable).
    """

    def __init__(
        self,
        num_latents_unsupervised: int,
        num_latents_background: int,
        seed: int,
    ) -> None:
        """Draw a fixed random orthogonal matrix and split it into frozen linear layers.

        Parameters
        ----------
        num_latents_unsupervised: dimensionality of the unsupervised/pattern subspace
        num_latents_background: dimensionality of the background/identity subspace
        seed: random seed for the orthogonal matrix, drawn independently of the model's
            training seed so it is stable across training-seed sweeps

        """
        super().__init__()

        num_latents_total = num_latents_unsupervised + num_latents_background
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.empty(num_latents_total, num_latents_total)
        nn.init.orthogonal_(matrix, generator=generator)

        self.to_unsupervised = nn.Linear(num_latents_total, num_latents_unsupervised, bias=False)
        self.to_background = nn.Linear(num_latents_total, num_latents_background, bias=False)
        with torch.no_grad():
            self.to_unsupervised.weight.copy_(matrix[:num_latents_unsupervised])
            self.to_background.weight.copy_(matrix[num_latents_unsupervised:])
        self.to_unsupervised.weight.requires_grad_(False)
        self.to_background.weight.requires_grad_(False)

    def forward(
        self,
        x: Float[torch.Tensor, 'batch num_latents_total'],
    ) -> tuple[
        Float[torch.Tensor, 'batch num_latents_unsupervised'],
        Float[torch.Tensor, 'batch num_latents_background'],
    ]:
        """Project the shared latent vector onto the two frozen orthogonal subspaces."""
        return self.to_unsupervised(x), self.to_background(x)


class MspsVae(BaseLightningModel):
    """MSPS-VAE model: ResNet autoencoder with an orthogonally-partitioned bottleneck."""

    def __init__(self, config: dict) -> None:
        """Initialize encoder, decoder, and orthogonally-split latent bottleneck.

        Parameters
        ----------
        config: full experiment configuration dict

        """
        super().__init__(config)

        params = config['model']['model_params']
        resnet_config, bottleneck = get_configs(params['backbone'])
        self.encoder = ResNetEncoder(configs=resnet_config, bottleneck=bottleneck)
        self.decoder = ResNetDecoder(configs=resnet_config[::-1], bottleneck=bottleneck)

        self.num_latents_unsupervised = params['num_latents_unsupervised']
        self.num_latents_background = params['num_latents_background']
        num_latents_total = self.num_latents_unsupervised + self.num_latents_background

        self.encoder_to_latents = LatentMapping(
            num_latents=num_latents_total, source='encoder', bottleneck=bottleneck,
        )
        self.latents_to_decoder = LatentMapping(
            num_latents=num_latents_total, source='latents', bottleneck=bottleneck,
        )
        self.orthogonal_split = OrthogonalSplit(
            num_latents_unsupervised=self.num_latents_unsupervised,
            num_latents_background=self.num_latents_background,
            seed=params['orthogonal_matrix_seed'],
        )

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> tuple[
        Float[torch.Tensor, 'batch channels img_height img_width'],  # reconstructions
        Float[torch.Tensor, 'batch num_latents_unsupervised'],       # z_u
        Float[torch.Tensor, 'batch num_latents_background'],         # z_b
    ]:
        """Encode input image to a split latent, decode back to image space.

        Returns
        -------
        tuple of (reconstructed_images, z_u, z_b)

        """
        features = self.encoder(x)
        shared_latent = self.encoder_to_latents(features)
        z_u, z_b = self.orthogonal_split(shared_latent)
        decoder_input = torch.cat([z_u, z_b], dim=1)
        decoder_features = self.latents_to_decoder(decoder_input)
        xhat = self.decoder(decoder_features)
        return xhat, z_u, z_b

    def get_model_outputs(
        self,
        batch_dict: dict,
        return_images: bool = True,
        return_reconstructions: bool = True,
    ) -> dict:
        """Run forward pass and return results dict with optional images and reconstructions.

        Parameters
        ----------
        batch_dict: dict containing 'image' tensor, and 'video' (list of str) when built by
            the triplet sampler/collate pipeline
        return_images: whether to include input images in results
        return_reconstructions: whether to include reconstructions in results

        Returns
        -------
        dict with 'z_u', 'z_b', 'video', and optionally 'images' and 'reconstructions'

        """
        x = batch_dict['image']
        xhat, z_u, z_b = self.forward(x)
        results_dict = {
            'z_u': z_u,
            'z_b': z_b,
            'video': batch_dict.get('video'),
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
        z_u: Float[torch.Tensor, 'batch num_latents_unsupervised'],
        z_b: Float[torch.Tensor, 'batch num_latents_background'],
        video: list[str] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Combine MSE reconstruction and triplet (on z_b) losses.

        The triplet term only applies during training, where the triplet batch sampler
        guarantees the (ref, pos) pairing structure it relies on — `video` is ordered as
        [ref_0, ..., ref_{B-1}, pos_0, ..., pos_{B-1}], so item i's positive is at
        `i +/- B`. Negatives are drawn per-anchor from other batch members with a
        different `video` value.

        Parameters
        ----------
        stage: training stage ('train', 'val', 'test', or None)
        images: original input images
        reconstructions: model reconstructions
        z_u: unsupervised/pattern latents (unused in the loss; logged/saved elsewhere)
        z_b: background/identity latents; triplet loss is computed on these
        video: per-item video name, ordered as [ref..., pos...] (train stage only)
        **kwargs: additional keyword arguments (ignored)

        Returns
        -------
        tuple of (loss tensor, list of logging dicts)

        """
        mse_loss = nn.functional.mse_loss(images, reconstructions, reduction='mean')
        log_list = [{'name': f'{stage}_mse', 'value': mse_loss}]
        loss = mse_loss

        if stage == 'train' and video is not None:
            triplet_loss = self._compute_triplet_loss(z_b, video)
            weight = self.config['model']['model_params']['triplet_weight']
            log_list.append({'name': f'{stage}_triplet', 'value': triplet_loss})
            loss = loss + weight * triplet_loss

        return loss, log_list

    def _compute_triplet_loss(
        self,
        z_b: Float[torch.Tensor, 'batch num_latents_background'],
        video: list[str],
    ) -> torch.Tensor:
        """Compute triplet margin loss on z_b using the batch's (ref, pos) pairing.

        Positives come from the fixed ref/pos pairing the triplet sampler built the batch
        with; negatives are drawn uniformly at random, per anchor, from other batch
        members with a different `video` value (no hard/semi-hard mining).
        """
        n = z_b.shape[0]
        half = n // 2
        positive_idx = torch.cat(
            [torch.arange(half, n), torch.arange(0, half)],
        ).to(z_b.device)

        video_arr = np.asarray(video)
        negative_idx = np.empty(n, dtype=np.int64)
        for i in range(n):
            candidates = np.flatnonzero(video_arr != video_arr[i])
            if candidates.size == 0:
                raise ValueError(
                    'Triplet loss requires at least two distinct videos per batch; got a '
                    f'single video ({video_arr[i]!r}) across the whole batch of size {n}.'
                )
            negative_idx[i] = np.random.choice(candidates)
        negative_idx_t = torch.from_numpy(negative_idx).to(z_b.device)

        margin = self.config['model']['model_params']['triplet_margin']
        return nn.functional.triplet_margin_loss(
            anchor=z_b,
            positive=z_b[positive_idx],
            negative=z_b[negative_idx_t],
            margin=margin,
        )

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Run inference on a single batch and return latents with metadata.

        `latents` is saved as `concat(z_u, z_b)` along dim 1 — the first
        `num_latents_unsupervised` columns are `z_u` (what downstream clustering should
        read), the remaining `num_latents_background` columns are `z_b`.

        Parameters
        ----------
        batch_dict: dict containing 'image', 'video', 'idx', 'image_path'
        batch_idx: index of the current batch

        Returns
        -------
        dict with 'latents' (concat(z_u, z_b)), optional 'reconstructions', and 'metadata'

        """
        results_dict = self.get_model_outputs(
            batch_dict,
            return_images=False,
            return_reconstructions=self.return_reconstructions,
        )
        z_u = results_dict.pop('z_u')
        z_b = results_dict.pop('z_b')
        results_dict.pop('video', None)
        results_dict['latents'] = torch.cat([z_u, z_b], dim=1)
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict
