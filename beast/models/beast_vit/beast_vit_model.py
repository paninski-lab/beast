"""Vision transformer autoencoder implementation."""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from transformers import (
    ViTMAEConfig,
    ViTMAEForPreTraining,
)

from beast.logging import log_step
from beast.models.base import BaseLightningModel
from beast.nn.perceptual import AlexPerceptual


class BatchNormProjector(nn.Module):
    """Three-layer MLP with batch normalization for projecting encoder representations."""

    def __init__(self, config: ViTMAEConfig) -> None:
        """Build three-layer MLP projection head from ViTMAE config.

        Parameters
        ----------
        config: ViTMAE model configuration specifying hidden_size and embed_size

        """
        super().__init__()
        self.config = config
        self.proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.embed_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden representation through the MLP head."""
        proj_hidden = self.proj(x)
        return proj_hidden


class VisionTransformer(BaseLightningModel):
    """Vision Transformer implementation."""

    def __init__(self, config: dict) -> None:
        """Initialize ViT-MAE model with optional pretrained weights and loss heads.

        Parameters
        ----------
        config: model configuration dict

        """
        super().__init__(config)
        # Set up ViT architecture
        vit_mae_config = ViTMAEConfig(**config['model']['model_params'])

        # Check if we should use pretrained weights or random initialization
        use_pretrained = not config['model']['model_params'].get('random_init', False)
        if use_pretrained:
            log_step(
                "Loading pretrained model from 'facebook/vit-mae-base'"
                ' (this may take several minutes if downloading)...',
                level='debug',
            )
            log_step("Note: Model will be cached locally after first download", level='debug')
            self.vit_mae = ViTMAE.from_pretrained("facebook/vit-mae-base", config=vit_mae_config)
        else:
            log_step("Using random initialization (random_init=True)", level='debug')
            self.vit_mae = ViTMAE(vit_mae_config)
            log_step("Randomly initialized model created", level='debug')

        self.mask_ratio = config['model']['model_params']['mask_ratio']
        # perceptual loss
        if config['model']['model_params'].get('use_perceptual_loss', False):
            self.perceptual_loss = AlexPerceptual(
                device=self.device,
                criterion=nn.MSELoss()
            )
        # contrastive loss
        if config['model']['model_params'].get('use_infoNCE', False):
            self.proj = BatchNormProjector(vit_mae_config)
            if config['model']['model_params'].get('temp_scale', False):
                self.temperature = nn.Parameter(torch.ones([]) * np.log(1))

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
        return_recon: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run ViT-MAE forward pass with optional reconstruction and contrastive projection.

        Parameters
        ----------
        x: input image batch of shape (batch, channels, height, width)
        return_recon: whether to compute and return reconstructed images

        Returns
        -------
        dict with 'latents' and 'loss', plus optional 'reconstructions',
        'perceptual_loss', 'z', and 'cls_token'

        """
        results_dict = self.vit_mae(pixel_values=x, return_recon=return_recon)
        if (
            self.config['model']['model_params'].get('use_perceptual_loss', False)
            and 'reconstructions' in results_dict
        ):
            results_dict['perceptual_loss'] = self.perceptual_loss(
                results_dict['reconstructions'], x
            )
        if self.config['model']['model_params'].get('use_infoNCE', False):
            cls_token = results_dict['latents'][:, 0, :]
            proj_hidden = self.proj(cls_token)
            # normalize projection
            z = proj_hidden / proj_hidden.norm(dim=-1, keepdim=True)
            results_dict['z'] = z
            results_dict['cls_token'] = cls_token

        return results_dict

    def get_model_outputs(
        self,
        batch_dict: dict,
        return_images: bool = True,
        return_reconstructions: bool = True,
    ) -> dict:
        """Run forward pass and return results dict.

        Parameters
        ----------
        batch_dict: dict containing 'image' tensor
        return_images: whether to include input images in results
        return_reconstructions: whether to compute and return reconstructed images

        Returns
        -------
        dict with model outputs and optionally 'images'

        """
        x = batch_dict['image']
        results_dict = self.forward(x, return_recon=return_reconstructions)
        if return_images:
            results_dict['images'] = x
        return results_dict

    def compute_loss(
        self,
        stage: str | None,
        **kwargs,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Combine MSE, perceptual, and infoNCE losses for logging and optimization.

        Parameters
        ----------
        stage: training stage ('train', 'val', 'test', or None)
        **kwargs: model output dict entries (loss, perceptual_loss, z, etc.)

        Returns
        -------
        tuple of (total loss tensor, list of logging dicts)

        """
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        mse_loss = kwargs['loss']
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss.clone()}
        ]
        loss = mse_loss
        if self.config['model']['model_params'].get('use_perceptual_loss', False):
            perceptual_loss = kwargs['perceptual_loss']
            log_list.append({
                'name': f'{stage}_perceptual',
                'value': perceptual_loss.clone()
            })
            loss += self.config['model']['model_params'].get(
                'lambda_perceptual', 10.0
            ) * perceptual_loss
        if self.config['model']['model_params'].get('use_infoNCE', False) and stage == 'train':
            z = kwargs['z']
            sim_matrix = z @ z.T
            if self.config['model']['model_params'].get('temp_scale', False):
                sim_matrix /= self.temperature.exp()
            loss_dict = batch_wise_contrastive_loss(sim_matrix)
            loss_dict['infoNCE_loss'] *= self.config['model']['model_params']['infoNCE_weight']
            log_list.append({
                'name': f'{stage}_infoNCE',
                'value': loss_dict['infoNCE_loss']
            })
            log_list.append({
                'name': f'{stage}_infoNCE_percent_correct',
                'value': loss_dict['percent_correct']
            })
            loss += loss_dict['infoNCE_loss']
        return loss, log_list

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Run inference on a single batch, extracting CLS token latents.

        Parameters
        ----------
        batch_dict: dict containing 'image', 'video', 'idx', 'image_path'
        batch_idx: index of the current batch

        Returns
        -------
        dict with 'latents' (CLS tokens), optional 'reconstructions', and 'metadata'

        """
        # set mask_ratio to 0 for inference
        self.vit_mae.config.mask_ratio = 0.0
        # get model outputs
        results_dict = self.get_model_outputs(
            batch_dict,
            return_images=False,
            return_reconstructions=self.return_reconstructions,
        )
        # reset mask_ratio to the original value
        self.vit_mae.config.mask_ratio = self.mask_ratio
        # just extract CLS tokens
        cls_tokens = results_dict['latents'][:, 0, :].clone()
        del results_dict['latents']
        results_dict['latents'] = cls_tokens
        # save metadata
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict


class ViTMAE(ViTMAEForPreTraining):
    """ViT-MAE for masked autoencoding. Returns latents, reconstructions, and MSE loss."""

    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        return_latent: bool = False,
        return_recon: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run masked autoencoder forward pass.

        Parameters
        ----------
        pixel_values: input image batch
        noise: optional noise tensor for reproducible masking
        head_mask: optional mask for attention heads
        output_attentions: whether to return attention weights
        output_hidden_states: whether to return all hidden states
        return_dict: whether to use return dict (defaults to config setting)
        return_latent: if True, return only the raw latent tensor
        return_recon: if True, run full decode and return reconstructions

        Returns
        -------
        dict with 'latents' and 'loss', plus 'reconstructions' if return_recon is True

        """
        # Setting default for return_dict based on the configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (self.training or self.config.mask_ratio > 0) or return_recon:
            outputs = self.vit(
                pixel_values,
                noise=noise,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            latent = outputs.last_hidden_state
        else:
            # use for fine-tuning, or inference
            # mask_ratio = 0
            embedding_output, mask, ids_restore = self.vit.embeddings(pixel_values)
            embedding_output_ = embedding_output[:, 1:, :]  # no cls token
            # unshuffle the embedding output
            index = ids_restore.unsqueeze(-1).repeat(
                1, 1, embedding_output_.shape[2]
            ).to(embedding_output_.device)
            embedding_output_ = torch.gather(embedding_output_, dim=1, index=index)
            # add cls token back
            embedding_output = torch.cat((embedding_output[:, :1, :], embedding_output_), dim=1)
            encoder_outputs = self.vit.encoder(  # pyright: ignore[reportCallIssue]
                embedding_output,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            latent = self.vit.layernorm(sequence_output)
            if not return_latent:
                return {'latents': latent, 'loss': torch.zeros(1, device=latent.device)}
        if return_latent:
            return latent
        # extract cls latent
        cls_latent = latent[:, 0]  # shape (batch_size, hidden_size)
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits
        # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        loss = self.forward_loss(pixel_values, logits, mask)  # pyright: ignore[reportCallIssue]

        if return_recon:
            return {
                'latents': latent,
                'loss': loss,
                'mse_loss': loss,
                'reconstructions': self.unpatchify(logits),
            }
        return {
            'latents': cls_latent,
            'loss': loss,
            'logits': logits,
        }


def topk(similarities: torch.Tensor, labels: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Compute top-k accuracy as the fraction of samples whose true label is in the top-k.

    Parameters
    ----------
    similarities: pairwise similarity matrix of shape (N, N-1) with diagonal removed
    labels: ground-truth label indices of shape (N,)
    k: number of top predictions to consider

    Returns
    -------
    sum of per-rank hit rates from rank 1 to k

    """
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = torch.tensor(0.0, device=similarities.device)
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities, dim=1)[:, -(i+1)] == labels) / len(labels)
    return topsum


def batch_wise_contrastive_loss(sim_matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute batch-wise infoNCE loss for temporally adjacent frame pairs.

    Assumes the first half of the batch contains reference frames and the second half
    contains their corresponding positive (adjacent) frames.

    Parameters
    ----------
    sim_matrix: pairwise cosine similarity matrix of shape (N, N)

    Returns
    -------
    dict with 'infoNCE_loss' and 'percent_correct'

    """
    N = sim_matrix.shape[0]
    # remove the diagonal from the sim_matrix
    mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix = sim_matrix[~mask].view(N, N-1)
    labels = torch.arange(N).to(sim_matrix.device)
    labels_i, labels_j = labels[:N//2], labels[N//2:] - 1
    labels = torch.cat([labels_j, labels_i]).to(sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    percent_correct = topk(sim_matrix, labels, k=1)
    return {
        "infoNCE_loss": loss,
        "percent_correct": percent_correct,
    }
