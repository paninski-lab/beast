"""Vision transformer autoencoder implementation."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from transformers import (
    ViTMAEConfig,
    ViTMAEForPreTraining,
)
from typeguard import typechecked

from beast.models.base import BaseLightningModel


class BatchNormProjector(nn.Module):
    def __init__(self, config):
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

    def forward(self, x):
        proj_hidden = self.proj(x)
        return proj_hidden


@typechecked
class VisionTransformer(BaseLightningModel):
    """Vision Transformer implementation."""

    def __init__(self, config):
        super().__init__(config)
        # Set up ViT architecture
        vit_mae_config = ViTMAEConfig(**config['model']['model_params'])
        self.vit_mae = ViTMAE(vit_mae_config).from_pretrained("facebook/vit-mae-base")
        self.mask_ratio = config['model']['model_params']['mask_ratio']
        # contrastive loss
        if config['model']['model_params']['use_infoNCE']:
            self.proj = BatchNormProjector(vit_mae_config)
            self.temperature = nn.Parameter(torch.ones([]) * np.log(1))

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> Dict[str, torch.Tensor]:
        results_dict = self.vit_mae(pixel_values=x, return_recon=True)
        if self.config['model']['model_params']['use_infoNCE']:
            cls_token = results_dict['latents'][:, 0, :]
            proj_hidden = self.proj(cls_token)
            # normalize projection
            z = proj_hidden / proj_hidden.norm(dim=-1, keepdim=True)
            results_dict['z'] = z
            results_dict['cls_token'] = cls_token

        return results_dict

    def get_model_outputs(self, batch_dict: dict, return_images: bool = True) -> dict:
        x = batch_dict['image']
        results_dict = self.forward(x)
        if return_images:
            results_dict['images'] = x
        return results_dict

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        mse_loss = kwargs['loss']
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss}
        ]
        loss = mse_loss
        if self.config['model']['model_params']['use_infoNCE']:
            z = kwargs['z']
            sim_matrix = z @ z.T
            if self.config['model']['model_params']['temp_scale']:
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
        # set mask_ratio to 0 for inference
        self.vit_mae.config.mask_ratio = 0
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        # reset mask_ratio to the original value
        self.vit_mae.config.mask_ratio = self.mask_ratio
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict


class ViTMAE(ViTMAEForPreTraining):
    # Overriding the forward method to return the latent and loss
    # This is used for training and inference
    # Huggingface Transformer library
    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_latent: bool = False,
        return_recon: bool = False,
    ) -> Dict[str, torch.Tensor]:
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
            encoder_outputs = self.vit.encoder(
                embedding_output,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            latent = self.vit.layernorm(sequence_output)
            if not return_latent:
                # return the cls token and 0 loss if not return_latent
                return latent[:, 0], 0
        if return_latent:
            return latent
        # extract cls latent
        cls_latent = latent[:, 0]  # shape (batch_size, hidden_size)
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits
        # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        loss = self.forward_loss(pixel_values, logits, mask)
        if return_recon:
            return {
                'latents': latent,
                'loss': loss,
                'reconstructions': self.unpatchify(logits),
            }
        return {
            'latents': cls_latent,
            'loss': loss,
            'logits': logits,
        }


def topk(similarities, labels, k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities, axis=1)[:, -(i+1)] == labels) / len(labels)
    return topsum


def batch_wise_contrastive_loss(sim_matrix):
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
