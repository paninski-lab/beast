"""DINOv3 feature extractor for patch and CLS token extraction."""

import torch.nn as nn
from transformers import AutoModel


class DinoV3(nn.Module):
    """DINOv3 feature extractor returning patch and CLS tokens."""

    def __init__(
        self,
        model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        freeze: bool = True,
    ) -> None:
        """Initialize DINOv3.

        Parameters
        ----------
        model_name: HuggingFace model identifier.
        freeze: whether to freeze all parameters (default True).

        """
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        self.embed_dim = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, images):
        """Extract patch and CLS tokens from multi-view images.

        Parameters
        ----------
        images: float tensor of shape (B, V, 3, H, W) in [0, 1].

        Returns
        -------
        tuple of (patch_tokens [B, V, N, embed_dim], cls_tokens [B, V, embed_dim]).

        """
        B, V = images.shape[:2]

        x = images.view(B * V, *images.shape[2:])

        outputs = self.model(pixel_values=x)

        hidden = outputs.last_hidden_state

        cls_tokens = hidden[:, 0]
        patch_tokens = hidden[:, 5:]

        N = patch_tokens.shape[1]

        patch_tokens = patch_tokens.view(B, V, N, self.embed_dim)
        cls_tokens = cls_tokens.view(B, V, self.embed_dim)

        return patch_tokens, cls_tokens
