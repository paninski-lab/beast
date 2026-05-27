"""Transformer building blocks: attention layers, MLP, and transformer blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _init_weights(module: nn.Module) -> None:
    """Apply standard weight initialisation to Linear, Embedding, and Conv2d layers.

    Parameters
    ----------
    module: module to initialise.

    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def _init_weights_layerwise(module: nn.Module, weight_init_std: float) -> None:
    """Apply layerwise weight initialisation with a custom standard deviation.

    Parameters
    ----------
    module: module to initialise.
    weight_init_std: standard deviation for normal initialisation.

    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=weight_init_std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=weight_init_std)


class MLP(nn.Module):
    """MLP layer.

    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65

    """

    def __init__(
        self,
        d: int,
        mlp_ratio: int = 4,
        mlp_bias: bool = False,
        mlp_dropout: float = 0.0,
        mlp_dim: int | None = None,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        d: token dimension.
        mlp_ratio: hidden dimension multiplier when mlp_dim is None.
        mlp_bias: whether to include bias in linear layers.
        mlp_dropout: dropout probability after the output projection.
        mlp_dim: optional explicit hidden dimension; overrides mlp_ratio.

        """
        super().__init__()
        if mlp_dim is None:
            mlp_dim = d * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim, bias=mlp_bias),
            nn.GELU(),
            nn.Linear(mlp_dim, d, bias=mlp_bias),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.mlp(x)
        return x


class RMSNorm(nn.Module):
    """Root mean square layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize.

        Parameters
        ----------
        dim: feature dimension to normalise over.
        eps: epsilon added to variance for numerical stability.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation without the learnable scale.

        Parameters
        ----------
        x: input tensor.

        Returns
        -------
        normalised tensor with the same shape as x.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(x)


class QK_Norm_SelfAttention(nn.Module):
    """Self-attention with optional QK normalisation via RMSNorm."""

    def __init__(
        self,
        d: int,
        d_head: int,
        attn_qkv_bias: bool = False,
        attn_fc_bias: bool = True,
        attn_dropout: float = 0.0,
        attn_fc_dropout: float = 0.0,
        use_qk_norm: bool = True,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        d: token dimension.
        d_head: per-head dimension; must evenly divide d.
        attn_qkv_bias: whether to include bias in the QKV projection.
        attn_fc_bias: whether to include bias in the output projection.
        attn_dropout: dropout probability on attention weights.
        attn_fc_dropout: dropout probability after the output projection.
        use_qk_norm: whether to apply RMSNorm to Q and K before attention.

        """
        super().__init__()
        assert (
            d % d_head == 0
        ), f'Token dimension {d} should be divisible by head dimension {d_head}'
        self.d = d
        self.d_head = d_head
        self.attn_dropout = attn_dropout

        self.to_qkv = nn.Linear(d, 3 * d, bias=attn_qkv_bias)
        self.fc = nn.Linear(d, d, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = RMSNorm(d_head)
            self.k_norm = RMSNorm(d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: input tensor of shape (b, l, d).

        """
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        q, k, v = (rearrange(t, 'b l (nh dh) -> b nh l dh', dh=self.d_head) for t in (q, k, v))

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        dropout_p = self.attn_dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x = rearrange(x, 'b nh l dh -> b l (nh dh)')
        x = self.attn_fc_dropout(self.fc(x))
        return x


class QK_Norm_TransformerBlock(nn.Module):
    """Pre-norm transformer block with QK-normalised self-attention."""

    def __init__(
        self,
        d: int,
        d_head: int,
        ln_bias: bool = False,
        attn_qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_fc_bias: bool = False,
        attn_fc_dropout: float = 0.0,
        mlp_ratio: int = 4,
        mlp_bias: bool = False,
        mlp_dropout: float = 0.0,
        use_qk_norm: bool = True,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        d: token dimension.
        d_head: per-head dimension.
        ln_bias: whether to include bias in LayerNorm.
        attn_qkv_bias: whether to include bias in the QKV projection.
        attn_dropout: dropout probability on attention weights.
        attn_fc_bias: whether to include bias in the attention output projection.
        attn_fc_dropout: dropout probability after the attention output projection.
        mlp_ratio: MLP hidden dimension multiplier.
        mlp_bias: whether to include bias in MLP linear layers.
        mlp_dropout: dropout probability in the MLP.
        use_qk_norm: whether to apply RMSNorm to Q and K before attention.

        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d, bias=ln_bias)
        self.attn = QK_Norm_SelfAttention(
            d, d_head, attn_qkv_bias, attn_fc_bias, attn_dropout, attn_fc_dropout, use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
