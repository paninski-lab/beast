"""Transformer building blocks: attention layers, MLP, and transformer blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def _init_weights(module: nn.Module) -> None:
    """Apply standard weight initialisation to Linear, Embedding, and Conv2d layers.

    Args:
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

    Args:
        module: module to initialise.
        weight_init_std: standard deviation for normal initialisation.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=weight_init_std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=weight_init_std)


def _init_weights_layerwise_correct(module: nn.Module, weight_init_std: float) -> None:
    """Apply layerwise weight initialisation with a custom standard deviation.

    Args:
        module: module to initialise.
        weight_init_std: standard deviation for normal initialisation.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=weight_init_std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=weight_init_std)


class ImageTokenizer(nn.Module):
    """Patch-based image tokenizer.

    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L134-L214
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        d: int,
        in_channels: int = 3,
        conv_bias: bool = False,
        patch_token_dropout: float = 0.0,
    ) -> None:
        """Initialize.

        Args:
            image_size: spatial size of the input image (assumed square).
            patch_size: spatial size of each patch (assumed square).
            d: token embedding dimension.
            in_channels: number of input image channels.
            conv_bias: whether to include bias in the patch embedding convolution.
            patch_token_dropout: dropout probability applied to patch tokens.
        """
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), f'Image size {image_size} must be divisible by the patch size {patch_size}.'
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.conv = nn.Conv2d(
            in_channels, d, kernel_size=patch_size, stride=patch_size, bias=conv_bias,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.patch_token_dropout = nn.Dropout(p=patch_token_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (b, c, h, w) --> (b, l, d)."""
        x = self.conv(x).flatten(2).transpose(1, 2)
        x = self.patch_token_dropout(x + self.pos_embed)
        return x


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

        Args:
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


class SelfAttention(nn.Module):
    """Self-attention layer.

    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        d: int,
        d_head: int,
        attn_qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_fc_bias: bool = False,
        attn_fc_dropout: float = 0.0,
        use_flashatt_v2: bool = False,
    ) -> None:
        """Initialize.

        Args:
            d: token dimension.
            d_head: per-head dimension; must evenly divide d.
            attn_qkv_bias: whether to include bias in the QKV projection.
            attn_dropout: dropout probability on attention weights.
            attn_fc_bias: whether to include bias in the output projection.
            attn_fc_dropout: dropout probability after the output projection.
            use_flashatt_v2: use xformers flash-attention v2 when available.
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

        self.use_flashatt_v2 = use_flashatt_v2

    def forward(self, x: torch.Tensor, subset_attention_size: int | None = None) -> torch.Tensor:
        """Forward pass with optional subset attention.

        Args:
            x: input tensor of shape (b, l, d).
            subset_attention_size: if set, restrict attention to a token subset.
        """
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        if self.use_flashatt_v2:
            q, k, v = map(
                lambda t: rearrange(t, 'b l (nh dh) -> b l nh dh', dh=self.d_head),
                (q, k, v),
            )

            if subset_attention_size is not None and subset_attention_size < q.shape[1]:
                x_subset = xops.memory_efficient_attention(
                    q[:, :subset_attention_size, :, :].contiguous(),
                    k[:, :subset_attention_size, :, :].contiguous(),
                    v[:, :subset_attention_size, :, :].contiguous(),
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
                x_rest = xops.memory_efficient_attention(
                    q[:, subset_attention_size:, :, :].contiguous(),
                    k,
                    v,
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
                x = torch.cat([x_subset, x_rest], dim=1)
            else:
                x = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
            x = rearrange(x, 'b l nh dh -> b l (nh dh)')
        else:
            q, k, v = (
                rearrange(q, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
                rearrange(k, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
                rearrange(v, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
            )
            dropout_p = self.attn_dropout if self.training else 0.0
            if subset_attention_size is not None and subset_attention_size < q.shape[2]:
                x = F.scaled_dot_product_attention(
                    q[:, :, :subset_attention_size, :].contiguous(),
                    k[:, :, :subset_attention_size, :].contiguous(),
                    v[:, :, :subset_attention_size, :].contiguous(),
                    dropout_p=dropout_p,
                )
                x_rest = F.scaled_dot_product_attention(
                    q[:, :, subset_attention_size:, :].contiguous(),
                    k,
                    v,
                    dropout_p=dropout_p,
                )
                x = torch.cat([x, x_rest], dim=2)
            else:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
                x = rearrange(x, 'b nh l dh -> b l (nh dh)')

        x = self.attn_fc_dropout(self.fc(x))
        return x


class RMSNorm(nn.Module):
    """Root mean square layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize.

        Args:
            dim: feature dimension to normalise over.
            eps: epsilon added to variance for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation without the learnable scale.

        Args:
            x: input tensor.

        Returns:
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

        Args:
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

    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (b, l, d).
            attn_bias: xformers BlockDiagonalMask.
        """
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        q, k, v = (rearrange(t, 'b l (nh dh) -> b l nh dh', dh=self.d_head) for t in (q, k, v))

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        use_xformers = (
            xops is not None and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)
        )
        if use_xformers:
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
            x = rearrange(x, 'b l nh dh -> b l (nh dh)')
        else:
            if attn_bias is not None:
                raise ValueError('attn_bias fallback requires xformers flash attention support')
            q, k, v = (
                rearrange(q, 'b l nh dh -> b nh l dh'),
                rearrange(k, 'b l nh dh -> b nh l dh'),
                rearrange(v, 'b l nh dh -> b nh l dh'),
            )
            dropout_p = self.attn_dropout if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            x = rearrange(x, 'b nh l dh -> b l (nh dh)')
        x = self.attn_fc_dropout(self.fc(x))
        return x


class QK_Norm_CrossAttention(nn.Module):
    """Cross-attention with optional QK normalisation via RMSNorm."""

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

        Args:
            d: token dimension.
            d_head: per-head dimension; must evenly divide d.
            attn_qkv_bias: whether to include bias in the Q/K/V projections.
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

        self.to_q = nn.Linear(d, d, bias=attn_qkv_bias)
        self.to_k = nn.Linear(d, d, bias=attn_qkv_bias)
        self.to_v = nn.Linear(d, d, bias=attn_qkv_bias)
        self.fc = nn.Linear(d, d, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = RMSNorm(d_head)
            self.k_norm = RMSNorm(d_head)

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor | None = None,
        attn_bias=None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            q_input: query tensor of shape (b, l, d).
            kv_input: key/value tensor; defaults to q_input for self-attention.
            attn_bias: xformers BlockDiagonalMask.
        """
        if kv_input is None:
            kv_input = q_input
        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        q, k, v = (rearrange(t, 'b l (nh dh) -> b l nh dh', dh=self.d_head) for t in (q, k, v))

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        use_xformers = (
            xops is not None
            and q_input.is_cuda
            and q_input.dtype in (torch.float16, torch.bfloat16)
        )
        if use_xformers:
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
            x = rearrange(x, 'b l nh dh -> b l (nh dh)')
        else:
            if attn_bias is not None:
                raise ValueError('attn_bias fallback requires xformers flash attention support')
            q, k, v = (
                rearrange(q, 'b l nh dh -> b nh l dh'),
                rearrange(k, 'b l nh dh -> b nh l dh'),
                rearrange(v, 'b l nh dh -> b nh l dh'),
            )
            dropout_p = self.attn_dropout if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            x = rearrange(x, 'b nh l dh -> b l (nh dh)')
        x = self.attn_fc_dropout(self.fc(x))
        return x


class QK_Norm_Attention(nn.Module):
    """General attention with separate Q, K, V projections and optional QK norm."""

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

        Args:
            d: token dimension.
            d_head: per-head dimension; must evenly divide d.
            attn_qkv_bias: whether to include bias in the Q/K/V projections.
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

        self.to_q = nn.Linear(d, d, bias=attn_qkv_bias)
        self.to_k = nn.Linear(d, d, bias=attn_qkv_bias)
        self.to_v = nn.Linear(d, d, bias=attn_qkv_bias)
        self.fc = nn.Linear(d, d, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = RMSNorm(d_head)
            self.k_norm = RMSNorm(d_head)

    def forward(
        self,
        q_input: torch.Tensor,
        k_input: torch.Tensor | None = None,
        v_input: torch.Tensor | None = None,
        attn_bias=None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            q_input: query input tensor of shape (b, l, d).
            k_input: key input; defaults to q_input.
            v_input: value input; defaults to q_input.
            attn_bias: xformers BlockDiagonalMask.
        """
        if k_input is None and v_input is None:
            k_input = v_input = q_input
        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = (rearrange(t, 'b l (nh dh) -> b l nh dh', dh=self.d_head) for t in (q, k, v))

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        use_xformers = (
            xops is not None
            and q_input.is_cuda
            and q_input.dtype in (torch.float16, torch.bfloat16)
        )
        if use_xformers:
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
            x = rearrange(x, 'b l nh dh -> b l (nh dh)')
        else:
            if attn_bias is not None:
                raise ValueError('attn_bias fallback requires xformers flash attention support')
            q, k, v = (
                rearrange(q, 'b l nh dh -> b nh l dh'),
                rearrange(k, 'b l nh dh -> b nh l dh'),
                rearrange(v, 'b l nh dh -> b nh l dh'),
            )
            dropout_p = self.attn_dropout if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            x = rearrange(x, 'b nh l dh -> b l (nh dh)')
        x = self.attn_fc_dropout(self.fc(x))
        return x


class MaskedSelfAttention(nn.Module):
    """Self-attention with an explicit boolean attention mask."""

    def __init__(
        self,
        d: int,
        d_head: int,
        attn_qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_fc_bias: bool = False,
        attn_fc_dropout: float = 0.0,
        use_qk_norm: bool = True,
    ) -> None:
        """Initialize.

        Args:
            d: token dimension.
            d_head: per-head dimension; must evenly divide d.
            attn_qkv_bias: whether to include bias in the QKV projection.
            attn_dropout: dropout probability on attention weights.
            attn_fc_bias: whether to include bias in the output projection.
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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        subset_attention_size: int | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (b, l, d).
            mask: boolean attention mask of shape (l, l).
            subset_attention_size: optional subset token count.
        """
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        q, k, v = (
            rearrange(q, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
            rearrange(k, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
            rearrange(v, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
        )

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])

        dropout_p = self.attn_dropout if self.training else 0.0
        if subset_attention_size is not None and subset_attention_size < q.shape[2]:
            x = F.scaled_dot_product_attention(
                q[:, :, :subset_attention_size, :].contiguous(),
                k[:, :, :subset_attention_size, :].contiguous(),
                v[:, :, :subset_attention_size, :].contiguous(),
                dropout_p=dropout_p,
                attn_mask=mask[:, :, :subset_attention_size, :subset_attention_size].contiguous(),
            )
            x_rest = F.scaled_dot_product_attention(
                q[:, :, subset_attention_size:, :].contiguous(),
                k,
                v,
                dropout_p=dropout_p,
                attn_mask=mask[:, :, subset_attention_size:, :].contiguous(),
            )
            x = torch.cat([x, x_rest], dim=2)
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, attn_mask=mask)
            x = rearrange(x, 'b nh l dh -> b l (nh dh)')

        x = self.attn_fc_dropout(self.fc(x))
        return x


class FastMaskAttention(nn.Module):
    """Self-attention with optional KV subset and xformers flash-attention support."""

    def __init__(
        self,
        d: int,
        d_head: int,
        attn_qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_fc_bias: bool = False,
        attn_fc_dropout: float = 0.0,
        use_flashatt_v2: bool = True,
        use_qk_norm: bool = False,
    ) -> None:
        """Initialize.

        Args:
            d: token dimension.
            d_head: per-head dimension; must evenly divide d.
            attn_qkv_bias: whether to include bias in the QKV projection.
            attn_dropout: dropout probability on attention weights.
            attn_fc_bias: whether to include bias in the output projection.
            attn_fc_dropout: dropout probability after the output projection.
            use_flashatt_v2: use xformers flash-attention v2 when available.
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

        self.use_flashatt_v2 = use_flashatt_v2

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(d_head)
            self.k_norm = RMSNorm(d_head)

    def forward(self, x: torch.Tensor, subset_kv_size: int | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (b, l, d).
            subset_kv_size: if set, restrict KV to a subset of tokens.
        """
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        if self.use_flashatt_v2:
            q, k, v = map(
                lambda t: rearrange(t, 'b l (nh dh) -> b l nh dh', dh=self.d_head),
                (q, k, v),
            )

            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            if subset_kv_size is not None and subset_kv_size < q.shape[1]:
                x = xops.memory_efficient_attention(
                    q,
                    k[:, subset_kv_size:, :, :].contiguous(),
                    v[:, subset_kv_size:, :, :].contiguous(),
                    attn_bias=None,
                    p=self.attn_dropout if self.training else 0.0,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
            else:
                x = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=None,
                    p=self.attn_dropout if self.training else 0.0,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
            x = rearrange(x, 'b l nh dh -> b l (nh dh)')
        else:
            q, k, v = (
                rearrange(q, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
                rearrange(k, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
                rearrange(v, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
            )

            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            dropout_p = self.attn_dropout if self.training else 0.0
            if subset_kv_size is not None and subset_kv_size < q.shape[2]:
                x = F.scaled_dot_product_attention(
                    q,
                    k[:, :, subset_kv_size:, :].contiguous(),
                    v[:, :, subset_kv_size:, :].contiguous(),
                    dropout_p=dropout_p,
                )
            else:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
                x = rearrange(x, 'b nh l dh -> b l (nh dh)')

        x = self.attn_fc_dropout(self.fc(x))
        return x


class SubsetAttention(nn.Module):
    """Attention supporting independent Q-subset or KV-subset masking."""

    def __init__(
        self,
        d: int,
        d_head: int,
        attn_qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_fc_bias: bool = False,
        attn_fc_dropout: float = 0.0,
        use_flashatt_v2: bool = True,
        use_qk_norm: bool = False,
    ) -> None:
        """Initialize.

        Args:
            d: token dimension.
            d_head: per-head dimension; must evenly divide d.
            attn_qkv_bias: whether to include bias in the QKV projection.
            attn_dropout: dropout probability on attention weights.
            attn_fc_bias: whether to include bias in the output projection.
            attn_fc_dropout: dropout probability after the output projection.
            use_flashatt_v2: use xformers flash-attention v2 when available.
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

        self.use_flashatt_v2 = use_flashatt_v2

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(d_head)
            self.k_norm = RMSNorm(d_head)

    def forward(
        self,
        x: torch.Tensor,
        subset_kv_size: int | None = None,
        subset_q_size: int | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (b, l, d).
            subset_kv_size: restrict KV to a token subset.
            subset_q_size: restrict Q to a token subset.
        """
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        assert not (subset_kv_size is not None and subset_q_size is not None), (
            'Only one of subset_kv_size or subset_q_size can be provided'
        )

        if self.use_flashatt_v2:
            q, k, v = map(
                lambda t: rearrange(t, 'b l (nh dh) -> b l nh dh', dh=self.d_head),
                (q, k, v),
            )

            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            if subset_kv_size is not None and subset_kv_size < k.shape[1]:
                x = xops.memory_efficient_attention(
                    q,
                    k[:, subset_kv_size:, :, :].contiguous(),
                    v[:, subset_kv_size:, :, :].contiguous(),
                    attn_bias=None,
                    p=self.attn_dropout if self.training else 0.0,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
            elif subset_q_size is not None and subset_q_size < q.shape[1]:
                x = xops.memory_efficient_attention(
                    q[:, :subset_q_size, :, :].contiguous(),
                    k,
                    v,
                    attn_bias=None,
                    p=self.attn_dropout if self.training else 0.0,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
            else:
                x = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=None,
                    p=self.attn_dropout if self.training else 0.0,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
            x = rearrange(x, 'b l nh dh -> b l (nh dh)')
        else:
            q, k, v = (
                rearrange(q, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
                rearrange(k, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
                rearrange(v, 'b l (nh dh) -> b nh l dh', dh=self.d_head),
            )

            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            dropout_p = self.attn_dropout if self.training else 0.0
            if subset_kv_size is not None and subset_kv_size < q.shape[2]:
                x = F.scaled_dot_product_attention(
                    q,
                    k[:, :, subset_kv_size:, :].contiguous(),
                    v[:, :, subset_kv_size:, :].contiguous(),
                    dropout_p=dropout_p,
                )
            else:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
                x = rearrange(x, 'b nh l dh -> b l (nh dh)')

        x = self.attn_fc_dropout(self.fc(x))
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with self-attention and MLP."""

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
    ) -> None:
        """Initialize.

        Args:
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
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d, bias=ln_bias)
        self.attn = SelfAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x: torch.Tensor, subset_attention_size: int | None = None) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x), subset_attention_size=subset_attention_size)
        x = x + self.mlp(self.norm2(x))
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

        Args:
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


class QK_Norm_Cross_TransformerBlock(nn.Module):
    """Pre-norm transformer block with QK-normalised cross-attention."""

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

        Args:
            d: token dimension.
            d_head: per-head dimension.
            ln_bias: whether to include bias in LayerNorm.
            attn_qkv_bias: whether to include bias in the Q/K/V projections.
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
        self.attn = QK_Norm_CrossAttention(
            d, d_head, attn_qkv_bias, attn_fc_bias, attn_dropout, attn_fc_dropout, use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: query input.
            y: key/value input.
        """
        x = x + self.attn(self.norm1(x), self.norm1(y))
        x = x + self.mlp(self.norm2(x))
        return x


class PAPR_QK_Norm_TransformerBlock(nn.Module):
    """PAPR-style transformer block with separate Q/K/V norms.

    Reference: https://github.com/zvict/papr/blob/main/models/model.py
    """

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

        Args:
            d: token dimension.
            d_head: per-head dimension.
            ln_bias: whether to include bias in LayerNorm.
            attn_qkv_bias: whether to include bias in the Q/K/V projections.
            attn_dropout: dropout probability on attention weights.
            attn_fc_bias: whether to include bias in the attention output projection.
            attn_fc_dropout: dropout probability after the attention output projection.
            mlp_ratio: MLP hidden dimension multiplier.
            mlp_bias: whether to include bias in MLP linear layers.
            mlp_dropout: dropout probability in the MLP.
            use_qk_norm: whether to apply RMSNorm to Q and K before attention.
        """
        super().__init__()
        self.norm1_q = nn.LayerNorm(d, bias=ln_bias)
        self.norm1_k = nn.LayerNorm(d, bias=ln_bias)
        self.norm1_v = nn.LayerNorm(d, bias=ln_bias)
        self.attn = QK_Norm_Attention(
            d, d_head, attn_qkv_bias, attn_fc_bias, attn_dropout, attn_fc_dropout, use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        q, k, v = self.norm1_q(q), self.norm1_k(k), self.norm1_v(v)
        x = q + self.attn(q, k, v)
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedTransformerBlock(nn.Module):
    """Transformer block with an explicit attention mask."""

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

        Args:
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
        self.attn = MaskedSelfAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout, use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        subset_attention_size: int | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x), mask, subset_attention_size=subset_attention_size)
        x = x + self.mlp(self.norm2(x))
        return x


class FaskMaskedTransformerBlock(nn.Module):
    """Fast masked transformer block using xformers flash attention."""

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

        Args:
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
        self.attn = FastMaskAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout,
            use_flashatt_v2=True, use_qk_norm=use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x: torch.Tensor, subset_attention_size: int | None = None) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x), subset_kv_size=subset_attention_size)
        x = x + self.mlp(self.norm2(x))
        return x


class QSubsetTransformerBlock(nn.Module):
    """Transformer block that attends only the first subset_attention_size query tokens."""

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

        Args:
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
        self.attn = SubsetAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout,
            use_flashatt_v2=True, use_qk_norm=use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x: torch.Tensor, subset_attention_size: int | None = None) -> torch.Tensor:
        """Forward pass."""
        x = x[:, :subset_attention_size, :] + self.attn(
            self.norm1(x), subset_q_size=subset_attention_size,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class KVSubsetTransformerBlock(nn.Module):
    """Transformer block that restricts KV to a token subset."""

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

        Args:
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
        self.attn = SubsetAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout,
            use_flashatt_v2=True, use_qk_norm=use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x: torch.Tensor, subset_attention_size: int | None = None) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x), subset_kv_size=subset_attention_size)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """Cross-attention with optional xformers flash-attention and causal masking."""

    def __init__(
        self,
        input_dim: int,
        d_head: int = 64,
        attn_qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_fc_bias: bool = False,
        attn_fc_dropout: float = 0.0,
        use_flashatt_v2: bool = True,
        num_heads: int | None = None,
        ctx_dim: int | None = None,
        causal: bool = False,
    ) -> None:
        """Initialize.

        Args:
            input_dim: query token dimension.
            d_head: per-head dimension.
            attn_qkv_bias: whether to include bias in the Q/K/V projections.
            attn_dropout: dropout probability on attention weights (must be 0.0).
            attn_fc_bias: whether to include bias in the output projection.
            attn_fc_dropout: dropout probability after the output projection.
            use_flashatt_v2: use xformers flash-attention v2 when available.
            num_heads: number of attention heads; defaults to input_dim // d_head.
            ctx_dim: key/value context dimension; defaults to input_dim.
            causal: whether to apply a causal (lower-triangular) mask.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_head = d_head
        self.num_heads = num_heads if num_heads is not None else input_dim // d_head
        self.ctx_dim = ctx_dim if ctx_dim is not None else input_dim
        self.att_dim = self.num_heads * self.d_head

        self.to_q = nn.Linear(self.input_dim, self.att_dim, bias=attn_qkv_bias)
        self.to_k = nn.Linear(self.ctx_dim, self.att_dim, bias=attn_qkv_bias)
        self.to_v = nn.Linear(self.ctx_dim, self.att_dim, bias=attn_qkv_bias)
        self.fc = nn.Linear(self.att_dim, self.input_dim, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)

        self.attn_dropout = attn_dropout
        assert self.attn_dropout == 0.0
        self.use_flashatt_v2 = use_flashatt_v2
        self.causal = causal

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass: x cross-attends to y.

        Args:
            x: query input of shape (b, l, d).
            y: key/value input of shape (b, l', d); defaults to x.
        """
        if y is None:
            y = x

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        if self.use_flashatt_v2:
            q, k, v = map(
                lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.num_heads),
                (q, k, v),
            )

            if self.causal:
                attention_bias = xops.LowerTriangularMask()
            else:
                attention_bias = None

            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attention_bias,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )

            x = rearrange(x, 'b n h d -> b n (h d)')
        else:
            q, k, v = map(
                lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                (q, k, v),
            )

            dropout_p = self.attn_dropout if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

            x = rearrange(x, 'b nh l dh -> b l (nh dh)')

        x = self.attn_fc_dropout(self.fc(x))
        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f'use_flashatt_v2={self.use_flashatt_v2}, '
            f'num_heads={self.num_heads}, '
            f'input_dim={self.input_dim}, '
            f'ctx_dim={self.ctx_dim}, '
            f'att_dim={self.att_dim}, '
        )


class CrossTransformerBlock(nn.Module):
    """Pre-norm cross-attention transformer block."""

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
    ) -> None:
        """Initialize.

        Args:
            d: token dimension.
            d_head: per-head dimension.
            ln_bias: whether to include bias in LayerNorm.
            attn_qkv_bias: whether to include bias in the Q/K/V projections.
            attn_dropout: dropout probability on attention weights.
            attn_fc_bias: whether to include bias in the attention output projection.
            attn_fc_dropout: dropout probability after the attention output projection.
            mlp_ratio: MLP hidden dimension multiplier.
            mlp_bias: whether to include bias in MLP linear layers.
            mlp_dropout: dropout probability in the MLP.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d, bias=ln_bias)

        self.attn = CrossAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x), self.norm1(y))
        x = x + self.mlp(self.norm2(x))
        return x


class FixedLengthTransformerLayer(nn.Module):
    """Transformer layer with fixed-length self-attention and optional cross-attention."""

    def __init__(
        self,
        dim: int,
        context_dim: int | None = None,
        fixed_length: int | None = None,
        num_heads: int = 8,
        head_dim: int = 64,
        use_ln_context: bool = True,
        mlp_dim: int | None = None,
    ) -> None:
        """Initialize.

        Args:
            dim: the input dim of x.
            context_dim: the input dim of context.
            fixed_length: the length of attention tokens.
            num_heads: the number of attention heads.
            head_dim: the dim of each attention head.
            use_ln_context: whether to apply LayerNorm to context.
            mlp_dim: optional MLP hidden dimension.
        """
        super().__init__()
        self.has_cross_att = context_dim is not None
        self.dim = dim
        self.fixed_length = fixed_length

        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(
            input_dim=dim,
            d_head=head_dim,
            num_heads=num_heads,
        )

        if self.has_cross_att:
            self.ln_cross = nn.LayerNorm(dim)
            if use_ln_context:
                self.ln_context = nn.LayerNorm(context_dim)
            else:
                self.ln_context = nn.Identity()
            self.cross_attn = CrossAttention(
                input_dim=dim,
                ctx_dim=context_dim,
                d_head=head_dim,
                num_heads=num_heads,
            )

        self.ln_fc = nn.LayerNorm(dim)
        self.fc = MLP(
            d=dim,
            mlp_dim=mlp_dim,
        )

    def init_weight(self, total_layers: int) -> None:
        """Scale output projections by 1/total_layers for stable initialisation."""
        linear_layer_list = [
            self.self_attn.fc,
            self.fc.mlp[2],
        ]
        if self.has_cross_att:
            linear_layer_list.append(self.cross_attn.fc)

        for linear_layer in linear_layer_list:
            linear_layer.weight.data /= total_layers
            if linear_layer.bias is not None:
                linear_layer.bias.data = 0.0

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape (b, orig_length, orig_dim).
            context: optional cross-attention context of shape (b, context_length, context_dim).
        """
        batch_size, orig_length, orig_dim = x.shape
        context = context or x

        assert orig_dim % self.dim == 0, f'orig_dim: {orig_dim}, dim: {self.dim}'
        if self.fixed_length is not None:
            assert (orig_length * orig_dim) % (self.fixed_length * self.dim) == 0, (
                f'orig_length: {orig_length}, token_length: {self.fixed_length}'
                f'orig_dim: {orig_dim}, dim: {self.dim}.'
                f'The product of orig_length * orig_dim must be divisible by token_length * dim.'
                f'O.w., it will break the batches'
            )

        x = x.reshape(-1, self.fixed_length or orig_length, self.dim)
        x = x + self.self_attn(self.ln_self(x))

        if self.has_cross_att:
            x = x.reshape(batch_size, -1, self.dim)
            x = x + self.cross_attn(self.ln_cross(x), self.ln_context(context))

        x = x + self.fc(self.ln_fc(x))

        x = x.reshape(batch_size, orig_length, orig_dim)

        return x

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f'token_length={self.fixed_length}, dim={self.dim}'
