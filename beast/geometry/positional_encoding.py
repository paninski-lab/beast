"""Sinusoidal positional embedding utilities for 1D and 2D grids."""

import torch


def get_1d_sincos_pos_emb_from_grid(
    embed_dim: int,
    pos: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings from grid positions.

    Args:
        embed_dim: the embedding dimension (must be even).
        pos: the grid positions, shape [b * gh * gw] or [batch_size, sequence_length].
        device: device for the output tensor.

    Returns:
        sinusoidal positional embeddings of shape [len(pos), embed_dim].
    """
    assert embed_dim % 2 == 0, 'Embedding dimension must be even for sine and cosine.'

    pos = pos.float()

    dim = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
    freq = 1.0 / (10000 ** (dim / (embed_dim // 2)))

    pos_emb_sin = torch.sin(pos[:, None] * freq)
    pos_emb_cos = torch.cos(pos[:, None] * freq)

    pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)

    return pos_emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, int],
    device: str = 'cpu',
) -> torch.Tensor:
    """Generate 2D sine-cosine positional embeddings with separate grid height and width.

    Args:
        embed_dim: the embedding dimension.
        grid_size: tuple specifying the grid height and width (grid_h, grid_w).
        device: the device to place the embeddings on.

    Returns:
        positional embeddings of shape [grid_h*grid_w, embed_dim].
    """
    grid_h_size, grid_w_size = grid_size

    grid_h = torch.arange(grid_h_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_w_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
    grid = torch.stack(grid, dim=0)

    grid = grid.view(2, 1, grid.size(1), grid.size(2))

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device=device)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """Generate 2D sine-cosine positional embeddings from a grid.

    Args:
        embed_dim: the embedding dimension.
        grid: the grid of shape [2, 1, grid_h, grid_w].
        device: the device to place the embeddings on.

    Returns:
        positional embeddings of shape [grid_h*grid_w, embed_dim].
    """
    assert embed_dim % 2 == 0, 'Embedding dimension must be even.'

    grid_h = grid[0].view(-1)
    grid_w = grid[1].view(-1)

    emb_h = get_1d_sincos_pos_emb_from_grid(embed_dim // 2, grid_h, device=device)
    emb_w = get_1d_sincos_pos_emb_from_grid(embed_dim // 2, grid_w, device=device)

    pos_embed = torch.cat([emb_h, emb_w], dim=-1)
    return pos_embed
