"""Tests for sinusoidal positional encoding utilities."""

import pytest
import torch

from beast.geometry.positional_encoding import (
    get_1d_sincos_pos_emb_from_grid,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
)


class TestGet1dSincosPosEmbFromGrid:
    """Test the get_1d_sincos_pos_emb_from_grid function."""

    def test_output_shape(self) -> None:
        pos = torch.arange(10, dtype=torch.float32)
        emb = get_1d_sincos_pos_emb_from_grid(embed_dim=16, pos=pos)
        assert emb.shape == (10, 16)

    def test_single_position(self) -> None:
        pos = torch.tensor([5.0])
        emb = get_1d_sincos_pos_emb_from_grid(embed_dim=8, pos=pos)
        assert emb.shape == (1, 8)

    def test_odd_embed_dim_raises(self) -> None:
        pos = torch.arange(4, dtype=torch.float32)
        with pytest.raises(AssertionError):
            get_1d_sincos_pos_emb_from_grid(embed_dim=7, pos=pos)

    def test_values_bounded(self) -> None:
        pos = torch.linspace(0, 100, 50)
        emb = get_1d_sincos_pos_emb_from_grid(embed_dim=32, pos=pos)
        assert emb.abs().max().item() <= 1.0 + 1e-6

    def test_zero_position_sin_components_are_zero(self) -> None:
        # sin(0 * freq) = 0 for all freqs; cos(0 * freq) = 1
        pos = torch.tensor([0.0])
        emb = get_1d_sincos_pos_emb_from_grid(embed_dim=16, pos=pos)
        sin_half = emb[0, :8]
        cos_half = emb[0, 8:]
        assert torch.allclose(sin_half, torch.zeros(8), atol=1e-6)
        assert torch.allclose(cos_half, torch.ones(8), atol=1e-6)

    def test_different_positions_give_different_embeddings(self) -> None:
        pos = torch.tensor([0.0, 1.0, 2.0])
        emb = get_1d_sincos_pos_emb_from_grid(embed_dim=16, pos=pos)
        assert not torch.allclose(emb[0], emb[1])
        assert not torch.allclose(emb[1], emb[2])


class TestGet2dSincosPosEmbed:
    """Test the get_2d_sincos_pos_embed function."""

    def test_square_grid_output_shape(self) -> None:
        emb = get_2d_sincos_pos_embed(embed_dim=32, grid_size=(4, 4))
        assert emb.shape == (16, 32)

    def test_non_square_grid_output_shape(self) -> None:
        emb = get_2d_sincos_pos_embed(embed_dim=32, grid_size=(2, 8))
        assert emb.shape == (16, 32)

    def test_single_token_grid(self) -> None:
        emb = get_2d_sincos_pos_embed(embed_dim=16, grid_size=(1, 1))
        assert emb.shape == (1, 16)

    def test_embed_dim_is_full_width(self) -> None:
        embed_dim = 64
        emb = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=(3, 5))
        assert emb.shape[-1] == embed_dim

    def test_different_grid_positions_give_different_embeddings(self) -> None:
        emb = get_2d_sincos_pos_embed(embed_dim=32, grid_size=(4, 4))
        # at least some rows should differ — adjacent spatial positions differ
        assert not torch.allclose(emb[0], emb[1])


class TestGet2dSincosPosEmbedFromGrid:
    """Test the get_2d_sincos_pos_embed_from_grid function."""

    def _make_grid(self, h: int, w: int) -> torch.Tensor:
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='ij'), dim=0)
        return grid.view(2, 1, w, h)

    def test_output_shape(self) -> None:
        grid = self._make_grid(4, 4)
        emb = get_2d_sincos_pos_embed_from_grid(embed_dim=32, grid=grid)
        assert emb.shape == (16, 32)

    def test_odd_embed_dim_raises(self) -> None:
        grid = self._make_grid(2, 2)
        with pytest.raises(AssertionError):
            get_2d_sincos_pos_embed_from_grid(embed_dim=7, grid=grid)

    def test_matches_get_2d_sincos_pos_embed(self) -> None:
        # get_2d_sincos_pos_embed is a thin wrapper around this function;
        # results should be identical for the same logical grid
        h, w, d = 3, 5, 32
        emb_direct = get_2d_sincos_pos_embed(embed_dim=d, grid_size=(h, w))
        grid = self._make_grid(h, w)
        emb_from_grid = get_2d_sincos_pos_embed_from_grid(embed_dim=d, grid=grid)
        assert torch.allclose(emb_direct, emb_from_grid, atol=1e-6)

    def test_output_is_concatenation_of_h_and_w_halves(self) -> None:
        # first half of embedding dim comes from H coords, second from W
        h, w, d = 4, 4, 32
        grid = self._make_grid(h, w)
        emb = get_2d_sincos_pos_embed_from_grid(embed_dim=d, grid=grid)
        # each half is itself a valid 1D sincos embedding (values in [-1, 1])
        assert emb[:, :d // 2].abs().max().item() <= 1.0 + 1e-6
        assert emb[:, d // 2:].abs().max().item() <= 1.0 + 1e-6
