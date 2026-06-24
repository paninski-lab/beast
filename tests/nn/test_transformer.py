"""Tests for transformer building blocks."""

import pytest
import torch

from beast.nn.transformer import MLP, QK_Norm_SelfAttention, QK_Norm_TransformerBlock, RMSNorm

_B, _L, _D, _D_HEAD = 2, 8, 32, 8  # batch, seq len, dim, head dim


class TestMLP:
    """Test the MLP module."""

    def test_output_shape(self) -> None:
        mlp = MLP(d=_D)
        x = torch.randn(_B, _L, _D)
        assert mlp(x).shape == (_B, _L, _D)

    def test_explicit_mlp_dim(self) -> None:
        mlp = MLP(d=_D, mlp_dim=24)
        x = torch.randn(_B, _L, _D)
        assert mlp(x).shape == (_B, _L, _D)


class TestRMSNorm:
    """Test the RMSNorm module."""

    def test_output_shape_preserved(self) -> None:
        norm = RMSNorm(dim=_D)
        x = torch.randn(_B, _L, _D)
        assert norm(x).shape == (_B, _L, _D)

    def test_reduces_rms_to_near_one(self) -> None:
        norm = RMSNorm(dim=_D)
        # with unit weight the output RMS should be close to 1 along the last dim
        x = torch.randn(_B, _L, _D) * 5
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


class TestQK_Norm_SelfAttention:
    """Test the QK_Norm_SelfAttention module."""

    def test_output_shape(self) -> None:
        attn = QK_Norm_SelfAttention(d=_D, d_head=_D_HEAD)
        x = torch.randn(_B, _L, _D)
        assert attn(x).shape == (_B, _L, _D)

    def test_output_shape_without_qk_norm(self) -> None:
        attn = QK_Norm_SelfAttention(d=_D, d_head=_D_HEAD, use_qk_norm=False)
        x = torch.randn(_B, _L, _D)
        assert attn(x).shape == (_B, _L, _D)

    def test_invalid_head_dim_raises(self) -> None:
        with pytest.raises(AssertionError):
            QK_Norm_SelfAttention(d=_D, d_head=7)  # 32 % 7 != 0


class TestQK_Norm_TransformerBlock:
    """Test the QK_Norm_TransformerBlock module."""

    def test_output_shape(self) -> None:
        block = QK_Norm_TransformerBlock(d=_D, d_head=_D_HEAD)
        x = torch.randn(_B, _L, _D)
        assert block(x).shape == (_B, _L, _D)

    def test_output_differs_from_input(self) -> None:
        torch.manual_seed(0)
        block = QK_Norm_TransformerBlock(d=_D, d_head=_D_HEAD)
        x = torch.randn(_B, _L, _D)
        assert not torch.allclose(block(x), x)

    def test_output_shape_without_qk_norm(self) -> None:
        block = QK_Norm_TransformerBlock(d=_D, d_head=_D_HEAD, use_qk_norm=False)
        x = torch.randn(_B, _L, _D)
        assert block(x).shape == (_B, _L, _D)
