"""Tests for the DinoV3 feature extractor."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from beast.nn.dino import DinoV3

_EMBED_DIM = 64
_N_TOKENS = 20  # CLS + registers + patches; patch_tokens = hidden[:, 5:]


def _make_mock_automodel(embed_dim: int = _EMBED_DIM, n_tokens: int = _N_TOKENS):
    """Return a mock that AutoModel.from_pretrained can be patched with."""
    mock_model = MagicMock()
    mock_model.config = SimpleNamespace(hidden_size=embed_dim)

    def _forward(pixel_values):
        bv = pixel_values.shape[0]
        hidden = torch.zeros(bv, n_tokens, embed_dim)
        return SimpleNamespace(last_hidden_state=hidden)

    mock_model.side_effect = _forward
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    return mock_model


class TestDinoV3:
    """Test the DinoV3 feature extractor."""

    def test_embed_dim_set_from_model_config(self) -> None:
        mock = _make_mock_automodel(embed_dim=128)
        with patch('beast.nn.dino.AutoModel.from_pretrained', return_value=mock):
            dino = DinoV3()
        assert dino.embed_dim == 128

    def test_freeze_true_freezes_parameters(self) -> None:
        param = torch.nn.Parameter(torch.zeros(4))
        mock = _make_mock_automodel()
        mock.parameters.return_value = iter([param])
        with patch('beast.nn.dino.AutoModel.from_pretrained', return_value=mock):
            DinoV3(freeze=True)
        assert not param.requires_grad

    def test_freeze_false_leaves_parameters_trainable(self) -> None:
        param = torch.nn.Parameter(torch.zeros(4))
        mock = _make_mock_automodel()
        mock.parameters.return_value = iter([param])
        with patch('beast.nn.dino.AutoModel.from_pretrained', return_value=mock):
            DinoV3(freeze=False)
        assert param.requires_grad

    def test_forward_output_shapes(self) -> None:
        b, v, h, w = 2, 3, 16, 16
        n_patch_tokens = _N_TOKENS - 5  # hidden[:, 5:]
        mock = _make_mock_automodel()
        with patch('beast.nn.dino.AutoModel.from_pretrained', return_value=mock):
            dino = DinoV3()
        images = torch.rand(b, v, 3, h, w)
        patch_tokens, cls_tokens = dino(images)
        assert patch_tokens.shape == (b, v, n_patch_tokens, _EMBED_DIM)
        assert cls_tokens.shape == (b, v, _EMBED_DIM)
