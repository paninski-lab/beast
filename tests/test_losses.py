"""Tests for beast.losses."""

import torch

from beast.losses import masked_mse_loss


class TestMaskedMseLoss:
    """Test the masked_mse_loss function."""

    def test_returns_scalar(self) -> None:
        rendering = torch.rand(2, 3, 4, 4)
        target = torch.rand(2, 3, 4, 4)
        mask = torch.ones(2, 1, 4, 4)
        loss = masked_mse_loss(rendering, target, mask)
        assert loss.ndim == 0

    def test_identical_inputs_give_zero_loss(self) -> None:
        x = torch.rand(2, 3, 4, 4)
        mask = torch.ones(2, 1, 4, 4)
        assert masked_mse_loss(x, x, mask).item() == 0.0

    def test_full_mask_matches_standard_mse(self) -> None:
        torch.manual_seed(0)
        rendering = torch.rand(2, 3, 8, 8)
        target = torch.rand(2, 3, 8, 8)
        mask = torch.ones(2, 1, 8, 8)
        loss = masked_mse_loss(rendering, target, mask)
        expected = torch.nn.functional.mse_loss(rendering, target)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_zero_mask_outside_region_ignores_those_pixels(self) -> None:
        # only top-left 2x2 pixels are unmasked; error only in that region
        rendering = torch.zeros(1, 3, 4, 4)
        target = torch.zeros(1, 3, 4, 4)
        target[:, :, :2, :2] = 1.0  # error only in the unmasked region
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :2, :2] = 1.0
        loss = masked_mse_loss(rendering, target, mask)
        assert loss.item() == 1.0

    def test_3d_mask_is_accepted(self) -> None:
        rendering = torch.rand(2, 3, 4, 4)
        target = torch.rand(2, 3, 4, 4)
        mask_3d = torch.ones(2, 4, 4)  # no channel dim
        loss = masked_mse_loss(rendering, target, mask_3d)
        assert loss.ndim == 0
        assert loss.item() >= 0
