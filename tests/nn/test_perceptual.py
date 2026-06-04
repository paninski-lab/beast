"""Tests for perceptual loss modules."""

import torch

from beast.nn.perceptual import AlexPerceptual, Perceptual


class TestPerceptual:
    """Test the base Perceptual class."""

    def test_forward(self) -> None:
        torch.manual_seed(0)
        mock_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            mock_net[0].weight.fill_(0.01)  # type: ignore[union-attr]
            mock_net[0].bias.zero_()  # type: ignore[union-attr]
        criterion = torch.nn.MSELoss()
        perceptual = Perceptual(network=mock_net, criterion=criterion)
        torch.manual_seed(1)
        x_hat = 0.01 * torch.randn((5, 3, 224, 224))
        x = 0.01 * torch.randn((5, 3, 224, 224))
        loss = perceptual(x_hat, x)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0


class TestAlexPerceptual:
    """Test the AlexPerceptual class."""

    def test_forward(self) -> None:
        torch.manual_seed(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = torch.nn.MSELoss()
        perceptual = AlexPerceptual(device=device, criterion=criterion)
        x_hat = torch.randn((5, 3, 224, 224), device=device)
        x = torch.randn((5, 3, 224, 224), device=device)
        loss = perceptual(x_hat, x)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_identical_inputs_produce_near_zero_loss(self) -> None:
        torch.manual_seed(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = torch.nn.MSELoss()
        perceptual = AlexPerceptual(device=device, criterion=criterion)
        x = torch.randn((2, 3, 224, 224), device=device)
        loss_same = perceptual(x, x)
        assert loss_same.item() < 1e-5

    def test_different_inputs_produce_nonzero_loss(self) -> None:
        torch.manual_seed(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = torch.nn.MSELoss()
        perceptual = AlexPerceptual(device=device, criterion=criterion)
        x = torch.randn((2, 3, 224, 224), device=device)
        x_hat = torch.randn((2, 3, 224, 224), device=device)
        loss_diff = perceptual(x_hat, x)
        assert loss_diff.item() > 0
