"""Tests for perceptual loss modules."""

import torch

from beast.nn.perceptual import AlexPerceptual, VGGPerceptual


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

    def test_no_gradients_through_alexnet(self) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perceptual = AlexPerceptual(device=device, criterion=torch.nn.MSELoss())
        for param in perceptual.net.parameters():
            assert not param.requires_grad


class TestVGGPerceptual:
    """Test the VGGPerceptual class."""

    def test_forward(self) -> None:
        torch.manual_seed(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perceptual = VGGPerceptual(device=device)
        x_hat = torch.rand((2, 3, 64, 64), device=device)
        x = torch.rand((2, 3, 64, 64), device=device)
        loss = perceptual(x_hat, x)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_identical_inputs_produce_near_zero_loss(self) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perceptual = VGGPerceptual(device=device)
        torch.manual_seed(0)
        x = torch.rand((2, 3, 64, 64), device=device)
        loss = perceptual(x, x)
        assert loss.item() < 1e-5

    def test_different_inputs_produce_nonzero_loss(self) -> None:
        torch.manual_seed(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perceptual = VGGPerceptual(device=device)
        x = torch.rand((2, 3, 64, 64), device=device)
        x_hat = torch.rand((2, 3, 64, 64), device=device)
        loss = perceptual(x_hat, x)
        assert loss.item() > 0

    def test_no_gradients_through_vgg(self) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perceptual = VGGPerceptual(device=device)
        for param in perceptual.parameters():
            assert not param.requires_grad
