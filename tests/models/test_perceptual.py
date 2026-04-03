import torch

from beast.models.perceptual import AlexPerceptual, Perceptual


def test_perceptual_forward():
    """Test base Perceptual class forward pass with a simple mock network."""
    # Mock network: Conv2d that preserves spatial dimensions for AlexNet-like feature output
    torch.manual_seed(0)
    mock_net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        torch.nn.ReLU(inplace=True),
    )
    with torch.no_grad():
        # make weights small to avoid NaNs
        mock_net[0].weight.fill_(0.01)
        mock_net[0].bias.zero_()
    criterion = torch.nn.MSELoss()
    perceptual = Perceptual(network=mock_net, criterion=criterion)
    torch.manual_seed(1)
    x_hat = 0.01 * torch.randn((5, 3, 224, 224))
    x = 0.01 * torch.randn((5, 3, 224, 224))
    loss = perceptual(x_hat, x)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_alex_perceptual_forward():
    """Test AlexPerceptual forward pass with pretrained AlexNet features."""
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


def test_alex_perceptual_different_inputs_produce_different_loss():
    """Test that different inputs produce different loss values."""
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.MSELoss()
    perceptual = AlexPerceptual(device=device, criterion=criterion)
    x = torch.randn((2, 3, 224, 224), device=device)
    loss_same = perceptual(x, x)
    assert loss_same.item() < 1e-5
    x_hat = torch.randn((2, 3, 224, 224), device=device)
    loss_diff = perceptual(x_hat, x)
    assert loss_diff.item() > 0
