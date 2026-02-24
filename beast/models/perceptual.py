import torch
import torchvision
from torch import nn
from typing import Any
# https://github.com/MLReproHub/SMAE/blob/main/src/loss/perceptual.py


class Perceptual(nn.Module):
    def __init__(self, *, network: nn.Module, criterion: nn.Module):
        """Initialize perceptual loss module.

        Parameters
        ----------
        network: feature extractor that maps input images to feature tensors
        criterion: loss function applied to extracted features (e.g. MSELoss)
        """
        super(Perceptual, self).__init__()
        self.net = network
        self.criterion = criterion
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_hat_features = self.sigmoid(self.net(x_hat))
        x_features = self.sigmoid(self.net(x))
        loss = self.criterion(x_hat_features, x_features)
        return loss


class AlexPerceptual(Perceptual):
    def __init__(self, *, device: str | torch.device, **kwargs: Any):
        """Perceptual loss using pretrained AlexNet features [Pihlgren et al. 2020].

        Extracts features from the first five layers of AlexNet (pretrained on ImageNet)
        and computes loss between reconstructed and target feature maps.

        Parameters
        ----------
        device: device to run the feature extractor on (e.g. 'cuda', 'cpu')
        **kwargs: passed to parent; must include criterion (e.g. nn.MSELoss())
        """
        # Load alex net pretrained on IN1k
        alex_net = torchvision.models.alexnet(weights='IMAGENET1K_V1')
        # Extract features after second relu activation
        # Append sigmoid layer to normalize features
        perceptual_net = alex_net.features[:5].to(device)
        # Don't record gradients for the perceptual net, the gradients will still propagate through.
        for parameter in perceptual_net.parameters():
            parameter.requires_grad = False

        super(AlexPerceptual, self).__init__(network=perceptual_net, **kwargs)