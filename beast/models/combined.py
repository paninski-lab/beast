from typing import Tuple

import torch
from torch import Tensor, tensor, zeros
from torch import nn

from loss.patch_wise import PatchWise
from loss.perceptual import SqueezePerceptual
# https://github.com/MLReproHub/SMAE/blob/main/src/loss/perceptual.p

class CombinedLoss(nn.Module):
    def __init__(self, pixel_criterion: nn.Module, perceptual_criterion: nn.Module, lambda_perceptual: float = 1.0,
                 calibration_steps: int = 10, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.pixel_loss = PatchWise(criterion=pixel_criterion)
        self.perceptual_loss = SqueezePerceptual(criterion=perceptual_criterion)
        self.register_buffer('lambda_perceptual', tensor(lambda_perceptual))
        # 0 Calibration steps disables calibration
        self.register_buffer('remaining_calibration_steps', tensor(calibration_steps))
        self.register_buffer('calibration_points', zeros((calibration_steps, 2)))

    @property
    def w_pixel(self) -> float:
        return self.pixel_scale.detach()

    @property
    def w_perceptual(self) -> float:
        return self.perceptual_scale.detach() * self.lambda_perceptual.detach()

    def forward(self, x_hat, x, patches_hat, patches) -> Tuple[Tensor, Tensor, Tensor]:
        pixel_loss = self.pixel_loss(patches_hat, patches)
        perceptual_loss = self.perceptual_loss(x_hat, x)

        # Calibrate the losses to be balanced during the first batches.
        if self.remaining_calibration_steps > 0:
            self.calibrate(pixel_loss, perceptual_loss)

        pixel_loss = self.pixel_scale * pixel_loss
        perceptual_loss = self.lambda_perceptual * self.perceptual_scale * perceptual_loss
        combined_loss = pixel_loss + perceptual_loss

        return combined_loss, pixel_loss, perceptual_loss

    def calibrate(self, initial_pixel_loss, initial_perceptual_loss):
        # Set the latest calibration point
        self.calibration_points[-self.remaining_calibration_steps] = 1 / tensor((initial_pixel_loss,
                                                                                initial_perceptual_loss))
        c = self.calibration_points
        # Update the scaling factors by taking the mean over all the populated calibration points
        pixel_scale, perceptual_scale = c[c.nonzero(as_tuple=True)].reshape(-1, 2).mean(dim=0)
        self.register_buffer('pixel_scale', pixel_scale)
        self.register_buffer('perceptual_scale', perceptual_scale)
        # Decrement the remaining calibration steps
        self.register_buffer('remaining_calibration_steps', self.remaining_calibration_steps - 1)