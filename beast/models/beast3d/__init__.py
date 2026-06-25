"""BEAST3D multi-view 3DGS model package (ERayZer with DINOv3 + GT cameras)."""

from beast.models.beast3d.beast3d_config import Beast3DModelConfig
from beast.models.beast3d.beast3d_model import Beast3D
from beast.models.beast3d.beast3d_train import train

__all__ = [
    'Beast3D',
    'Beast3DModelConfig',
    'train',
]
