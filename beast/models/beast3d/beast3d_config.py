"""Pydantic config schemas for the BEAST3D multi-view 3DGS model.

BEAST3D is an ERayZer subclass, so its model config inherits every ERayZer model
field and adds only the BEAST3D-specific knobs (DINOv3 tokenizer, frustum
constraint, mask/alpha supervision, random-background compositing). Training and
optimizer schemas are shared with ERayZer; see ``beast.config.Beast3DBeastConfig``.
"""

from typing import Literal

from beast.models.erayzer.erayzer_config import ERayZerModelConfig


class Beast3DModelConfig(ERayZerModelConfig):
    """Model-section config for BEAST3D.

    Inherits all ERayZer model fields (``image_tokenizer``, ``target_image``,
    ``transformer``, ``pose_latent``, ``gaussians``, ``hard_pixelalign``, …).
    BEAST3D reads ground-truth cameras from the batch, so the pose-prediction
    branch is disabled by setting ``transformer.encoder_n_layer = 0``.
    """

    model_class: Literal['beast3d']

    # DINOv3 image tokenizer (replaces the learned patch tokenizer when enabled).
    # ViT-base/16 emits 768-dim patch tokens, so transformer.d must equal 768.
    use_dinov3: bool = True
    freeze_dinov3: bool = True
    dino_model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m'

    # restrict Gaussians to the intersection of all input view frustums
    frustum_constraint: bool = True

    # composite a random background into masked targets during training and
    # supervise the rendered alpha against the foreground mask
    random_background: bool = True
    mask_loss_weight: float = 0.1
