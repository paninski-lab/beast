"""Pydantic config schemas for the Vision Transformer autoencoder model."""

from typing import Any, Literal

from pydantic import BaseModel, model_validator

# encoder/decoder shapes for each named backbone; the MAE paper uses the same decoder for all
_DECODER: dict[str, int] = {
    'decoder_hidden_size': 512,
    'decoder_num_hidden_layers': 8,
    'decoder_num_attention_heads': 16,
    'decoder_intermediate_size': 2048,
}

# ``pretrained`` is the huggingface repo with MAE weights, or None if none exist
VIT_BACKBONES: dict[str, dict[str, Any]] = {
    'vit-t': {
        'hidden_size': 192,
        'num_hidden_layers': 12,
        'num_attention_heads': 3,
        'intermediate_size': 768,
        **_DECODER,
        'pretrained': None,
    },
    'vit-s': {
        'hidden_size': 384,
        'num_hidden_layers': 12,
        'num_attention_heads': 6,
        'intermediate_size': 1536,
        **_DECODER,
        'pretrained': None,
    },
    'vit-b': {
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        **_DECODER,
        'pretrained': 'facebook/vit-mae-base',
    },
    'vit-l': {
        'hidden_size': 1024,
        'num_hidden_layers': 24,
        'num_attention_heads': 16,
        'intermediate_size': 4096,
        **_DECODER,
        'pretrained': 'facebook/vit-mae-large',
    },
}

# legacy configs without ``backbone`` spell out the architecture and load vit-b weights
LEGACY_PRETRAINED = 'facebook/vit-mae-base'


def resolve_vit_params(params: dict) -> dict:
    """Expand a ``backbone`` alias into explicit architecture fields.

    Configs without ``backbone`` are returned unchanged (legacy behavior). The function is
    idempotent, so an already-resolved dict passes through.

    Parameters
    ----------
    params: ``model.model_params`` dict, with or without ``backbone``

    Returns
    -------
    new dict with the preset's architecture fields filled in; ``embed_size`` defaults to the
    preset's ``hidden_size``

    Raises
    ------
    ValueError: if ``backbone`` is unknown, if an architecture field conflicts with the
        preset, or if pretrained weights are requested for a backbone without any

    """
    backbone = params.get('backbone')
    if backbone is None:
        return dict(params)
    if backbone not in VIT_BACKBONES:
        raise ValueError(
            f'{backbone!r} is an invalid entry in model.model_params.backbone; '
            f'choose from {sorted(VIT_BACKBONES)}'
        )
    preset = {k: v for k, v in VIT_BACKBONES[backbone].items() if k != 'pretrained'}
    for key, value in preset.items():
        if params.get(key, value) != value:
            raise ValueError(
                f'model.model_params.{key}={params[key]} conflicts with backbone {backbone!r} '
                f'({key}={value}); remove it or drop the backbone field'
            )
    if VIT_BACKBONES[backbone]['pretrained'] is None and not params.get('random_init', False):
        raise ValueError(
            f'no pretrained MAE weights exist for {backbone!r}; '
            'set model.model_params.random_init: true'
        )
    resolved = {**params, **preset}
    resolved.setdefault('embed_size', preset['hidden_size'])
    return resolved


def get_pretrained_repo(params: dict) -> str | None:
    """Return the huggingface repo holding MAE weights for a (resolved) config, if any."""
    backbone = params.get('backbone')
    if backbone is None:
        return LEGACY_PRETRAINED
    return VIT_BACKBONES[backbone]['pretrained']


class VitModelParams(BaseModel):
    """Parameters for the ViT-MAE backbone and decoder.

    Set ``backbone`` to pick a preset size; the architecture fields below are then filled in
    automatically. Without ``backbone`` they are read as given (defaults match vit-b).
    """

    backbone: Literal['vit-t', 'vit-s', 'vit-b', 'vit-l'] | None = None
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = 'gelu'
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    decoder_num_attention_heads: int = 16
    decoder_hidden_size: int = 512
    decoder_num_hidden_layers: int = 8
    decoder_intermediate_size: int = 2048
    mask_ratio: float = 0.75
    norm_pix_loss: bool = False
    embed_size: int = 768
    temp_scale: bool = False
    random_init: bool = False
    use_infoNCE: bool = False
    infoNCE_weight: float = 0.03
    use_perceptual_loss: bool = False
    lambda_perceptual: float = 10.0

    @model_validator(mode='before')
    @classmethod
    def _expand_backbone(cls, data: Any) -> Any:
        """Fill in architecture fields from the ``backbone`` alias before validation."""
        return resolve_vit_params(data) if isinstance(data, dict) else data


class VitModelConfig(BaseModel):
    """Top-level model-section config for the Vision Transformer autoencoder."""

    model_class: Literal['vit']
    model_params: VitModelParams
    seed: int = 0
    checkpoint: str | None = None
