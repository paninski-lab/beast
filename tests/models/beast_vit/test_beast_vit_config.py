"""Tests for the ViT backbone presets in beast.models.beast_vit.beast_vit_config."""

import pytest
from pydantic import ValidationError

from beast.models.beast_vit.beast_vit_config import (
    LEGACY_PRETRAINED,
    VIT_BACKBONES,
    VitModelParams,
    get_pretrained_repo,
    resolve_vit_params,
)


class TestResolveVitParams:
    """Test the function resolve_vit_params."""

    @pytest.mark.parametrize(
        'backbone,hidden_size,layers,heads',
        [
            ('vit-t', 192, 12, 3),
            ('vit-s', 384, 12, 6),
            ('vit-b', 768, 12, 12),
            ('vit-l', 1024, 24, 16)
        ],
    )
    def test_resolve_vit_params_presets(self, backbone, hidden_size, layers, heads):
        out = resolve_vit_params({'backbone': backbone, 'random_init': True})
        assert out['hidden_size'] == hidden_size
        assert out['num_hidden_layers'] == layers
        assert out['num_attention_heads'] == heads
        assert out['intermediate_size'] == 4 * hidden_size
        assert out['embed_size'] == hidden_size
        assert out['decoder_hidden_size'] == 512
        assert 'pretrained' not in out

    def test_resolve_vit_params_legacy_unchanged(self):
        params = {'hidden_size': 123, 'mask_ratio': 0.5}
        assert resolve_vit_params(params) == params

    def test_resolve_vit_params_keeps_explicit_embed_size(self):
        out = resolve_vit_params({'backbone': 'vit-l', 'embed_size': 256})
        assert out['embed_size'] == 256

    def test_resolve_vit_params_idempotent(self):
        once = resolve_vit_params({'backbone': 'vit-l'})
        assert resolve_vit_params(once) == once

    def test_resolve_vit_params_matching_explicit_field_ok(self):
        out = resolve_vit_params({'backbone': 'vit-b', 'hidden_size': 768})
        assert out['hidden_size'] == 768

    def test_resolve_vit_params_conflicting_field_raises(self):
        with pytest.raises(ValueError, match='conflicts with backbone'):
            resolve_vit_params({'backbone': 'vit-l', 'hidden_size': 768})

    def test_resolve_vit_params_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match='invalid entry'):
            resolve_vit_params({'backbone': 'vit-xl'})

    @pytest.mark.parametrize('backbone', ['vit-t', 'vit-s'])
    def test_resolve_vit_params_small_requires_random_init(self, backbone):
        with pytest.raises(ValueError, match='random_init'):
            resolve_vit_params({'backbone': backbone})


class TestGetPretrainedRepo:
    """Test the function get_pretrained_repo."""

    def test_get_pretrained_repo(self):
        assert get_pretrained_repo({'backbone': 'vit-b'}) == 'facebook/vit-mae-base'
        assert get_pretrained_repo({'backbone': 'vit-l'}) == 'facebook/vit-mae-large'
        assert get_pretrained_repo({'backbone': 'vit-s'}) is None
        assert get_pretrained_repo({'backbone': 'vit-t'}) is None

    def test_get_pretrained_repo_legacy(self):
        assert get_pretrained_repo({}) == LEGACY_PRETRAINED


class TestVitModelParams:
    """Test the VitModelParams schema with backbone aliases."""

    def test_vit_model_params_backbone_fills_fields(self):
        params = VitModelParams(backbone='vit-l')
        assert params.hidden_size == 1024
        assert params.num_hidden_layers == 24
        assert params.embed_size == 1024

    def test_vit_model_params_legacy_defaults_are_vit_b(self):
        params = VitModelParams()
        assert params.backbone is None
        for key, value in VIT_BACKBONES['vit-b'].items():
            if key != 'pretrained':
                assert getattr(params, key) == value

    def test_vit_model_params_roundtrip(self):
        dumped = VitModelParams(backbone='vit-l').model_dump()
        assert VitModelParams(**dumped).model_dump() == dumped

    def test_vit_model_params_invalid_backbone(self):
        with pytest.raises(ValidationError):
            VitModelParams(backbone='vit-xl')  # type: ignore[arg-type]

    def test_vit_model_params_small_without_random_init_raises(self):
        with pytest.raises(ValidationError, match='random_init'):
            VitModelParams(backbone='vit-s')
