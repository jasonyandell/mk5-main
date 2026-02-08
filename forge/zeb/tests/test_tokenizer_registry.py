"""Tests for the named tokenizer registry and spec-driven model."""

import pytest
import torch

from forge.zeb.tokenizer_registry import (
    Feature,
    TokenizerSpec,
    V1_SPEC,
    get_tokenizer_spec,
    register_tokenizer,
    register_gpu_tokenizer,
    get_gpu_tokenizer,
)
from forge.zeb.model import ZebModel, ZebEmbeddings, get_model_config


class TestFeature:
    def test_valid_sizes(self):
        Feature('x', 5, 'large')
        Feature('y', 3, 'small')

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="size must be"):
            Feature('z', 2, 'medium')

    def test_frozen(self):
        f = Feature('x', 5, 'large')
        with pytest.raises(AttributeError):
            f.name = 'y'


class TestTokenizerSpec:
    def test_n_features(self):
        spec = TokenizerSpec(
            name='test',
            features=(Feature('a', 3, 'large'), Feature('b', 2, 'small')),
            max_tokens=10,
            n_hand_slots=3,
        )
        assert spec.n_features == 2


class TestV1Spec:
    def test_v1_has_8_features(self):
        assert V1_SPEC.n_features == 8

    def test_v1_max_tokens(self):
        assert V1_SPEC.max_tokens == 36

    def test_v1_hand_slots(self):
        assert V1_SPEC.n_hand_slots == 7

    def test_v1_feature_names(self):
        names = [f.name for f in V1_SPEC.features]
        assert names == [
            'high_pip', 'low_pip', 'is_double', 'count',
            'player', 'is_in_hand', 'decl', 'token_type',
        ]


class TestRegistry:
    def test_lookup_v1(self):
        spec = get_tokenizer_spec('v1')
        assert spec is V1_SPEC

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown tokenizer"):
            get_tokenizer_spec('nonexistent')

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_tokenizer(V1_SPEC)


class TestGPUTokenizerFactory:
    def test_unknown_gpu_tokenizer_raises(self):
        with pytest.raises(KeyError, match="no GPU tokenizer"):
            get_gpu_tokenizer('nonexistent', 'cpu')


class TestSpecDrivenEmbeddings:
    """Verify spec-driven ZebEmbeddings produces identical state_dict keys and output shapes."""

    def test_state_dict_keys_match_old(self):
        """state_dict keys from spec-driven constructor must match the original hard-coded ones."""
        embed_dim = 128
        new_embed = ZebEmbeddings(embed_dim, spec=V1_SPEC)

        expected_keys = {
            'high_pip_embed.weight',
            'low_pip_embed.weight',
            'is_double_embed.weight',
            'count_embed.weight',
            'player_embed.weight',
            'is_in_hand_embed.weight',
            'decl_embed.weight',
            'token_type_embed.weight',
            'proj.weight',
            'proj.bias',
        }
        assert set(new_embed.state_dict().keys()) == expected_keys

    def test_state_dict_keys_match_old_small(self):
        """Also works with smaller embed_dim."""
        embed_dim = 64
        new_embed = ZebEmbeddings(embed_dim, spec=V1_SPEC)
        keys = set(new_embed.state_dict().keys())
        assert 'high_pip_embed.weight' in keys
        assert 'proj.weight' in keys

    def test_output_shape(self):
        """[2, 36, 8] input -> [2, 36, 128] output."""
        embed_dim = 128
        embed = ZebEmbeddings(embed_dim, spec=V1_SPEC)
        tokens = torch.zeros(2, 36, 8, dtype=torch.long)
        out = embed(tokens)
        assert out.shape == (2, 36, embed_dim)

    def test_embedding_dimensions(self):
        """Verify large features get base dim, small features get base//2."""
        embed_dim = 128
        embed = ZebEmbeddings(embed_dim, spec=V1_SPEC)
        base = embed_dim // 8  # = 16

        # large features: base=16
        assert embed.high_pip_embed.embedding_dim == base
        assert embed.player_embed.embedding_dim == base

        # small features: base//2=8
        assert embed.is_double_embed.embedding_dim == base // 2
        assert embed.count_embed.embedding_dim == base // 2

    def test_default_spec_is_v1(self):
        """Omitting spec defaults to v1."""
        embed = ZebEmbeddings(128)
        assert hasattr(embed, 'high_pip_embed')


class TestZebModelTokenizer:
    def test_default_tokenizer_v1(self):
        """ZebModel() without tokenizer kwarg defaults to v1."""
        model = ZebModel(embed_dim=64, n_heads=2, n_layers=2, ff_dim=128)
        assert hasattr(model.embeddings, 'high_pip_embed')

    def test_explicit_tokenizer_v1(self):
        """ZebModel(tokenizer='v1') works."""
        model = ZebModel(embed_dim=64, n_heads=2, n_layers=2, ff_dim=128, tokenizer='v1')
        tokens = torch.zeros(2, 36, 8, dtype=torch.long)
        mask = torch.ones(2, 36, dtype=torch.bool)
        hand_idx = torch.arange(1, 8).unsqueeze(0).expand(2, -1)
        hand_mask = torch.ones(2, 7, dtype=torch.bool)
        policy, value = model(tokens, mask, hand_idx, hand_mask)
        assert policy.shape == (2, 7)
        assert value.shape == (2,)

    def test_old_config_without_tokenizer(self):
        """Old configs that don't have 'tokenizer' key still work (defaults v1)."""
        old_config = dict(embed_dim=64, n_heads=2, n_layers=2, ff_dim=128)
        model = ZebModel(**old_config)
        assert hasattr(model.embeddings, 'high_pip_embed')

    def test_state_dict_compatible_with_old_model(self):
        """A model built with tokenizer='v1' should load state_dict from one built without."""
        old_config = dict(embed_dim=64, n_heads=2, n_layers=2, ff_dim=128)
        old_model = ZebModel(**old_config)
        old_sd = old_model.state_dict()

        new_config = dict(embed_dim=64, n_heads=2, n_layers=2, ff_dim=128, tokenizer='v1')
        new_model = ZebModel(**new_config)
        # Should load without errors
        new_model.load_state_dict(old_sd)


class TestGetModelConfig:
    def test_includes_tokenizer(self):
        config = get_model_config('small')
        assert config['tokenizer'] == 'v1'

    def test_custom_tokenizer(self):
        config = get_model_config('medium', tokenizer='v2')
        assert config['tokenizer'] == 'v2'

    def test_config_roundtrip(self):
        """get_model_config -> ZebModel(**config) works."""
        config = get_model_config('small')
        model = ZebModel(**config)
        assert model.embed_dim == 64


class TestObservationConstants:
    def test_observation_derives_from_spec(self):
        """observation.py constants should match V1_SPEC."""
        from forge.zeb.observation import N_FEATURES, MAX_TOKENS
        assert N_FEATURES == V1_SPEC.n_features
        assert MAX_TOKENS == V1_SPEC.max_tokens
