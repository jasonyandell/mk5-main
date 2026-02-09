"""Named tokenizer registry for observation encoding.

Each model's config carries a ``"tokenizer": "v1"`` key.  Workers look up the
right GPU tokenizer at startup so two models with different encodings can run
side-by-side.

This module is **pure data** — no torch imports, no GPU code.  The GPU
tokenizer factory lives here but actual implementations register themselves
at import time (e.g. ``gpu_training_pipeline.py``).
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Feature & spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Feature:
    """One feature column in the token representation.

    Attributes:
        name:        Identifier used as the ``nn.Embedding`` attribute suffix
                     (e.g. ``"high_pip"`` → ``self.high_pip_embed``).
        cardinality: Number of distinct values (embedding num_embeddings).
        size:        ``'large'`` → base dim, ``'small'`` → base // 2.
    """
    name: str
    cardinality: int
    size: str  # 'large' | 'small'

    def __post_init__(self):
        if self.size not in ('large', 'small'):
            raise ValueError(f"size must be 'large' or 'small', got {self.size!r}")


@dataclass(frozen=True)
class TokenizerSpec:
    """Immutable specification for an observation tokenizer.

    Attributes:
        name:         Registry key (e.g. ``"v1"``).
        features:     Ordered tuple of Feature descriptors.
        max_tokens:   Sequence length (1 decl + hand slots + max plays).
        n_hand_slots: Fixed number of hand-slot positions.
    """
    name: str
    features: tuple[Feature, ...]
    max_tokens: int
    n_hand_slots: int

    @property
    def n_features(self) -> int:
        return len(self.features)


# ---------------------------------------------------------------------------
# V1 spec — matches current hard-coded constants exactly
# ---------------------------------------------------------------------------

V1_FEATURES: tuple[Feature, ...] = (
    Feature('high_pip',   7,  'large'),   # 0-6
    Feature('low_pip',    7,  'large'),   # 0-6
    Feature('is_double',  2,  'small'),   # 0-1
    Feature('count',      3,  'small'),   # 0-2
    Feature('player',     4,  'large'),   # 0-3 relative
    Feature('is_in_hand', 2,  'small'),   # 0-1
    Feature('decl',       10, 'large'),   # 0-9
    Feature('token_type', 3,  'large'),   # 0-2
)

V1_SPEC = TokenizerSpec(
    name='v1',
    features=V1_FEATURES,
    max_tokens=36,      # 1 decl + 7 hand + 28 plays
    n_hand_slots=7,
)


# ---------------------------------------------------------------------------
# Tokenizer spec registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, TokenizerSpec] = {}


def register_tokenizer(spec: TokenizerSpec) -> None:
    """Register a tokenizer spec.  Raises on duplicate names."""
    if spec.name in _REGISTRY:
        raise ValueError(f"tokenizer {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec


def get_tokenizer_spec(name: str) -> TokenizerSpec:
    """Look up a tokenizer spec by name.  Raises KeyError if unknown."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown tokenizer {name!r}; registered: {sorted(_REGISTRY)}")


# Register v1 immediately
register_tokenizer(V1_SPEC)


# ---------------------------------------------------------------------------
# GPU tokenizer factory
# ---------------------------------------------------------------------------

_GPU_FACTORIES: dict[str, type] = {}


def register_gpu_tokenizer(name: str, cls: type) -> None:
    """Register a GPU tokenizer class for *name*.

    The class must accept ``(device)`` as its constructor argument.
    """
    _GPU_FACTORIES[name] = cls


def get_gpu_tokenizer(name: str, device: object) -> object:
    """Instantiate the GPU tokenizer registered under *name*."""
    try:
        cls = _GPU_FACTORIES[name]
    except KeyError:
        raise KeyError(f"no GPU tokenizer registered for {name!r}; registered: {sorted(_GPU_FACTORIES)}")
    return cls(device)
