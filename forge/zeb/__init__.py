"""Zeb: AlphaZero-style self-play learning for Texas 42.

This module implements self-play reinforcement learning that learns from
game outcomes rather than oracle labels.

Key components:
- types: Core data structures (ZebGameState, TrajectoryStep, etc.)
- game: Game state wrapper with full lifecycle support
- observation: Imperfect-information observation encoding
- model: Policy + value network architecture
- self_play: Batched trajectory generation
- module: PyTorch Lightning training module
- evaluate: Baseline players for evaluation

All public names are available via lazy imports: ``from forge.zeb import ZebModel``
works without eagerly loading every submodule (e.g. lightning).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# For static analysis / IDE autocomplete, expose the full API.
# At runtime these imports are skipped — __getattr__ handles them lazily.
if TYPE_CHECKING:
    from .types import (
        GamePhase,
        BidState,
        ZebGameState,
        TrajectoryStep,
        TrajectoryGame,
        TrainingConfig,
    )
    from .game import (
        new_game,
        apply_action,
        current_player,
        legal_actions,
        is_terminal,
        get_outcome,
    )
    from .observation import (
        observe,
        get_legal_mask,
        slot_to_domino,
        N_FEATURES,
        N_HAND_SLOTS,
        MAX_TOKENS,
        FEAT_HIGH,
        FEAT_LOW,
        FEAT_IS_DOUBLE,
        FEAT_COUNT,
        FEAT_PLAYER,
        FEAT_IS_IN_HAND,
        FEAT_DECL,
        FEAT_TOKEN_TYPE,
        TOKEN_TYPE_DECL,
        TOKEN_TYPE_HAND,
        TOKEN_TYPE_PLAY,
        PAD_VALUE,
    )
    from .model import ZebModel, ZebEmbeddings, get_model_config
    from .module import ZebLightningModule
    from .self_play import play_games_batched, trajectories_to_batch
    from .evaluate import (
        RandomPlayer,
        RuleBasedPlayer,
        NeuralPlayer,
        play_match,
        evaluate_vs_random,
        evaluate_vs_heuristic,
    )

# Mapping: attribute name -> (submodule, name_in_submodule)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # types
    "GamePhase": (".types", "GamePhase"),
    "BidState": (".types", "BidState"),
    "ZebGameState": (".types", "ZebGameState"),
    "TrajectoryStep": (".types", "TrajectoryStep"),
    "TrajectoryGame": (".types", "TrajectoryGame"),
    "TrainingConfig": (".types", "TrainingConfig"),
    # game
    "new_game": (".game", "new_game"),
    "apply_action": (".game", "apply_action"),
    "current_player": (".game", "current_player"),
    "legal_actions": (".game", "legal_actions"),
    "is_terminal": (".game", "is_terminal"),
    "get_outcome": (".game", "get_outcome"),
    # observation
    "observe": (".observation", "observe"),
    "get_legal_mask": (".observation", "get_legal_mask"),
    "slot_to_domino": (".observation", "slot_to_domino"),
    "N_FEATURES": (".observation", "N_FEATURES"),
    "N_HAND_SLOTS": (".observation", "N_HAND_SLOTS"),
    "MAX_TOKENS": (".observation", "MAX_TOKENS"),
    "FEAT_HIGH": (".observation", "FEAT_HIGH"),
    "FEAT_LOW": (".observation", "FEAT_LOW"),
    "FEAT_IS_DOUBLE": (".observation", "FEAT_IS_DOUBLE"),
    "FEAT_COUNT": (".observation", "FEAT_COUNT"),
    "FEAT_PLAYER": (".observation", "FEAT_PLAYER"),
    "FEAT_IS_IN_HAND": (".observation", "FEAT_IS_IN_HAND"),
    "FEAT_DECL": (".observation", "FEAT_DECL"),
    "FEAT_TOKEN_TYPE": (".observation", "FEAT_TOKEN_TYPE"),
    "TOKEN_TYPE_DECL": (".observation", "TOKEN_TYPE_DECL"),
    "TOKEN_TYPE_HAND": (".observation", "TOKEN_TYPE_HAND"),
    "TOKEN_TYPE_PLAY": (".observation", "TOKEN_TYPE_PLAY"),
    "PAD_VALUE": (".observation", "PAD_VALUE"),
    # model
    "ZebModel": (".model", "ZebModel"),
    "ZebEmbeddings": (".model", "ZebEmbeddings"),
    "get_model_config": (".model", "get_model_config"),
    # module (lightning)
    "ZebLightningModule": (".module", "ZebLightningModule"),
    # self_play
    "play_games_batched": (".self_play", "play_games_batched"),
    "trajectories_to_batch": (".self_play", "trajectories_to_batch"),
    # evaluate
    "RandomPlayer": (".evaluate", "RandomPlayer"),
    "RuleBasedPlayer": (".evaluate", "RuleBasedPlayer"),
    "NeuralPlayer": (".evaluate", "NeuralPlayer"),
    "play_match": (".evaluate", "play_match"),
    "evaluate_vs_random": (".evaluate", "evaluate_vs_random"),
    "evaluate_vs_heuristic": (".evaluate", "evaluate_vs_heuristic"),
}

__all__ = list(_LAZY_IMPORTS.keys()) + ["extract_model_config", "load_model"]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __package__)
        value = getattr(module, attr_name)
        # Cache on the module so __getattr__ isn't called again
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Shared checkpoint loading helpers
# ---------------------------------------------------------------------------

# Keys that identify model architecture (used for legacy config fallback).
_MODEL_CONFIG_KEYS = frozenset(
    ('embed_dim', 'n_heads', 'n_layers', 'ff_dim', 'dropout', 'max_tokens', 'belief_head')
)


def extract_model_config(ckpt: dict) -> dict:
    """Extract model_config from a checkpoint dict.

    Supports three checkpoint layouts:
      1. ``ckpt['model_config']``           -- top-level key (preferred)
      2. ``ckpt['config']['model_config']`` -- nested under 'config'
      3. Filter known model keys from ``ckpt['config']`` -- legacy flat format

    Raises ``ValueError`` if no model config can be found.
    """
    if 'model_config' in ckpt:
        return ckpt['model_config']
    if 'config' in ckpt and 'model_config' in ckpt['config']:
        return ckpt['config']['model_config']
    if 'config' in ckpt:
        config = ckpt['config']
        model_config = {k: v for k, v in config.items() if k in _MODEL_CONFIG_KEYS}
        if model_config:
            return model_config
    raise ValueError("Checkpoint missing model config")


def load_model(
    path: str,
    device: str = 'cpu',
    eval_mode: bool = True,
) -> tuple:
    """Load a ZebModel from a checkpoint file.

    Args:
        path: Path to the ``.pt`` checkpoint.
        device: Target device (e.g. ``'cpu'``, ``'cuda'``).
        eval_mode: Call ``model.eval()`` after loading (default ``True``).

    Returns:
        ``(model, ckpt)`` -- the loaded ZebModel and the raw checkpoint dict
        so callers can still access epoch, optimizer state, or other metadata.
    """
    import torch as _torch  # local to avoid top-level heavy import

    ckpt = _torch.load(path, map_location=device, weights_only=False)
    model_config = extract_model_config(ckpt)

    # Lazy-import ZebModel to keep the package lightweight at import time
    _ZebModel = __getattr__('ZebModel')
    model = _ZebModel(**model_config)

    # Old checkpoints lack belief_head weights — load with strict=False and warn
    if model_config.get('belief_head', False):
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            import warnings
            warnings.warn(f"Initializing missing belief head weights: {missing}")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    if eval_mode:
        model.eval()
    model.to(device)
    return model, ckpt
