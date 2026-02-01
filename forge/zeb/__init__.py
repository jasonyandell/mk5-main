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
"""

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
    # Constants
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

__all__ = [
    # Types
    "GamePhase",
    "BidState",
    "ZebGameState",
    "TrajectoryStep",
    "TrajectoryGame",
    "TrainingConfig",
    # Game
    "new_game",
    "apply_action",
    "current_player",
    "legal_actions",
    "is_terminal",
    "get_outcome",
    # Observation
    "observe",
    "get_legal_mask",
    "slot_to_domino",
    "N_FEATURES",
    "N_HAND_SLOTS",
    "MAX_TOKENS",
    "FEAT_HIGH",
    "FEAT_LOW",
    "FEAT_IS_DOUBLE",
    "FEAT_COUNT",
    "FEAT_PLAYER",
    "FEAT_IS_IN_HAND",
    "FEAT_DECL",
    "FEAT_TOKEN_TYPE",
    "TOKEN_TYPE_DECL",
    "TOKEN_TYPE_HAND",
    "TOKEN_TYPE_PLAY",
    "PAD_VALUE",
    # Model
    "ZebModel",
    "ZebEmbeddings",
    "get_model_config",
    # Training
    "ZebLightningModule",
    "play_games_batched",
    "trajectories_to_batch",
    # Evaluation
    "RandomPlayer",
    "RuleBasedPlayer",
    "NeuralPlayer",
    "play_match",
    "evaluate_vs_random",
    "evaluate_vs_heuristic",
]
