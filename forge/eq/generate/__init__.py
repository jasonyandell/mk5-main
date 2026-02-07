"""GPU-native E[Q] generation package.

This package provides the GPU-native pipeline for generating E[Q] training data
by playing games using a Stage 1 oracle.

Main entry points:
- generate_eq_games_gpu: Main generation function
- PosteriorConfig: Configuration for posterior weighting
- AdaptiveConfig: Configuration for adaptive convergence-based sampling
- DecisionRecordGPU: Record for one decision
- GameRecordGPU: Record for one complete game
"""

from .adaptive import set_adaptive_log
from .pipeline import generate_eq_games_gpu
from .types import AdaptiveConfig, DecisionRecordGPU, GameRecordGPU, PosteriorConfig

# Re-export internal functions for tests that need them
from .actions import select_actions as _select_actions
from .deals import build_hypothetical_deals as _build_hypothetical_deals
from .sampling import sample_worlds_batched as _sample_worlds_batched
from .tokenization import tokenize_batched as _tokenize_batched

__all__ = [
    "generate_eq_games_gpu",
    "PosteriorConfig",
    "AdaptiveConfig",
    "DecisionRecordGPU",
    "GameRecordGPU",
    "set_adaptive_log",
    # Internal functions for tests
    "_select_actions",
    "_build_hypothetical_deals",
    "_sample_worlds_batched",
    "_tokenize_batched",
]
