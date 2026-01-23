"""E[Q] (Expected Quality) training utilities.

Stage 2 operates on PUBLIC information only - what's visible during actual play.
GPU-native pipeline only.
"""
from __future__ import annotations

from .game import GameState
from .generate_gpu import (
    DecisionRecordGPU,
    GameRecordGPU,
    PosteriorConfig,
    generate_eq_games_gpu,
)
from .oracle import Stage1Oracle
from .sampling import sample_consistent_worlds
from .tokenize_gpu import GPUTokenizer, PastStatesGPU
from .transcript_tokenize import tokenize_transcript
from .types import ExplorationPolicy
from .voids import infer_voids

__all__ = [
    "DecisionRecordGPU",
    "ExplorationPolicy",
    "GPUTokenizer",
    "GameRecordGPU",
    "GameState",
    "PastStatesGPU",
    "PosteriorConfig",
    "Stage1Oracle",
    "generate_eq_games_gpu",
    "infer_voids",
    "sample_consistent_worlds",
    "tokenize_transcript",
]
