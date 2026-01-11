"""E[Q] (Expected Quality) training utilities.

Stage 2 operates on PUBLIC information only - what's visible during actual play.
"""
from __future__ import annotations

from .game import GameState
from .generate import DecisionRecord, GameRecord, generate_eq_game
from .oracle import Stage1Oracle
from .sampling import sample_consistent_worlds
from .transcript_tokenize import tokenize_transcript
from .voids import infer_voids

__all__ = [
    "DecisionRecord",
    "GameRecord",
    "GameState",
    "Stage1Oracle",
    "generate_eq_game",
    "infer_voids",
    "sample_consistent_worlds",
    "tokenize_transcript",
]
