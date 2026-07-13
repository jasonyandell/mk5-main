"""Unified evaluation framework for Crystal Forge players."""

from .engine import MatchConfig, run_match
from .players import PlayerSpec, parse_player_spec
from .results import HalfResult, MatchResult, compute_elo_ratings, format_elo

__all__ = [
    'MatchConfig',
    'run_match',
    'PlayerSpec',
    'parse_player_spec',
    'HalfResult',
    'MatchResult',
    'compute_elo_ratings',
    'format_elo',
]
