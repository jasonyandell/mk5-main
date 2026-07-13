"""Tests for Bradley-Terry Elo rating computation."""

import pytest

from forge.zeb.eval.results import compute_elo_ratings, format_elo


def test_equal_players_equal_ratings():
    """Two players with equal records should get equal ratings."""
    names = ["A", "B"]
    wins = [
        [0, 50],
        [50, 0],
    ]
    ratings = compute_elo_ratings(names, wins, anchor="A", anchor_elo=1600.0)
    assert abs(ratings["A"] - ratings["B"]) < 1.0


def test_75pct_win_rate_elo_gap():
    """75% win rate corresponds to ~190 Elo gap in Bradley-Terry."""
    names = ["strong", "weak"]
    wins = [
        [0, 750],
        [250, 0],
    ]
    ratings = compute_elo_ratings(names, wins, anchor="weak", anchor_elo=1600.0)
    gap = ratings["strong"] - ratings["weak"]
    # BT 75% → Elo gap ~191. Allow ±10 for numerical tolerance.
    assert 180 < gap < 200, f"Expected ~190 Elo gap, got {gap:.1f}"


def test_anchor_gets_anchor_elo():
    """The anchor player should receive exactly the anchor Elo."""
    names = ["X", "Y", "Z"]
    wins = [
        [0, 60, 70],
        [40, 0, 55],
        [30, 45, 0],
    ]
    ratings = compute_elo_ratings(names, wins, anchor="Y", anchor_elo=1500.0)
    assert abs(ratings["Y"] - 1500.0) < 0.1


def test_single_player():
    """Single player gets anchor Elo."""
    ratings = compute_elo_ratings(["solo"], [[0]], anchor_elo=1600.0)
    assert ratings["solo"] == 1600.0


def test_three_players_ordering():
    """Player ordering should match win rates: A > B > C."""
    names = ["A", "B", "C"]
    wins = [
        [0, 70, 80],   # A beats B 70%, C 80%
        [30, 0, 65],   # B beats A 30%, C 65%
        [20, 35, 0],   # C beats both less
    ]
    ratings = compute_elo_ratings(names, wins)
    assert ratings["A"] > ratings["B"] > ratings["C"]


def test_default_anchor_is_first():
    """When no anchor specified, first player is anchored."""
    names = ["first", "second"]
    wins = [[0, 60], [40, 0]]
    ratings = compute_elo_ratings(names, wins, anchor_elo=1600.0)
    assert abs(ratings["first"] - 1600.0) < 0.1


def test_format_elo_sorted_descending():
    """format_elo should list ratings highest first."""
    ratings = {"low": 1400.0, "mid": 1600.0, "high": 1800.0}
    text = format_elo(ratings)
    lines = text.strip().split('\n')
    assert "high" in lines[1]
    assert "mid" in lines[2]
    assert "low" in lines[3]


def test_format_elo_custom_label():
    """format_elo should use custom label."""
    ratings = {"a": 1600.0}
    text = format_elo(ratings, "Offensive Elo")
    assert text.startswith("Offensive Elo:")
