"""Tests for the otis fate parser.

Two fully hand-constructed, legal 28-play games with manually-derived fates cover
every required scenario:

  * count tile led and held           (GAME_A: 6-4, 5-5 ; GAME_B: 4-1)
  * count fed to partner, following   (GAME_A: 4-1 followed, won by partner)
  * count sloughed to opponent trick  (GAME_A: 3-2 ; GAME_B: 3-2)
  * trumped-in capture                (GAME_B: 6-4 trumps in on a twos lead)
  * trump-suit-led edge               (GAME_B: trick 4 is a trump lead; 3-2 sloughs)
  * a double as count (5-5) captured  (GAME_A & GAME_B: 5-5)

Plus a corpus round-trip on real eq-corpus games (the parser asserts the P1
point identity internally, so a clean parse IS the invariant check).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from otis.fates import (
    COUNT_TILE_PIPS,
    NeutralGame,
    parse_game_fates,
    pips_to_domino_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_CHUNK = REPO_ROOT / "gus" / "data" / "corpus_train_chunk_0-99.pt"


# --------------------------------------------------------------------------- #
# Construction helper
# --------------------------------------------------------------------------- #


def mk_game(
    game_id: str,
    decl_id: int,
    bidder: int,
    hands_pips: list[list[str]],
    plays_pips: list[tuple[int, str]],
) -> NeutralGame:
    """Build a NeutralGame from pip-string hands and a pip-string play order."""
    hands = [[pips_to_domino_id(p) for p in hand] for hand in hands_pips]
    plays = [(seat, pips_to_domino_id(p)) for seat, p in plays_pips]
    return NeutralGame(
        game_id=game_id,
        hands=hands,
        decl_id=decl_id,
        bidder=bidder,
        bid_value=42,
        plays=plays,
    )


def tile_by_pip(fates, pip: str):
    for t in fates.tiles:
        if t.tile == pip:
            return t
    raise AssertionError(f"tile {pip} not found in {fates.game_id}")


# --------------------------------------------------------------------------- #
# GAME A — no-trump. All seven tricks swept by seat 3 (team 1).
# --------------------------------------------------------------------------- #

GAME_A = mk_game(
    game_id="A-notrump",
    decl_id=9,  # notrump
    bidder=3,
    hands_pips=[
        ["0-0", "1-0", "2-0", "3-0", "4-0", "5-0", "6-0"],  # S0 (5-0 count)
        ["1-1", "2-1", "3-1", "4-1", "5-1", "6-1", "2-2"],  # S1 (4-1 count)
        ["3-2", "4-2", "5-2", "6-2", "3-3", "4-3", "5-3"],  # S2 (3-2 count)
        ["6-3", "4-4", "5-4", "6-4", "5-5", "6-5", "6-6"],  # S3 (6-4, 5-5 count)
    ],
    plays_pips=[
        # trick 0: S3 leads 6-4, opponents forced to their only six; 6-4 wins (held).
        (3, "6-4"), (0, "6-0"), (1, "6-1"), (2, "6-2"),
        # trick 1: S3 leads 5-5 (double) wins; S0's 5-0 follows and is lost.
        (3, "5-5"), (0, "5-0"), (1, "5-1"), (2, "5-2"),
        # trick 2: S3 leads 4-4 wins; S1's 4-1 followed, won by partner S3.
        (3, "4-4"), (0, "4-0"), (1, "4-1"), (2, "4-2"),
        # trick 3: S3 leads 6-6; all void in sixes; S2 sloughs 3-2, lost to S3.
        (3, "6-6"), (0, "0-0"), (1, "1-1"), (2, "3-2"),
        # tricks 4-6: no count tiles; S3 keeps winning.
        (3, "6-3"), (0, "1-0"), (1, "2-1"), (2, "3-3"),
        (3, "6-5"), (0, "2-0"), (1, "3-1"), (2, "4-3"),
        (3, "5-4"), (0, "3-0"), (1, "2-2"), (2, "5-3"),
    ],
)


def test_game_a_totals():
    f = parse_game_fates(GAME_A)
    # Seat 3 (team 1) swept every trick and all count.
    assert f.team1_tricks == 7 and f.team0_tricks == 0
    assert f.team1_count == 35 and f.team0_count == 0
    assert f.team1_points == 42 and f.team0_points == 0
    assert f.team0_points + f.team1_points == 42
    assert f.decl_name == "notrump"


def test_game_a_count_led_and_held():
    f = parse_game_fates(GAME_A)
    t = tile_by_pip(f, "6-4")
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (3, 0, "led")
    assert t.winner_seat == 3 and t.capture_side == "holder_team"
    assert t.won_by_trump is False


def test_game_a_double_count_captured():
    f = parse_game_fates(GAME_A)
    t = tile_by_pip(f, "5-5")  # 5-5 is a double AND a count tile
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (3, 1, "led")
    assert t.capture_side == "holder_team"


def test_game_a_count_followed_lost_to_opponent():
    f = parse_game_fates(GAME_A)
    t = tile_by_pip(f, "5-0")  # S0 follows suit into S3's win
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (0, 1, "followed")
    assert t.winner_seat == 3 and t.capture_side == "opp_team"


def test_game_a_count_fed_to_partner_following():
    f = parse_game_fates(GAME_A)
    t = tile_by_pip(f, "4-1")  # S1 follows suit; partner S3 wins the trick
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (1, 2, "followed")
    assert t.winner_seat == 3 and t.capture_side == "holder_team"


def test_game_a_count_sloughed_to_opponent():
    f = parse_game_fates(GAME_A)
    t = tile_by_pip(f, "3-2")  # S2 void in sixes, sloughs into S3's win
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (2, 3, "sloughed")
    assert t.winner_seat == 3 and t.capture_side == "opp_team"


# --------------------------------------------------------------------------- #
# GAME B — sixes trump. Exercises trumped_in and a trump-suit-led trick.
# --------------------------------------------------------------------------- #

GAME_B = mk_game(
    game_id="B-sixes",
    decl_id=6,  # sixes trump
    bidder=0,
    hands_pips=[
        ["6-6", "6-0", "0-0", "1-0", "2-0", "3-0", "5-0"],  # S0 (5-0 count)
        ["6-1", "5-5", "1-1", "2-1", "3-1", "4-1", "5-1"],  # S1 (5-5, 4-1 count)
        ["6-2", "3-2", "2-2", "4-2", "5-2", "3-3", "4-3"],  # S2 (3-2 count)
        ["6-5", "6-4", "6-3", "4-0", "5-3", "4-4", "5-4"],  # S3 (6-4 count, trump)
    ],
    plays_pips=[
        # trick 0: S0 leads 2-0 (twos); S3 is void in twos, trumps in with 6-4 and wins.
        (0, "2-0"), (1, "2-1"), (2, "2-2"), (3, "6-4"),
        # trick 1: S3 leads trump 6-5; S0 overtrumps with 6-6 (trump-led edge).
        (3, "6-5"), (0, "6-6"), (1, "6-1"), (2, "6-2"),
        # trick 2: S0 leads 5-0; S1's 5-5 (double) follows and wins.
        (0, "5-0"), (1, "5-5"), (2, "5-2"), (3, "5-4"),
        # trick 3: S1 leads 4-1; partner S3 wins with 4-4; S0 void sloughs 0-0.
        (1, "4-1"), (2, "4-2"), (3, "4-4"), (0, "0-0"),
        # trick 4: S3 leads trump 6-3; S2 void in trump sloughs 3-2, won by trump.
        (3, "6-3"), (0, "6-0"), (1, "1-1"), (2, "3-2"),
        # tricks 5-6: no count tiles.
        (3, "5-3"), (0, "1-0"), (1, "5-1"), (2, "3-3"),
        (3, "4-0"), (0, "3-0"), (1, "3-1"), (2, "4-3"),
    ],
)


def test_game_b_totals():
    f = parse_game_fates(GAME_B)
    assert f.team1_tricks == 5 and f.team0_tricks == 2
    assert f.team1_count == 35 and f.team0_count == 0
    assert f.team1_points == 40 and f.team0_points == 2
    assert f.team0_points + f.team1_points == 42
    assert f.decl_name == "sixes"


def test_game_b_trumped_in_capture():
    f = parse_game_fates(GAME_B)
    t = tile_by_pip(f, "6-4")  # trump count tile played in while void in twos
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (3, 0, "trumped_in")
    assert t.winner_seat == 3 and t.capture_side == "holder_team"
    assert t.won_by_trump is True


def test_game_b_trump_led_slough_won_by_trump():
    f = parse_game_fates(GAME_B)
    t = tile_by_pip(f, "3-2")  # sloughed onto a trump-led trick, won by trump
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (2, 4, "sloughed")
    assert t.winner_seat == 3 and t.capture_side == "opp_team"
    assert t.won_by_trump is True


def test_game_b_double_count_followed_and_held():
    f = parse_game_fates(GAME_B)
    t = tile_by_pip(f, "5-5")
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (1, 2, "followed")
    assert t.winner_seat == 1 and t.capture_side == "holder_team"


def test_game_b_count_led_and_held():
    f = parse_game_fates(GAME_B)
    t = tile_by_pip(f, "4-1")  # S1 leads 4-1, partner S3 wins
    assert (t.holder_seat, t.trick_idx, t.played_mode) == (1, 3, "led")
    assert t.winner_seat == 3 and t.capture_side == "holder_team"


# --------------------------------------------------------------------------- #
# Structural invariants on the hand-built games
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("game", [GAME_A, GAME_B], ids=["A", "B"])
def test_every_count_tile_has_a_fate(game):
    f = parse_game_fates(game)
    assert {t.tile for t in f.tiles} == set(COUNT_TILE_PIPS)
    assert len(f.tiles) == 5


# --------------------------------------------------------------------------- #
# Corpus round-trip on real eq-corpus games
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not CORPUS_CHUNK.exists(), reason="eq corpus chunk not present")
def test_corpus_roundtrip():
    from otis.corpus import load_corpus_games  # local import: needs torch

    games = list(load_corpus_games(CORPUS_CHUNK, limit=3))
    assert len(games) == 3
    for g in games:
        f = parse_game_fates(g)  # asserts the P1 identity internally
        # Redundant external checks — the invariant must hold from the outside too.
        assert f.team0_points + f.team1_points == 42
        assert f.team0_tricks + f.team1_tricks == 7
        assert f.team0_count + f.team1_count == 35
        assert len(f.tiles) == 5
        assert {t.tile for t in f.tiles} == set(COUNT_TILE_PIPS)
        for t in f.tiles:
            assert t.played_mode in ("led", "followed", "trumped_in", "sloughed")
            assert t.capture_side in ("holder_team", "opp_team")
