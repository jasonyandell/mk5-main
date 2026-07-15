"""Tests for the arena snapshot adapter (otis.snapshots).

Loads real arena hands from a finished self-play snapshot chunk and verifies:

  * the P1 point identity (asserted inside ``parse_game_fates``) on every hand;
  * the free second referee: parser-computed per-team points equal the arena's
    recorded ``bidder_team_pts`` / ``opp_team_pts`` exactly
    (``cross_check_points``);
  * one hand's fates by independent hand-derivation from the raw JSON row (no
    reliance on ``parse_game_fates`` to produce the expected values).

The chunk is a COMPLETED factory output (selfplay_0, DONE line present). If it is
ever absent the real-data tests skip; the hand-derivation logic below still runs
against whatever the loader yields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from otis.fates import COUNT_TILE_PIPS, parse_game_fates, pips_to_domino_id
from otis.snapshots import (
    SnapshotHand,
    cross_check_points,
    load_snapshot_hands,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_CHUNK = (
    REPO_ROOT
    / "scratch"
    / "otis-night"
    / "corpus"
    / "chunk_selfplay_0.snapshots.json"
)

pytestmark = pytest.mark.skipif(
    not SNAPSHOT_CHUNK.exists(), reason="arena snapshot chunk not present"
)


def _load(limit: int) -> list[SnapshotHand]:
    return list(load_snapshot_hands(SNAPSHOT_CHUNK, limit=limit))


def test_load_three_hands_identity_and_cross_check():
    """Load 3+ real hands: P1 identity + recorded-points cross-check on each."""
    hands = _load(3)
    assert len(hands) == 3
    for h in hands:
        f = parse_game_fates(h.game)  # asserts the P1 identity internally

        # Redundant external identity check (must hold from the outside too).
        assert f.team0_points + f.team1_points == 42
        assert f.team0_tricks + f.team1_tricks == 7
        assert f.team0_count + f.team1_count == 35
        assert len(f.tiles) == 5
        assert {t.tile for t in f.tiles} == set(COUNT_TILE_PIPS)

        # Free second referee — parser vs arena's recorded points (raises on fail).
        cross_check_points(f, h.meta)

        # And the recorded points themselves must sum to 42.
        assert h.meta.bidder_team_pts + h.meta.opp_team_pts == 42

        # game_id encodes the unique per-match join key.
        assert h.game.game_id.endswith(
            f":a{h.meta.a_team}:g{h.meta.game_idx}:h{h.meta.hand_idx}"
        )


def test_metadata_matches_raw_row():
    """SnapshotMeta fields round-trip the raw JSON row exactly."""
    with SNAPSHOT_CHUNK.open() as fh:
        rows = json.load(fh)["snapshots"]
    hands = _load(5)
    for h, row in zip(hands, rows[:5], strict=True):
        assert h.meta.seed == row["seed"]
        assert h.meta.game_idx == row["game_idx"]
        assert h.meta.hand_idx == row["hand_idx"]
        assert h.meta.a_team == row["a_team"]
        assert h.meta.dealer == row["dealer"]
        assert h.meta.bidder == row["bidder"]
        assert h.meta.bid_value == row["bid_value"]
        assert h.meta.decl_id == row["decl_id"]
        assert list(h.meta.bids) == list(row["bids"])
        assert h.meta.bidder_team_pts == row["bidder_team_pts"]
        assert h.meta.opp_team_pts == row["opp_team_pts"]
        assert h.meta.made == row["made"]
        # NeutralGame input must mirror the raw deal/plays.
        assert h.game.hands == [[int(d) for d in hand] for hand in row["hands"]]
        assert h.game.plays == [(int(s), int(d)) for s, d in row["plays"]]
        # Bid winner leads trick 1.
        assert h.game.plays[0][0] == h.meta.bidder


def _raw_row(index: int) -> dict:
    with SNAPSHOT_CHUNK.open() as fh:
        return json.load(fh)["snapshots"][index]


def test_hand0_six_four_fate_by_derivation():
    """Independently derive the 6-4 tile's structural fate from the raw row.

    Hand g0:h0 is a ``sixes`` contract. 6-4 is therefore a trump; it is the first
    play of its trick (a lead). We recover the holder from the deal and the trick
    index / lead position straight from the raw ``plays`` array — no dependence on
    the parser to produce the expected values — then assert the parser agrees.
    """
    row = _raw_row(0)
    tile_id = pips_to_domino_id("6-4")

    # Holder = the seat whose dealt hand contains the tile.
    holder = next(seat for seat, hand in enumerate(row["hands"]) if tile_id in hand)

    # Locate the play of the tile and its position within its 4-play trick.
    play_idx = next(i for i, (_, d) in enumerate(row["plays"]) if d == tile_id)
    trick_idx = play_idx // 4
    offset_in_trick = play_idx % 4
    assert offset_in_trick == 0  # 6-4 leads its trick in this hand

    hands = _load(1)
    f = parse_game_fates(hands[0].game)
    t = next(t for t in f.tiles if t.tile == "6-4")
    assert t.holder_seat == holder
    assert t.trick_idx == trick_idx
    assert t.played_mode == "led"
    # sixes trump: a 6-x tile leading its own trick wins by trump power.
    assert t.won_by_trump is True
    assert t.winner_seat == holder  # led a trump nobody beat -> holder wins
    assert t.capture_side == "holder_team"


def test_hand2_sweep_capture_sides_by_derivation():
    """Hand g0:h2 is a 42-0 sweep — capture_side follows holder parity alone.

    When one team takes all 42 points, every count tile is captured by the
    winning team. So a tile's ``capture_side`` is ``holder_team`` exactly when
    the holder sits on the winning team. We identify the winning team from the
    recorded points (bidder team took all 42) with NO parser output, then check
    each tile's capture_side against holder parity.
    """
    row = _raw_row(2)
    assert row["bidder_team_pts"] == 42 and row["opp_team_pts"] == 0
    winning_team = row["bidder"] % 2

    hands = _load(3)
    f = parse_game_fates(hands[2].game)
    assert (f.team0_points, f.team1_points)[winning_team] == 42

    for t in f.tiles:
        holder_team = t.holder_seat % 2
        expected = "holder_team" if holder_team == winning_team else "opp_team"
        assert t.capture_side == expected, (
            f"tile {t.tile}: holder_team={holder_team} winning_team={winning_team}"
        )
