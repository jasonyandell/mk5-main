"""Unit tests for ``burl.tools.rules`` — the rule-answering tool surface.

These tests exercise each tool against real ``forge.zeb.game`` states (so we
get the same duck-typed contract the harness uses) plus a few synthesised
mini-states where we need a specific trick geometry.

No training, no Modal, no network. Pure CPU pytest.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import pytest

from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    N_DOMINOES,
    can_follow,
    led_suit_for_lead_domino,
)
from forge.zeb.game import apply_action, legal_actions, new_game

from burl.tools.rules import (
    contract_progress,
    count_dominoes_remaining,
    trick_winner_if,
    what_beats_what,
)


# --------------------------------------------------------------------------- #
# State fixtures                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def fresh_state():
    """Seed-2026 freshly-dealt state, skip_bidding=True."""
    return new_game(seed=2026, skip_bidding=True)


def _advance_n_plays(state, n: int, seed: int = 0):
    rng = random.Random(seed)
    cur = state
    for _ in range(n):
        slots = legal_actions(cur)
        if not slots:
            break
        cur = apply_action(cur, rng.choice(slots))
    return cur


# --------------------------------------------------------------------------- #
# count_dominoes_remaining                                                     #
# --------------------------------------------------------------------------- #


def test_count_dominoes_remaining_fresh_state(fresh_state):
    out = count_dominoes_remaining(fresh_state)

    # Double-six set has 3 five-pointers and 2 ten-pointers; all live at deal.
    assert len(out["unplayed_5pt"]) == 3
    assert len(out["unplayed_10pt"]) == 2
    assert out["played_5pt"] == []
    assert out["played_10pt"] == []
    assert out["loose_count_points"] == 35
    assert out["my_team_captured"] == 0
    assert out["opp_team_captured"] == 0

    # The 5 count dominoes are exactly {5-5, 6-4, 5-0, 4-1, 3-2}.
    ids_5 = {e["domino_id"] for e in out["unplayed_5pt"]}
    ids_10 = {e["domino_id"] for e in out["unplayed_10pt"]}
    labels_5 = {e["pip_label"] for e in out["unplayed_5pt"]}
    labels_10 = {e["pip_label"] for e in out["unplayed_10pt"]}

    assert labels_5 == {"3-2", "4-1", "5-0"}
    assert labels_10 == {"5-5", "6-4"}

    # ids match the canonical double-six ordering.
    assert ids_5 == {8, 11, 15}
    assert ids_10 == {20, 25}

    # Every entry carries count_value.
    for entry in out["unplayed_5pt"]:
        assert entry["count_value"] == 5
    for entry in out["unplayed_10pt"]:
        assert entry["count_value"] == 10


def test_count_dominoes_remaining_after_some_plays(fresh_state):
    cur = _advance_n_plays(fresh_state, 8, seed=7)
    out = count_dominoes_remaining(cur)

    n_live = len(out["unplayed_5pt"]) + len(out["unplayed_10pt"])
    n_played = len(out["played_5pt"]) + len(out["played_10pt"])
    assert n_live + n_played == 5
    assert out["loose_count_points"] == (
        5 * len(out["unplayed_5pt"]) + 10 * len(out["unplayed_10pt"])
    )
    # Per-entry counts line up with the scoring table for the engine.
    for entry in out["unplayed_5pt"] + out["played_5pt"]:
        assert DOMINO_COUNT_POINTS[entry["domino_id"]] == 5
    for entry in out["unplayed_10pt"] + out["played_10pt"]:
        assert DOMINO_COUNT_POINTS[entry["domino_id"]] == 10


def test_count_dominoes_remaining_serialisable(fresh_state):
    """All returned fields must be JSON primitives (ints/strs/bools/lists)."""
    import json

    payload = count_dominoes_remaining(fresh_state)
    json.dumps(payload)  # raises if any non-JSON type slipped in


# --------------------------------------------------------------------------- #
# trick_winner_if                                                              #
# --------------------------------------------------------------------------- #


def test_trick_winner_if_when_leading(fresh_state):
    """At the fresh state, me == trick_leader and current_trick is empty."""
    me_hand = fresh_state.hands[fresh_state.trick_leader]
    d = me_hand[0]
    out = trick_winner_if(fresh_state, int(d))

    assert out["completes_trick"] is False
    assert out["i_am_leading"] is True
    assert out["winner_seat_role"] is None
    assert out["winner_seat_absolute"] is None
    assert out["my_team_wins"] is None
    assert out["led_suit_if_played"] in {
        "blanks", "ones", "twos", "threes", "fours",
        "fives", "sixes", "called",
    }


def test_trick_winner_if_partial_trick(fresh_state):
    """Drive forward 2 plays so there's a partial (non-completing) trick."""
    cur = _advance_n_plays(fresh_state, 2, seed=3)
    assert 1 <= len(cur.current_trick) <= 3
    # Pick any legal slot for current player to simulate.
    slots = legal_actions(cur)
    assert slots, "expected legal plays"
    me = (cur.trick_leader + len(cur.current_trick)) % 4
    d = cur.hands[me][slots[0]]

    out = trick_winner_if(cur, int(d))
    # If we happen to land on position==3 (trick completing), skip.
    if out["completes_trick"]:
        pytest.skip("lucky random advance landed on a completing play")

    assert out["i_am_leading"] is False
    assert out["currently_leading_seat_role"] in {
        "me", "partner", "left_opponent", "right_opponent",
    }
    assert isinstance(out["currently_leading_seat_absolute"], int)
    assert isinstance(out["my_play_would_lead_partial"], bool)
    assert out["plays_remaining_after_me"] >= 1
    assert out["points_in_trick_so_far"] >= 0


def test_trick_winner_if_completes_trick(fresh_state):
    """Advance until a play has exactly 3 dominoes in the trick, then
    simulate the completing play. Under any declaration the trick must
    resolve to exactly 4 seats' worth of play and points ≥ 1 (trick point)."""
    cur = fresh_state
    for steps in range(1, 28):
        cur = _advance_n_plays(fresh_state, steps, seed=11)
        if len(cur.current_trick) == 3:
            break
    else:
        pytest.skip("could not reach a position=3 state within 28 steps")

    slots = legal_actions(cur)
    me = (cur.trick_leader + 3) % 4
    d = cur.hands[me][slots[0]]

    out = trick_winner_if(cur, int(d))
    assert out["completes_trick"] is True
    assert out["i_am_leading"] is False
    assert out["winner_seat_role"] in {
        "me", "partner", "left_opponent", "right_opponent",
    }
    assert 0 <= out["winner_seat_absolute"] < 4
    assert isinstance(out["my_team_wins"], bool)
    # Every trick is worth at least 1 (the trick point) and at most 36.
    assert 1 <= out["points_at_stake"] <= 36
    # Agreement: my_team_wins iff winner's parity equals my parity.
    me_abs = (cur.trick_leader + len(cur.current_trick)) % 4
    expected = (out["winner_seat_absolute"] % 2) == (me_abs % 2)
    assert out["my_team_wins"] is expected


def test_trick_winner_if_rejects_bad_domino_id(fresh_state):
    with pytest.raises(ValueError):
        trick_winner_if(fresh_state, 99)


# --------------------------------------------------------------------------- #
# what_beats_what                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class _MiniState:
    """Smallest possible duck-typed state for comparison tests.

    We only need decl_id + current_trick to satisfy what_beats_what when a
    trick is in progress (the tool can pull the lead off current_trick) and
    we need the full suite of fields when it isn't. Unused fields are set
    to reasonable defaults.
    """
    decl_id: int
    current_trick: tuple = field(default_factory=tuple)
    played: frozenset = field(default_factory=frozenset)
    play_history: tuple = field(default_factory=tuple)
    trick_leader: int = 0
    hands: tuple = field(
        default_factory=lambda: ((), (), (), ())
    )


def test_what_beats_what_trump_beats_offsuit_pip_declaration():
    """Fives-trump (decl 5): 5-0 (id=15) is trump; 6-0 (id=21) is not.
    If 6-0 leads, 5-0 follows via trump."""
    state = _MiniState(decl_id=5)  # fives-trump
    out = what_beats_what(state, 15, 21, lead_domino=21)
    assert out["winner"] == "a"
    assert out["a"]["is_trump"] is True
    assert out["b"]["is_trump"] is False
    assert out["led_suit"] == "sixes"
    assert "trump" in out["reason"]


def test_what_beats_what_higher_follower_under_notrump():
    """No-trump (decl 9): no trump power. If 6-0 leads, 6-5 beats 6-4."""
    state = _MiniState(decl_id=9)
    out = what_beats_what(state, 26, 25, lead_domino=21)  # 6-5 vs 6-4, lead 6-0
    assert out["winner"] == "a"
    assert out["a"]["is_trump"] is False
    assert out["b"]["is_trump"] is False
    assert out["a"]["can_follow"] is True
    assert out["b"]["can_follow"] is True
    assert "higher-ranked" in out["reason"]


def test_what_beats_what_uses_current_trick_when_lead_absent():
    """If a trick is in progress, lead_domino defaults to that trick's lead."""
    state = _MiniState(decl_id=5, current_trick=(21,))  # 6-0 already led
    out = what_beats_what(state, 15, 25)  # 5-0 (trump) vs 6-4 (not trump)
    assert out["winner"] == "a"
    assert out["lead_domino_id"] == 21


def test_what_beats_what_requires_lead_when_no_trick():
    """No trick in progress and no lead_domino passed → error."""
    state = _MiniState(decl_id=5)
    with pytest.raises(ValueError):
        what_beats_what(state, 15, 25)


def test_what_beats_what_rejects_bad_id():
    state = _MiniState(decl_id=5, current_trick=(21,))
    with pytest.raises(ValueError):
        what_beats_what(state, 99, 25)


def test_what_beats_what_both_offsuit_is_neither():
    """Twos-trump (decl 2), lead 6-5 (neither domino in suit or trump)."""
    state = _MiniState(decl_id=2)
    # 0-0 (id=0) and 3-3 (id=9) are neither twos nor sixes/fives.
    out = what_beats_what(state, 0, 9, lead_domino=26)  # lead 6-5
    assert out["winner"] == "neither"
    assert out["a"]["can_follow"] is False
    assert out["b"]["can_follow"] is False


# --------------------------------------------------------------------------- #
# contract_progress                                                            #
# --------------------------------------------------------------------------- #


def test_contract_progress_fresh_state(fresh_state):
    out = contract_progress(fresh_state)

    # skip_bidding always assigns a valid bidder.
    assert 0 <= out["bidder_seat_absolute"] < 4
    assert out["bidder_team"] in (0, 1)
    assert out["my_team_role"] in ("offense", "defense")
    assert out["bid_target"] >= 30
    # Fresh state: nobody captured anything, 35 count loose + 7 trick points.
    assert out["my_team_captured"] == 0
    assert out["opp_team_captured"] == 0
    assert out["loose_count_points"] == 35
    assert out["trick_points_remaining"] == 7
    assert out["tricks_completed"] == 0
    assert out["tricks_remaining"] == 7
    assert out["hand_points_remaining"] == 42
    # With no count captured and 42 points in play, offense can plausibly
    # still make a ≤42 bid ⇒ contract_in_play.
    assert out["status"] == "contract_in_play"
    assert out["offense_count_still_needed"] == out["bid_target"]
    assert 1 <= out["defense_points_to_set_bid"] <= 13


def test_contract_progress_offense_defense_labels(fresh_state):
    """my_team_role should depend on whether bidder_team == me's team."""
    out = contract_progress(fresh_state)
    me = (fresh_state.trick_leader + len(fresh_state.current_trick)) % 4
    my_team = me % 2
    if out["bidder_team"] == my_team:
        assert out["my_team_role"] == "offense"
        assert out["offense_captured"] == out["my_team_captured"]
        assert out["defense_captured"] == out["opp_team_captured"]
    else:
        assert out["my_team_role"] == "defense"
        assert out["offense_captured"] == out["opp_team_captured"]
        assert out["defense_captured"] == out["my_team_captured"]


def test_contract_progress_status_transitions():
    """Synthesise end-game-ish states and verify the status tag moves."""
    base = new_game(seed=42, skip_bidding=True)

    @dataclass
    class _Override:
        """Mutable shallow clone with the fields contract_progress reads."""
        decl_id: int
        hands: tuple
        played: frozenset
        play_history: tuple
        current_trick: tuple
        trick_leader: int
        bidder: int
        bid_state: Any
        team_points: tuple

    def _clone_with(points, played_ids, history_len):
        return _Override(
            decl_id=base.decl_id,
            hands=base.hands,
            played=frozenset(played_ids),
            play_history=tuple([(0, 0)] * history_len),
            current_trick=(),
            trick_leader=base.trick_leader,
            bidder=base.bidder,
            bid_state=base.bid_state,
            team_points=points,
        )

    # Resolve which team is offense, since tests must drive the offense total.
    me = (base.trick_leader + 0) % 4
    my_team = me % 2
    bidder_team = base.bidder % 2
    target = base.bid_state.high_bid

    # Case A: offense has already captured >= target → made.
    pts_a = (target, 0) if bidder_team == 0 else (0, target)
    state_a = _clone_with(pts_a, [], 0)
    out_a = contract_progress(state_a)
    assert out_a["status"] == "offense_has_made_bid"

    # Case B: hand has ended with offense below target → set.
    #   All 28 dominoes played, 0 count loose, 0 trick points remaining,
    #   offense captured < target.
    low = max(0, target - 1)
    pts_b = (low, 42 - low) if bidder_team == 0 else (42 - low, low)
    state_b = _clone_with(pts_b, list(range(N_DOMINOES)), 28)
    out_b = contract_progress(state_b)
    assert out_b["status"] == "defense_has_set_bid"

    # Case C: mid-hand, nobody at target yet, points remain.
    pts_c = (0, 0)
    state_c = _clone_with(pts_c, [], 4)  # 1 trick in; 6 left to play
    out_c = contract_progress(state_c)
    assert out_c["status"] == "contract_in_play"

    # Also verify hand_points_remaining arithmetic on a partially-played state.
    # 4 dominoes played, none of them counters → 35 loose + 6 trick points.
    state_d = _clone_with(pts_c, [0, 1, 2, 3], 4)
    out_d = contract_progress(state_d)
    assert out_d["hand_points_remaining"] == 35 + 6


def test_contract_progress_bidder_unknown_fallback(fresh_state):
    """If bidder is absent/invalid, tool reports 'bidder_unknown' instead of
    raising. Exercises the tolerant fallback path."""
    @dataclass
    class _NoBidder:
        decl_id: int
        hands: tuple
        played: frozenset
        play_history: tuple
        current_trick: tuple
        trick_leader: int
        # intentionally no bidder, no bid_state, no team_points.

    state = _NoBidder(
        decl_id=fresh_state.decl_id,
        hands=fresh_state.hands,
        played=frozenset(),
        play_history=(),
        current_trick=(),
        trick_leader=fresh_state.trick_leader,
    )
    out = contract_progress(state)
    assert out["status"] == "bidder_unknown"
    assert out["bidder_team"] is None
    assert out["my_team_role"] == "unknown"
    assert out["bid_target"] == 30  # floor fallback
    assert out["offense_captured"] is None
    assert out["offense_count_still_needed"] is None


# --------------------------------------------------------------------------- #
# Cross-tool sanity: the four tools agree with the engine they wrap            #
# --------------------------------------------------------------------------- #


def test_trick_winner_if_agrees_with_engine_under_self_play(fresh_state):
    """Drive forward to a 3-in-trick state, take the completing play via the
    engine's own resolver, and cross-check the winner trick_winner_if
    reported matches the apply_action outcome (state.team_points update)."""
    cur = fresh_state
    for steps in range(1, 28):
        cur = _advance_n_plays(fresh_state, steps, seed=101)
        if len(cur.current_trick) == 3:
            break
    else:
        pytest.skip("did not reach position=3 state in self-play")

    slots = legal_actions(cur)
    me = (cur.trick_leader + 3) % 4
    d = cur.hands[me][slots[0]]

    predicted = trick_winner_if(cur, int(d))
    before_points = cur.team_points

    # Now actually play that slot.
    after = apply_action(cur, slots[0])
    after_points = after.team_points

    # Whoever won should have a non-decreasing team_points entry.
    winner_team = predicted["winner_seat_absolute"] % 2
    delta = after_points[winner_team] - before_points[winner_team]
    assert delta == predicted["points_at_stake"], (
        f"engine added {delta} to team {winner_team} but trick_winner_if "
        f"predicted {predicted['points_at_stake']}"
    )


def test_what_beats_what_consistent_with_can_follow(fresh_state):
    """If A follows the led suit and B does not, and neither is trump, A wins."""
    cur = _advance_n_plays(fresh_state, 1, seed=5)
    lead = cur.current_trick[0]
    led_suit = led_suit_for_lead_domino(lead, cur.decl_id)
    # Find any pair (a, b) where a follows, b doesn't, neither is trump-capable.
    for a in range(N_DOMINOES):
        if not can_follow(a, led_suit, cur.decl_id):
            continue
        for b in range(N_DOMINOES):
            if b == a:
                continue
            if can_follow(b, led_suit, cur.decl_id):
                continue
            # Found a follow-vs-offsuit pair.
            out = what_beats_what(cur, a, b, lead_domino=int(lead))
            # a follows, b doesn't → a must rank higher unless b is trump.
            if not out["b"]["is_trump"]:
                assert out["winner"] == "a"
            return
    pytest.skip("no suitable follow-vs-offsuit pair found")
