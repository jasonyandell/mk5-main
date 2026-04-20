"""Engine-as-fact-checker for candlewax traces.

Given the ``ZebGameState`` before a commit and the domino_id the model chose,
apply the commit and fill in any remaining seats of the current trick using
the Stage-1 oracle's E[Q] argmax. Return a small structured record of what
actually happened:

    {
      "trick_winner_seat": 0,           # which seat leads the next trick
      "winning_team": 0 or 1,           # 0 = bidder's team, 1 = defense
      "count_captured_this_trick": 11,  # count points + 1 trick marker
      "my_team": 0 or 1,                # narrator's team
      "count_to_my_team": 0,            # bidder's perspective is "my"
      "count_to_opponents": 11,
      "filled_in": [[seat, domino_id], ...],  # plays we simulated for other seats
    }

This is the ground truth the model's reasoning ("my team gets 10 count")
can be checked against. No LLM judge required — the engine is the rules.
"""

from __future__ import annotations

from typing import Any

from burl.tools.eq_distribution import eq_outcome_distribution
from forge.zeb.game import apply_action, legal_actions
from forge.zeb.types import GamePhase


def _current_player(state) -> int:
    leader = getattr(state, "trick_leader", None)
    if leader is None:
        leader = getattr(state, "leader", 0)
    return (int(leader) + len(state.current_trick)) % 4


def _slot_of_domino(hand: tuple[int, ...] | list[int], domino_id: int) -> int | None:
    for i, d in enumerate(hand):
        if int(d) == int(domino_id):
            return i
    return None


def _pick_oracle_argmax_slot(
    state,
    legal_slots: tuple[int, ...],
    *,
    oracle,
    device: str,
    n_samples: int = 10,
) -> int:
    """Return the slot index with the highest E[Q] mean among legal options."""
    if not legal_slots:
        raise ValueError("no legal actions to pick from")
    if len(legal_slots) == 1:
        return legal_slots[0]

    cur = _current_player(state)
    hand = state.hands[cur]
    best_slot = legal_slots[0]
    best_mean = float("-inf")
    for slot in legal_slots:
        dom = int(hand[slot])
        try:
            dist = eq_outcome_distribution(
                state, dom,
                n_samples=n_samples,
                oracle=oracle, device=device,
                enumerate="auto",
                suggest_counterfactuals=False,
                include_spike_drivers=False,
            )
            m = float(dist.mean)
        except Exception:
            m = float("-inf")
        if m > best_mean:
            best_mean = m
            best_slot = slot
    return best_slot


def simulate_post_commit(
    state_before,
    committed_domino_id: int,
    *,
    narrator_seat: int,
    oracle,
    device: str,
    n_samples: int = 10,
) -> dict[str, Any]:
    """Apply the commit, fill in the trick, return the observed outcome.

    ``state_before`` is the game state at the moment of Qwen's decision (i.e.
    before the commit is applied). ``committed_domino_id`` is the domino the
    model played. ``narrator_seat`` is the seat whose team "my" refers to in
    the model's reasoning.
    """
    cur = _current_player(state_before)
    hand = state_before.hands[cur]
    slot = _slot_of_domino(hand, committed_domino_id)
    if slot is None:
        return {"error": f"domino {committed_domino_id} not in seat {cur} hand {list(hand)}"}

    pre_points = tuple(state_before.team_points)

    state = apply_action(state_before, slot)
    filled_in: list[list[int]] = []

    # Roll forward while we're still mid-trick.
    while (
        state.phase == GamePhase.PLAYING
        and 0 < len(state.current_trick) < 4
    ):
        legal = legal_actions(state)
        if not legal:
            break
        bot_slot = _pick_oracle_argmax_slot(
            state, legal, oracle=oracle, device=device, n_samples=n_samples,
        )
        bot_cur = _current_player(state)
        bot_dom = int(state.hands[bot_cur][bot_slot])
        filled_in.append([int(bot_cur), bot_dom])
        state = apply_action(state, bot_slot)

    post_points = tuple(state.team_points)
    delta = (post_points[0] - pre_points[0], post_points[1] - pre_points[1])

    if delta == (0, 0):
        # Trick didn't complete (shouldn't happen at trick 6 pos 1-4, but
        # safe to report a null outcome).
        return {
            "error": "trick did not complete",
            "pre_points": list(pre_points),
            "post_points": list(post_points),
            "phase": str(state.phase),
            "current_trick_len": len(state.current_trick),
        }

    winning_team = 0 if delta[0] > 0 else 1
    count_captured = int(max(delta))
    # state.trick_leader is the winner of the just-completed trick
    # (whether mid-hand or terminal).
    winner_seat = int(state.trick_leader)

    my_team = int(narrator_seat) % 2
    if my_team == winning_team:
        count_to_mine = count_captured
        count_to_theirs = 0
    else:
        count_to_mine = 0
        count_to_theirs = count_captured

    return {
        "trick_winner_seat": winner_seat,
        "winning_team": winning_team,
        "count_captured_this_trick": count_captured,
        "my_team": my_team,
        "count_to_my_team": count_to_mine,
        "count_to_opponents": count_to_theirs,
        "filled_in": filled_in,
        "pre_team_points": list(pre_points),
        "post_team_points": list(post_points),
    }
