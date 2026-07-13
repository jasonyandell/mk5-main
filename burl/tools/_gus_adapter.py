"""Adapter from Burl's duck-typed ``game_state`` -> Gus's tokenized input.

Burl's tools (see ``burl/tools/engine.py``) take a duck-typed state with fields
``hands`` (initial 4x7 deal), ``played`` (set of domino_ids played globally),
``play_history`` (tuple of (player, domino[, lead])), ``decl_id``, ``leader``
(or ``trick_leader``), and ``current_trick``.

Gus's tokenizer (``gus/model/tokenize.py::tokenize_decision``) wants a
``(game_hands, decl_id, decisions_list, decision_idx)`` quadruple where
``decisions_list`` is the corpus's per-decision records with ``.player`` and
``.action_taken`` (a SLOT index into the initial hand, not a domino_id).

The bridge: synthesize a minimal decisions-list by walking ``play_history``
and resolving each (player, domino_id) to a slot index in the initial hand.
The tokenizer only reads ``.player`` and ``.action_taken`` off each decision
(see ``features.py::reconstruct_prior_plays``), so a lightweight shim is
enough — no need to mock the full corpus decision struct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gus.model.features import reconstruct_prior_plays
from gus.model.tokenize import tokenize_decision
from gus.model.voids import voids_feature_vector


@dataclass
class _ShimDecision:
    """The minimal decision-record shape ``reconstruct_prior_plays`` reads."""
    player: int
    action_taken: int


def _current_player(state: Any) -> int:
    leader = getattr(state, "trick_leader", None)
    if leader is None:
        leader = state.leader
    return (int(leader) + len(state.current_trick)) % 4


def _normalize_play_history(state: Any) -> list[tuple[int, int]]:
    """Return (player, domino_id) list regardless of whether the state uses
    2-tuples (ZebGameState) or 3-tuples ``(player, domino, lead)``
    (forge.eq.game.GameState)."""
    history = state.play_history
    if not history:
        return []
    first = history[0]
    if len(first) == 3:
        return [(int(p), int(d)) for (p, d, _lead) in history]
    return [(int(p), int(d)) for (p, d) in history]


def _initial_hands_as_lists(state: Any) -> list[list[int]]:
    """Coerce ``state.hands`` into the ``list[list[int]]`` shape the tokenizer
    expects. Zeb stores them as tuples; normalize."""
    return [[int(d) for d in state.hands[p]] for p in range(4)]


def _slot_in_initial_hand(initial_hand: list[int], domino_id: int) -> int:
    """Return the slot (0..6) in ``initial_hand`` holding ``domino_id``.

    Raises ``ValueError`` if not present — this would indicate the history
    references a domino nobody was dealt, which is a bug in the caller's state.
    """
    for slot, d in enumerate(initial_hand):
        if int(d) == int(domino_id):
            return slot
    raise ValueError(
        f"domino {domino_id} not found in initial hand {initial_hand}; "
        f"play_history is inconsistent with hands"
    )


def build_gus_inputs(state: Any) -> dict[str, Any]:
    """Build Gus's forward-pass inputs from a Burl-shaped ``game_state``.

    Returns a dict with:
      - tokens         : [SEQ_LEN=33, 5] long
      - attention_mask : [SEQ_LEN] bool
      - voids          : [24] float
      - current_player : int (absolute seat, for caller convenience)
      - decision_idx   : int (how many plays have happened; equals len(history))
      - game_hands     : list[list[int]]  (for label reconstruction)
      - prior_plays    : list[(abs_player, domino)]  (for label reconstruction)
      - decl_id        : int
    """
    game_hands = _initial_hands_as_lists(state)
    decl_id = int(state.decl_id)
    history = _normalize_play_history(state)
    current_player = _current_player(state)

    # Build the shim decisions list: one _ShimDecision per completed play.
    decisions: list[_ShimDecision] = []
    for (p, dom) in history:
        slot = _slot_in_initial_hand(game_hands[p], dom)
        decisions.append(_ShimDecision(player=p, action_taken=slot))

    decision_idx = len(decisions)
    # Append a sentinel decision for the CURRENT play so the tokenizer tokenizes
    # state up to (but not including) decision_idx — i.e. everything already
    # played, not including the move we're about to commit. ``tokenize_decision``
    # reads decisions[decision_idx].player for the "to act" seat.
    decisions.append(_ShimDecision(player=current_player, action_taken=-1))

    tokens, attention_mask = tokenize_decision(
        game_hands, decl_id, decisions, decision_idx,
    )

    # voids feature needs prior plays (history only, no sentinel).
    prior_plays = reconstruct_prior_plays(game_hands, decisions, decision_idx)
    voids = voids_feature_vector(prior_plays, decl_id, current_player)  # [24]

    return {
        "tokens": tokens,                         # [33, 5] long
        "attention_mask": attention_mask,         # [33] bool
        "voids": voids,                           # [24] float
        "current_player": current_player,
        "decision_idx": decision_idx,
        "game_hands": game_hands,
        "prior_plays": prior_plays,
        "decl_id": decl_id,
    }
