"""World reconstruction helpers for E[Q] generation.

These helpers translate sampled *remaining* hands into *hypothetical initial deals*,
and provide common mapping utilities used by both single-game and batched generation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from forge.eq.game import GameState


def _build_hypothetical_worlds_batched(
    game: GameState,
    remaining_worlds: list[list[list[int]]],
    played_by: dict[int, list[int]],
    current_player: int,
) -> tuple[list[list[list[int]]], dict]:
    """Reconstruct hypothetical initial hands for each sampled world (batched).

    For proper E[Q] marginalization, we query the oracle on HYPOTHETICAL worlds.
    Each sampled world gives remaining hands for opponents. We reconstruct initial
    hands as: initial[p] = remaining[p] + played_by[p].

    trick_plays uses domino_id (public info) instead of local_idx so that trick
    tokens are world-invariant and can be batched efficiently.
    """
    n_worlds = len(remaining_worlds)
    hypothetical_deals: list[list[list[int]]] = []

    remaining_bitmask = np.zeros((n_worlds, 4), dtype=np.int32)

    for world_idx, remaining_hands in enumerate(remaining_worlds):
        initial_hands: list[list[int]] = []
        for p in range(4):
            initial = list(remaining_hands[p]) + list(played_by[p])
            initial.sort()
            initial_hands.append(initial)

        hypothetical_deals.append(initial_hands)

        for p in range(4):
            for local_idx, domino in enumerate(initial_hands[p]):
                if domino not in game.played:
                    remaining_bitmask[world_idx, p] |= 1 << local_idx

    trick_plays = [(play_player, domino_id) for play_player, domino_id in game.current_trick]

    game_state_info = {
        "decl_id": game.decl_id,
        "leader": game.leader,
        "trick_plays": trick_plays,  # Uses domino_id, not local_idx
        "remaining": remaining_bitmask,  # (N, 4) for all worlds
    }

    return hypothetical_deals, game_state_info


def _build_legal_mask(legal_actions: tuple[int, ...], hand: list[int]) -> Tensor:
    """Build a boolean mask over remaining-hand indices for legal actions."""
    mask = torch.zeros(len(hand), dtype=torch.bool)
    if not legal_actions:
        return mask

    domino_to_index = {d: i for i, d in enumerate(hand)}
    for domino_id in legal_actions:
        idx = domino_to_index.get(domino_id)
        if idx is not None:
            mask[idx] = True
    return mask

