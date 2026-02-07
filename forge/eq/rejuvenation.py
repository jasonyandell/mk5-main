"""Particle rejuvenation kernel for posterior weighting (Phase 8)."""

from __future__ import annotations

import numpy as np
from torch import Tensor

from forge.eq.oracle import Stage1Oracle
from forge.eq.types import PosteriorConfig
from forge.oracle.tables import can_follow, led_suit_for_lead_domino


def _rejuvenate_particles(
    hypothetical_deals: list[list[list[int]]],
    weights: Tensor,
    play_history: list[tuple[int, int, int]],
    decl_id: int,
    oracle: Stage1Oracle,
    config: PosteriorConfig,
    window_k: int,
    rng: np.random.Generator,
) -> tuple[list[list[list[int]]], int]:
    """Resample and rejuvenate particles using MCMC swap kernel."""
    n_worlds = len(hypothetical_deals)

    weights_np = weights.cpu().numpy()
    indices = rng.choice(n_worlds, size=n_worlds, replace=True, p=weights_np)

    new_deals = [[list(hand) for hand in hypothetical_deals[idx]] for idx in indices]

    played_set = {d for _, d, _ in play_history}

    voids: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    for player, domino, lead_domino in play_history:
        led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
        if not can_follow(domino, led_suit, decl_id):
            voids[player].add(led_suit)

    if play_history:
        conditioned_player = play_history[-1][0]
    else:
        conditioned_player = 0

    n_accepts = 0

    for particle_idx in range(n_worlds):
        particle = new_deals[particle_idx]

        for _ in range(config.rejuvenation_steps):
            latent_players = [p for p in range(4) if p != conditioned_player]
            if len(latent_players) < 2:
                continue

            u, v = (int(x) for x in rng.choice(latent_players, size=2, replace=False))

            u_unplayed = [d for d in particle[u] if d not in played_set]
            v_unplayed = [d for d in particle[v] if d not in played_set]

            if not u_unplayed or not v_unplayed:
                continue

            a = int(rng.choice(u_unplayed))
            b = int(rng.choice(v_unplayed))

            if a == b:
                continue

            u_void_violation = _domino_violates_voids(
                b, voids[u], decl_id, play_history, u
            )
            v_void_violation = _domino_violates_voids(
                a, voids[v], decl_id, play_history, v
            )

            if u_void_violation or v_void_violation:
                continue

            particle[u].remove(a)
            particle[u].append(b)
            particle[u].sort()

            particle[v].remove(b)
            particle[v].append(a)
            particle[v].sort()

            n_accepts += 1

    return new_deals, n_accepts


def _domino_violates_voids(
    domino: int,
    player_voids: set[int],
    decl_id: int,
    play_history: list[tuple[int, int, int]],
    player: int,
) -> bool:
    """Check if giving this domino to a player violates their void constraints."""
    if not player_voids:
        return False

    for void_suit in player_voids:
        if can_follow(domino, void_suit, decl_id):
            return True

    return False

