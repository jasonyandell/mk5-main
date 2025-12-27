"""
SeedContext - Seed-specific precomputed tables for GPU solver.

This module builds tables that depend on the specific deal (L table) and
declaration (absorption_id, power_id). These tables enable O(1) lookups
for follow-suit and trick resolution.

Key insight from SUIT_ALGEBRA.md: All pip trumps are isomorphic via the S7
symmetry group. The context tables capture this for a specific deal.
"""

from dataclasses import dataclass
import torch
import numpy as np
from numpy.typing import NDArray

from tables import EFFECTIVE_SUIT, SUIT_MASK, HAS_POWER, RANK, DOMINO_PIPS
from rng import deal_with_seed


@dataclass
class SeedContext:
    """
    Precomputed tables for a specific seed and declaration.

    All tensors are on CPU initially (will be moved to GPU later).
    """
    L: torch.Tensor               # (4, 7) int8 - global domino ID per (player, local_idx)
    LOCAL_FOLLOW: torch.Tensor    # (112,) int8 - flattened follow masks
    TRICK_WINNER: torch.Tensor    # (9604,) int8 - winner offset (0-3)
    TRICK_POINTS: torch.Tensor    # (9604,) int8 - points (1-31, NOT 1-11!)
    absorption_id: int
    power_id: int


def _get_domino_points(global_id: int) -> int:
    """
    Get point value of a domino for scoring.

    Matches TypeScript getDominoPoints:
    - 5-5 = 10 points
    - 6-4 = 10 points
    - pip_sum == 5 = 5 points (5-0, 4-1, 3-2)
    - otherwise = 0 points
    """
    lo, hi = DOMINO_PIPS[global_id]
    pip_sum = lo + hi

    # 5-5 double
    if lo == 5 and hi == 5:
        return 10
    # 6-4
    if lo == 4 and hi == 6:
        return 10
    # Sum equals 5: 5-0 (id=5), 4-1 (id=8), 3-2 (id=14)
    if pip_sum == 5:
        return 5
    return 0


def _rank_in_trick(
    global_id: int,
    led_suit: int,
    absorption_id: int,
    power_id: int
) -> int:
    """
    Compute three-tier rank for a domino in a trick.

    Implements τ(d, ℓ, δ) = (tier << 4) + rank from SUIT_ALGEBRA.md §8.

    Three-tier ranking:
    - Tier 2 (trump/power): 32-46
    - Tier 1 (follows suit): 16-30
    - Tier 0 (slough): 0

    Returns: (tier << 4) + rank
    """
    lo, hi = DOMINO_PIPS[global_id]
    is_double = (lo == hi)
    pip_sum = lo + hi

    # Determine tier
    has_power = bool(HAS_POWER[global_id, power_id])

    # Can this domino follow the led suit?
    # Check if global_id bit is set in SUIT_MASK[absorption_id][led_suit]
    suit_mask = SUIT_MASK[absorption_id, led_suit]
    follows_suit = bool((suit_mask >> global_id) & 1)

    if has_power:
        tier = 2
    elif follows_suit:
        tier = 1
    else:
        tier = 0

    # Tier 0 (slough): all return 0
    if tier == 0:
        return 0

    # Determine rank within tier
    # Per SUIT_ALGEBRA.md §8: when absorption_id = 7 (doubles trump), doubles rank by pip value
    if absorption_id == 7 and is_double:
        rank = hi  # Pip value (0-6)
    elif is_double:
        rank = 14  # Highest in suit
    else:
        rank = pip_sum  # 0-12

    return (tier << 4) + rank


def _build_local_follow(
    L: NDArray[np.int32],
    absorption_id: int
) -> torch.Tensor:
    """
    Build LOCAL_FOLLOW table: (112,) int8 flattened.

    For each (leader, lead_local_idx, follower_offset):
    - Compute which local indices of the follower can follow the led suit
    - Return as 7-bit mask

    Indexing: leader * 28 + lead_local_idx * 4 + follower_offset
    (But follower_offset 0 is meaningless since leader can't follow themselves)

    Total: 4 leaders * 7 lead positions * 4 offsets = 112 entries
    """
    result = np.zeros(112, dtype=np.int8)

    for leader in range(4):
        for lead_local_idx in range(7):
            # Get the global ID of the led domino
            global_id = L[leader, lead_local_idx]

            # Get the led suit for this domino
            led_suit = EFFECTIVE_SUIT[global_id, absorption_id]

            for follower_offset in range(4):
                # Compute follower's player index
                follower = (leader + follower_offset) % 4

                # Build 7-bit mask for which of follower's dominoes can follow
                mask = 0
                for local_idx in range(7):
                    follower_global_id = L[follower, local_idx]

                    # Check if this domino can follow the led suit
                    suit_mask = SUIT_MASK[absorption_id, led_suit]
                    can_follow = bool((suit_mask >> follower_global_id) & 1)

                    if can_follow:
                        mask |= (1 << local_idx)

                # Store in flattened array
                idx = leader * 28 + lead_local_idx * 4 + follower_offset
                result[idx] = mask

    return torch.tensor(result, dtype=torch.int8)


def _build_trick_tables(
    L: NDArray[np.int32],
    absorption_id: int,
    power_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build TRICK_WINNER and TRICK_POINTS tables.

    For each possible 4-domino trick combination:
    - TRICK_WINNER: who wins (0-3, offset from leader)
    - TRICK_POINTS: total points (1-31)

    Indexing: leader * 2401 + p0 * 343 + p1 * 49 + p2 * 7 + p3
    where p0-p3 are local indices for leader, +1, +2, +3 respectively.

    Total: 4 * 7^4 = 9604 entries
    """
    winner = np.zeros(9604, dtype=np.int8)
    points = np.zeros(9604, dtype=np.int8)

    for leader in range(4):
        for p0 in range(7):
            # Leader's play determines the led suit
            g0 = L[leader, p0]
            led_suit = EFFECTIVE_SUIT[g0, absorption_id]

            # Leader's rank
            rank0 = _rank_in_trick(g0, led_suit, absorption_id, power_id)
            pts0 = _get_domino_points(g0)

            for p1 in range(7):
                g1 = L[(leader + 1) % 4, p1]
                rank1 = _rank_in_trick(g1, led_suit, absorption_id, power_id)
                pts1 = _get_domino_points(g1)

                for p2 in range(7):
                    g2 = L[(leader + 2) % 4, p2]
                    rank2 = _rank_in_trick(g2, led_suit, absorption_id, power_id)
                    pts2 = _get_domino_points(g2)

                    for p3 in range(7):
                        g3 = L[(leader + 3) % 4, p3]
                        rank3 = _rank_in_trick(g3, led_suit, absorption_id, power_id)
                        pts3 = _get_domino_points(g3)

                        # Find winner (highest rank wins, ties go to earlier player)
                        ranks = [rank0, rank1, rank2, rank3]
                        max_rank = max(ranks)
                        winner_offset = ranks.index(max_rank)

                        # Total points = domino points + 1 for trick
                        total_points = pts0 + pts1 + pts2 + pts3 + 1

                        # Store
                        idx = leader * 2401 + p0 * 343 + p1 * 49 + p2 * 7 + p3
                        winner[idx] = winner_offset
                        points[idx] = total_points

    return torch.tensor(winner, dtype=torch.int8), torch.tensor(points, dtype=torch.int8)


def build_context(seed: int, decl_id: int) -> SeedContext:
    """
    Build a SeedContext for a specific seed and declaration.

    Args:
        seed: RNG seed for dealing
        decl_id: Declaration ID (0-6 for pip trump in MVP)

    Returns:
        SeedContext with all precomputed tables

    For decl_id 0-6 (pip trump):
    - absorption_id = decl_id (dominoes containing that pip -> suit 7)
    - power_id = decl_id (same dominoes have trump power)
    """
    # Get the deal
    L = deal_with_seed(seed)  # (4, 7) int32

    # For MVP, only pip trump (decl_id 0-6)
    if decl_id < 0 or decl_id > 6:
        raise ValueError(f"decl_id must be 0-6 for pip trump, got {decl_id}")

    absorption_id = decl_id
    power_id = decl_id

    # Build tables
    local_follow = _build_local_follow(L, absorption_id)
    trick_winner, trick_points = _build_trick_tables(L, absorption_id, power_id)

    return SeedContext(
        L=torch.tensor(L, dtype=torch.int8),
        LOCAL_FOLLOW=local_follow,
        TRICK_WINNER=trick_winner,
        TRICK_POINTS=trick_points,
        absorption_id=absorption_id,
        power_id=power_id,
    )
