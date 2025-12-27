"""
Texas 42 Solver - Per-Seed Context Builder

Builds precomputed tables for a specific seed + declaration combination.
This is the key optimization: precompute all trick outcomes before solving.

Tables built per-seed:
- L[4][7]: local index -> global domino ID
- FOLLOW_LOCAL[4][8]: player x led_suit -> 7-bit local follow mask
- TRICK_WINNER[4][7][7][7][7]: leader x i0 x i1 x i2 x i3 -> winner player ID
- TRICK_POINTS[4][7][7][7][7]: leader x i0 x i1 x i2 x i3 -> points in trick
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

from .tables import (
    DOMINO_PIPS, EFFECTIVE_SUIT, SUIT_MASK, RANK, POINTS,
    domino_to_id, get_absorption_id, get_power_id, CALLED_SUIT
)
from .deal import deal_dominoes_with_seed


@dataclass
class SeedContext:
    """
    Precomputed context for solving a specific seed + declaration.

    All game rule logic is compiled into these tables before solving.
    The DP hot path only does table lookups.
    """
    seed: int
    decl_id: int
    absorption_id: int
    power_id: int

    # L[player][local_idx] -> global domino ID (0-27)
    L: np.ndarray  # shape (4, 7), dtype int8

    # FOLLOW_LOCAL[player][led_suit] -> 7-bit local follow mask
    FOLLOW_LOCAL: np.ndarray  # shape (4, 8), dtype uint8

    # TRICK_WINNER[leader][i0][i1][i2][i3] -> winner player ID (0-3)
    TRICK_WINNER: np.ndarray  # shape (4, 7, 7, 7, 7), dtype int8

    # TRICK_POINTS[leader][i0][i1][i2][i3] -> points in trick (1-11)
    TRICK_POINTS: np.ndarray  # shape (4, 7, 7, 7, 7), dtype int8

    # Initial hand masks for each player (28-bit each)
    initial_hands: Tuple[int, int, int, int]


def build_context(seed: int, decl_id: int, verbose: bool = False) -> SeedContext:
    """
    Build the per-seed precomputed context.

    This precomputes all 4 * 7^4 = 9,604 possible trick outcomes.
    """
    absorption_id = get_absorption_id(decl_id)
    power_id = get_power_id(decl_id)

    if verbose:
        print(f"Building context for seed={seed}, decl={decl_id}")
        print(f"  absorption_id={absorption_id}, power_id={power_id}")

    # Deal the dominoes
    hands = deal_dominoes_with_seed(seed)

    # Build L table: local index -> global domino ID
    L = np.zeros((4, 7), dtype=np.int8)
    for player in range(4):
        for local_idx, (high, low) in enumerate(hands[player]):
            L[player, local_idx] = domino_to_id(high, low)

    if verbose:
        print(f"  L table built: {L.shape}")
        for p in range(4):
            dominos = [f"{DOMINO_PIPS[L[p, i]][1]}-{DOMINO_PIPS[L[p, i]][0]}" for i in range(7)]
            print(f"    Player {p}: {dominos}")

    # Build FOLLOW_LOCAL table: which local indices can follow each suit
    FOLLOW_LOCAL = np.zeros((4, 8), dtype=np.uint8)
    for player in range(4):
        for suit in range(8):
            mask = 0
            for local_idx in range(7):
                global_id = int(L[player, local_idx])
                # Check if this domino can follow the suit
                suit_mask = int(SUIT_MASK[absorption_id, suit])
                if (suit_mask >> global_id) & 1:
                    mask |= (1 << local_idx)
            FOLLOW_LOCAL[player, suit] = mask

    if verbose:
        print(f"  FOLLOW_LOCAL table built: {FOLLOW_LOCAL.shape}")

    # Build TRICK_WINNER and TRICK_POINTS tables
    TRICK_WINNER = np.zeros((4, 7, 7, 7, 7), dtype=np.int8)
    TRICK_POINTS = np.zeros((4, 7, 7, 7, 7), dtype=np.int8)

    trick_count = 0
    for leader in range(4):
        for i0 in range(7):
            for i1 in range(7):
                for i2 in range(7):
                    for i3 in range(7):
                        # Get global domino IDs in play order
                        players = [(leader + k) % 4 for k in range(4)]
                        local_indices = [i0, i1, i2, i3]
                        global_ids = [int(L[players[k], local_indices[k]]) for k in range(4)]

                        # Compute led suit from first domino
                        led_suit = int(EFFECTIVE_SUIT[global_ids[0], absorption_id])

                        # Compute tau for each domino
                        taus = []
                        for k in range(4):
                            d = global_ids[k]
                            # Check if this domino has power
                            has_power = bool(int(RANK[d, power_id]) >= 32)

                            # Check if this domino follows the led suit
                            suit_mask = int(SUIT_MASK[absorption_id, led_suit])
                            follows_suit = (suit_mask >> d) & 1

                            # Compute tau: tier << 4 + rank
                            if has_power:
                                tau = int(RANK[d, power_id])  # Already encoded as tier 2
                            elif follows_suit:
                                # Tier 1: (1 << 4) + rank
                                raw_rank = int(RANK[d, power_id]) & 0x0F  # Extract rank bits
                                tau = (1 << 4) + raw_rank
                            else:
                                # Tier 0: slough
                                tau = 0

                            taus.append(tau)

                        # Find winner: highest tau (position in play order)
                        winner_pos = max(range(4), key=lambda k: taus[k])
                        winner_player = players[winner_pos]

                        # Compute points: sum of POINTS + 1 for winning the trick
                        points = sum(int(POINTS[d]) for d in global_ids) + 1

                        TRICK_WINNER[leader, i0, i1, i2, i3] = winner_player
                        TRICK_POINTS[leader, i0, i1, i2, i3] = points
                        trick_count += 1

    if verbose:
        print(f"  TRICK_WINNER/TRICK_POINTS tables built: {TRICK_WINNER.shape}")
        print(f"  Total trick outcomes precomputed: {trick_count:,}")

    # Convert hands to bitmasks
    initial_hands = tuple(
        sum(1 << domino_to_id(h, l) for h, l in hand)
        for hand in hands
    )

    return SeedContext(
        seed=seed,
        decl_id=decl_id,
        absorption_id=absorption_id,
        power_id=power_id,
        L=L,
        FOLLOW_LOCAL=FOLLOW_LOCAL,
        TRICK_WINNER=TRICK_WINNER,
        TRICK_POINTS=TRICK_POINTS,
        initial_hands=initial_hands
    )


def get_local_legal_mask(remaining: int, led_local_idx: int | None, player: int, ctx: SeedContext) -> int:
    """
    Get legal plays as a 7-bit local mask.

    Args:
        remaining: 7-bit mask of remaining local indices for this player
        led_local_idx: The local index that was led (None if leading)
        player: Current player (0-3)
        ctx: Seed context

    Returns: 7-bit mask of legal local indices
    """
    if led_local_idx is None:
        return remaining  # Leading: any remaining domino

    # Get the led suit from the lead domino
    leader_player = (player - 1) % 4  # Who led? (We need to know whose local index it is)
    # Actually, we need to track who led the trick, not just previous player
    # For now, assume led_local_idx is from the leader's perspective
    # This will be handled properly in the state module

    led_global = ctx.L[leader_player, led_local_idx]
    led_suit = int(EFFECTIVE_SUIT[led_global, ctx.absorption_id])

    # Get which of our local indices can follow
    can_follow = remaining & ctx.FOLLOW_LOCAL[player, led_suit]
    return can_follow if can_follow != 0 else remaining


def print_context_summary(ctx: SeedContext):
    """Print a summary of the context for debugging."""
    print(f"=== Context for seed={ctx.seed}, decl={ctx.decl_id} ===")
    print(f"  absorption_id={ctx.absorption_id}, power_id={ctx.power_id}")
    print()
    print("L table (local -> global domino ID):")
    for p in range(4):
        dominoes = [f"{DOMINO_PIPS[ctx.L[p, i]][1]}-{DOMINO_PIPS[ctx.L[p, i]][0]}" for i in range(7)]
        print(f"  Player {p}: {dominoes}")
    print()
    print("FOLLOW_LOCAL table (which local indices follow each suit):")
    for p in range(4):
        suits = [f"s{s}:{bin(ctx.FOLLOW_LOCAL[p, s])}" for s in range(8)]
        print(f"  Player {p}: {' '.join(suits)}")
    print()
    print(f"TRICK_WINNER/TRICK_POINTS: {ctx.TRICK_WINNER.shape}")
    print(f"  Memory: {ctx.TRICK_WINNER.nbytes + ctx.TRICK_POINTS.nbytes:,} bytes")

    # Sample a few trick outcomes
    print("  Sample trick outcomes:")
    for leader in [0, 1]:
        for i0, i1, i2, i3 in [(0, 0, 0, 0), (3, 3, 3, 3), (6, 5, 4, 3)]:
            winner = ctx.TRICK_WINNER[leader, i0, i1, i2, i3]
            points = ctx.TRICK_POINTS[leader, i0, i1, i2, i3]
            print(f"    leader={leader}, ({i0},{i1},{i2},{i3}): winner={winner}, points={points}")


if __name__ == "__main__":
    # Test context building
    ctx = build_context(12345, 3, verbose=True)
    print()
    print_context_summary(ctx)
