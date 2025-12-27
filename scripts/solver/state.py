"""
Texas 42 Solver - State Representation

Defines the game state for backward induction and provides:
- State dataclass with all game information
- Pack/unpack functions for efficient storage
- State transitions (apply_move)
- Legal move generation
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .tables import EFFECTIVE_SUIT
from .context import SeedContext


@dataclass(frozen=True)
class State:
    """
    Game state for the DP solver.

    Immutable dataclass representing a position in the game tree.
    All fields use small integers for efficient packing.
    """
    # 4 x 7-bit masks: which local indices each player still has
    remaining: Tuple[int, int, int, int]

    # Team 0's points so far (0-42)
    team0_points: int

    # Who leads this trick (0-3)
    leader: int

    # Whose turn is it (0-3)
    current_player: int

    # How many plays in current trick (0-3)
    trick_len: int

    # Local indices played so far in this trick (in play order)
    # Only first trick_len entries are valid
    trick: Tuple[int, int, int, int]

    def total_remaining(self) -> int:
        """Total dominoes remaining in all hands."""
        return sum(bin(r).count('1') for r in self.remaining)

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (all dominoes played)."""
        return all(r == 0 for r in self.remaining)

    def value(self) -> int:
        """Terminal value: team0_points - team1_points = 2*team0_points - 42."""
        return 2 * self.team0_points - 42


def pack_state(s: State) -> int:
    """
    Pack state into a 64-bit integer for efficient storage.

    Layout (52 bits total):
    - bits 0-6:   remaining[0] (7 bits)
    - bits 7-13:  remaining[1] (7 bits)
    - bits 14-20: remaining[2] (7 bits)
    - bits 21-27: remaining[3] (7 bits)
    - bits 28-33: team0_points (6 bits, 0-42)
    - bits 34-35: leader (2 bits, 0-3)
    - bits 36-37: current_player (2 bits, 0-3)
    - bits 38-39: trick_len (2 bits, 0-3)
    - bits 40-42: trick[0] (3 bits, 0-6)
    - bits 43-45: trick[1] (3 bits, 0-6)
    - bits 46-48: trick[2] (3 bits, 0-6)
    - bits 49-51: trick[3] (3 bits, 0-6)
    """
    return (
        s.remaining[0] |
        (s.remaining[1] << 7) |
        (s.remaining[2] << 14) |
        (s.remaining[3] << 21) |
        (s.team0_points << 28) |
        (s.leader << 34) |
        (s.current_player << 36) |
        (s.trick_len << 38) |
        (s.trick[0] << 40) |
        (s.trick[1] << 43) |
        (s.trick[2] << 46) |
        (s.trick[3] << 49)
    )


def unpack_state(packed: int) -> State:
    """Unpack a 64-bit integer back into a State."""
    return State(
        remaining=(
            packed & 0x7F,
            (packed >> 7) & 0x7F,
            (packed >> 14) & 0x7F,
            (packed >> 21) & 0x7F,
        ),
        team0_points=(packed >> 28) & 0x3F,
        leader=(packed >> 34) & 0x3,
        current_player=(packed >> 36) & 0x3,
        trick_len=(packed >> 38) & 0x3,
        trick=(
            (packed >> 40) & 0x7,
            (packed >> 43) & 0x7,
            (packed >> 46) & 0x7,
            (packed >> 49) & 0x7,
        )
    )


def get_legal_mask(state: State, ctx: SeedContext) -> int:
    """
    Get legal moves as a 7-bit local mask.

    Returns: 7-bit mask where bit i is set if local index i is legal to play.
    """
    player = state.current_player
    remaining = state.remaining[player]

    if state.trick_len == 0:
        # Leading: any remaining domino is legal
        return remaining

    # Following: must follow suit if possible
    # Get the led domino's global ID
    lead_local = state.trick[0]
    leader = state.leader
    lead_global = int(ctx.L[leader, lead_local])
    led_suit = int(EFFECTIVE_SUIT[lead_global, ctx.absorption_id])

    # Which of our remaining dominoes can follow?
    can_follow = remaining & int(ctx.FOLLOW_LOCAL[player, led_suit])
    return can_follow if can_follow != 0 else remaining


def apply_move(state: State, local_idx: int, ctx: SeedContext) -> State:
    """
    Apply a move and return the new state.

    Args:
        state: Current state
        local_idx: Local index of domino to play (0-6)
        ctx: Seed context

    Returns: New state after the move
    """
    player = state.current_player

    # Remove the domino from player's hand
    new_remaining = list(state.remaining)
    new_remaining[player] = new_remaining[player] & ~(1 << local_idx)
    new_remaining = tuple(new_remaining)

    # Add to current trick
    new_trick = list(state.trick)
    new_trick[state.trick_len] = local_idx
    new_trick = tuple(new_trick)

    if state.trick_len + 1 < 4:
        # Trick not complete: advance to next player
        next_player = (player + 1) % 4
        return State(
            remaining=new_remaining,
            team0_points=state.team0_points,
            leader=state.leader,
            current_player=next_player,
            trick_len=state.trick_len + 1,
            trick=new_trick
        )
    else:
        # Trick complete: resolve winner and points
        leader = state.leader
        i0, i1, i2, i3 = new_trick

        winner = int(ctx.TRICK_WINNER[leader, i0, i1, i2, i3])
        points = int(ctx.TRICK_POINTS[leader, i0, i1, i2, i3])

        # Add points to team 0 if winner is on team 0
        new_team0_points = state.team0_points
        if winner % 2 == 0:
            new_team0_points += points

        return State(
            remaining=new_remaining,
            team0_points=new_team0_points,
            leader=winner,
            current_player=winner,
            trick_len=0,
            trick=(0, 0, 0, 0)
        )


def initial_state(ctx: SeedContext, first_leader: int = 0) -> State:
    """Create the initial state from a seed context."""
    return State(
        remaining=(0x7F, 0x7F, 0x7F, 0x7F),  # All 7 local indices for each player
        team0_points=0,
        leader=first_leader,
        current_player=first_leader,
        trick_len=0,
        trick=(0, 0, 0, 0)
    )


def format_state(state: State, ctx: SeedContext) -> str:
    """Format a state for debugging."""
    from .tables import DOMINO_PIPS

    lines = []
    lines.append(f"Team0: {state.team0_points} pts, Leader: P{state.leader}, Current: P{state.current_player}")
    lines.append(f"Trick ({state.trick_len}/4): {state.trick[:state.trick_len]}")

    for p in range(4):
        hand = []
        for i in range(7):
            if (state.remaining[p] >> i) & 1:
                gid = int(ctx.L[p, i])
                lo, hi = DOMINO_PIPS[gid]
                hand.append(f"{hi}-{lo}")
        lines.append(f"  P{p}: [{', '.join(hand)}]")

    return "\n".join(lines)


if __name__ == "__main__":
    from .context import build_context

    # Test state packing/unpacking
    print("=== State Pack/Unpack Test ===")
    ctx = build_context(12345, 3)

    s0 = initial_state(ctx, first_leader=0)
    print(f"Initial state:\n{format_state(s0, ctx)}")
    print(f"Packed: {hex(pack_state(s0))}")

    # Verify round-trip
    s0_unpacked = unpack_state(pack_state(s0))
    assert s0 == s0_unpacked, "Pack/unpack round-trip failed!"
    print("Pack/unpack round-trip: OK")

    # Test legal moves
    print(f"\nLegal moves at start: {bin(get_legal_mask(s0, ctx))} (all 7)")

    # Make a move
    legal = get_legal_mask(s0, ctx)
    first_move = (legal & -legal).bit_length() - 1  # Lowest set bit
    s1 = apply_move(s0, first_move, ctx)
    print(f"\nAfter P0 plays local {first_move}:")
    print(format_state(s1, ctx))
    print(f"Legal for P1: {bin(get_legal_mask(s1, ctx))}")

    # Play out a full trick
    s = s1
    for _ in range(3):
        legal = get_legal_mask(s, ctx)
        move = (legal & -legal).bit_length() - 1
        s = apply_move(s, move, ctx)

    print(f"\nAfter full trick:")
    print(format_state(s, ctx))
    print(f"Trick winner led next trick")
