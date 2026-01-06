"""Symmetry utilities for state space compression.

This module provides functions to exploit game symmetries:
- team_swap(): Swap teams 0 ↔ 1 (negates V)
- seat_rotate(): Rotate seats by 2 within teams (preserves V)
- canonical_form(): Compute canonical representative of orbit
- enumerate_orbits(): Group states by orbit
- orbit_compression_ratio(): Measure compression from symmetry

**Symmetries in Texas 42:**

1. Team reflection (Z2): Swap teams 0↔1
   - Players: 0↔1, 2↔3
   - Effect: V → -V (opponents become allies)

2. Seat rotation (Z2): Rotate by 2 seats within teams
   - Players: 0↔2, 1↔3
   - Effect: V → V (just relabeling partners)

Combined: Klein four-group (Z2 × Z2), 4 elements
Maximum compression: 4x (if all orbits full size)
"""

from __future__ import annotations

import numpy as np

from forge.oracle import schema


# =============================================================================
# State Bit Constants (from schema.py)
# =============================================================================

REMAINING_BITS = 7
REMAINING_MASK = 0x7F
LEADER_SHIFT = 28
LEADER_MASK = 0x3
TRICK_LEN_SHIFT = 30
TRICK_LEN_MASK = 0x3
P0_SHIFT = 32
P1_SHIFT = 35
P2_SHIFT = 38
PLAY_MASK = 0x7


# =============================================================================
# Single State Packing/Unpacking
# =============================================================================

def unpack_state_single(state: int) -> tuple[np.ndarray, int, int, int, int, int]:
    """
    Unpack a single int64 state.

    Returns:
        (remaining, leader, trick_len, p0, p1, p2)
    """
    remaining = np.zeros(4, dtype=np.uint8)
    for p in range(4):
        remaining[p] = (state >> (p * REMAINING_BITS)) & REMAINING_MASK
    leader = (state >> LEADER_SHIFT) & LEADER_MASK
    trick_len = (state >> TRICK_LEN_SHIFT) & TRICK_LEN_MASK
    p0 = (state >> P0_SHIFT) & PLAY_MASK
    p1 = (state >> P1_SHIFT) & PLAY_MASK
    p2 = (state >> P2_SHIFT) & PLAY_MASK
    return remaining, leader, trick_len, p0, p1, p2


def pack_state_single(
    remaining: np.ndarray,
    leader: int,
    trick_len: int,
    p0: int,
    p1: int,
    p2: int,
) -> int:
    """Pack state components into int64."""
    state = 0
    for p in range(4):
        state |= int(remaining[p]) << (p * REMAINING_BITS)
    state |= (leader & LEADER_MASK) << LEADER_SHIFT
    state |= (trick_len & TRICK_LEN_MASK) << TRICK_LEN_SHIFT
    state |= (p0 & PLAY_MASK) << P0_SHIFT
    state |= (p1 & PLAY_MASK) << P1_SHIFT
    state |= (p2 & PLAY_MASK) << P2_SHIFT
    return state


# =============================================================================
# Team Swap Symmetry (Z2)
# =============================================================================

def team_swap_single(state: int) -> int:
    """
    Apply team swap symmetry to a single state.

    Swaps players: 0↔1, 2↔3
    Effect on V: V → -V

    Args:
        state: Packed int64 state

    Returns:
        Transformed packed state
    """
    remaining, leader, trick_len, p0, p1, p2 = unpack_state_single(state)

    # Swap hands: 0↔1, 2↔3
    new_remaining = np.array([
        remaining[1],  # new player 0 gets old player 1's hand
        remaining[0],  # new player 1 gets old player 0's hand
        remaining[3],  # new player 2 gets old player 3's hand
        remaining[2],  # new player 3 gets old player 2's hand
    ], dtype=np.uint8)

    # Transform leader: toggle bit 0 (0↔1, 2↔3)
    new_leader = leader ^ 1

    # Trick plays are positional relative to leader
    # After swap, the plays at each position move to swapped players
    # But since we also swapped hands, the local indices stay the same
    # The plays rotate through: position 0,1,2,3 map to new positions
    # Old position i was played by (leader + i) % 4
    # New position i is played by (new_leader + i) % 4 = ((leader^1) + i) % 4
    #
    # Actually the trick structure is complex. Let's think about it:
    # - p0 is the play at position 0 (by the leader)
    # - After team swap, the new leader is leader^1
    # - The same physical play happened, just by swapped players
    # - Since hands also swapped, the local index meaning changes
    #
    # For simplicity in mid-trick states, we'd need the full trick to reorder.
    # For now, only apply to trick_len=0 states (between tricks).

    if trick_len == 0:
        return pack_state_single(new_remaining, new_leader, 0, 7, 7, 7)
    else:
        # For mid-trick states, more complex transformation needed
        # The plays p0,p1,p2 are at positions relative to leader
        # We need to reorder them for the new leader
        #
        # Old: plays at positions 0,1,2 (relative to old leader)
        # New: need plays at positions 0,1,2 (relative to new leader)
        # new_leader = leader ^ 1
        #
        # Old player at position i: (leader + i) % 4
        # This player maps to new player: old_player ^ (1 if old_player is even else 1)
        # Actually: 0→1, 1→0, 2→3, 3→2, so new_player = old_player ^ 1
        #
        # New position of this player: (new_player - new_leader) % 4
        #   = ((old_player ^ 1) - (leader ^ 1)) % 4
        #   = (old_player ^ 1 - leader ^ 1) % 4
        #
        # Let's compute: if leader=0, new_leader=1
        # Old pos 0 (player 0) → new player 1, new pos = (1-1)%4 = 0
        # Old pos 1 (player 1) → new player 0, new pos = (0-1)%4 = 3
        # Old pos 2 (player 2) → new player 3, new pos = (3-1)%4 = 2
        # Old pos 3 (player 3) → new player 2, new pos = (2-1)%4 = 1
        #
        # So plays[0,1,2,3] → plays[0,3,2,1] (in new positions)
        # Or: new_plays[0] = old_plays[0], new_plays[1] = old_plays[3]...
        # But we only have p0,p1,p2 (not p3)
        #
        # new_plays[new_pos] = old_plays[old_pos] for each play
        # new_pos 0 = old_pos 0 → new_p0 = old_p0
        # new_pos 1 = old_pos 3 → new_p1 = old_p3 (not stored!)
        # new_pos 2 = old_pos 2 → new_p2 = old_p2
        # new_pos 3 = old_pos 1 → (would be p3, not stored)
        #
        # This means for mid-trick, we can't fully transform without p3.
        # For trick_len=1: only p0 exists, it stays at position 0
        # For trick_len=2: p0→p0, p1→? need p3 which doesn't exist
        # For trick_len=3: same issue
        #
        # Actually wait - we need to think about this differently.
        # If trick_len < 4, not all positions have plays yet.
        #
        # For trick_len=1: only p0 (by leader) has been played
        #   After swap, new_p0 should be that same local index
        #   But the hand swapped! Old leader's local index i referred to
        #   hands[leader][i]. After swap, new player at position 0 is new_leader.
        #   new_leader's hand = old hands[leader^1], not hands[leader]
        #   So the local index doesn't directly transfer.
        #
        # This is getting complicated. Let me simplify:
        # For orbit enumeration, we can focus on trick_len=0 states only.

        # Return original state for mid-trick (not transformable simply)
        return state


def team_swap(states: np.ndarray) -> np.ndarray:
    """
    Apply team swap symmetry to multiple states (vectorized).

    Only transforms trick_len=0 states correctly.
    Mid-trick states are returned unchanged.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) transformed packed states
    """
    remaining, leader, trick_len, p0, p1, p2 = schema.unpack_state(states)

    # Swap hands: 0↔1, 2↔3
    new_remaining = np.zeros_like(remaining)
    new_remaining[:, 0] = remaining[:, 1]
    new_remaining[:, 1] = remaining[:, 0]
    new_remaining[:, 2] = remaining[:, 3]
    new_remaining[:, 3] = remaining[:, 2]

    # Transform leader
    new_leader = leader ^ 1

    # Only transform trick_len=0 states
    mask = trick_len == 0

    # Pack results
    result = np.zeros_like(states)

    # For trick_len=0: full transformation
    for i in range(4):
        result += new_remaining[:, i].astype(np.int64) << (i * REMAINING_BITS)
    result += (new_leader.astype(np.int64) & LEADER_MASK) << LEADER_SHIFT
    result += (trick_len.astype(np.int64) & TRICK_LEN_MASK) << TRICK_LEN_SHIFT
    result += (np.int64(7) & PLAY_MASK) << P0_SHIFT  # No plays for trick_len=0
    result += (np.int64(7) & PLAY_MASK) << P1_SHIFT
    result += (np.int64(7) & PLAY_MASK) << P2_SHIFT

    # Keep original for mid-trick states
    result = np.where(mask, result, states)

    return result


# =============================================================================
# Seat Rotation Symmetry (Z2)
# =============================================================================

def seat_rotate_single(state: int) -> int:
    """
    Apply seat rotation symmetry to a single state.

    Rotates seats by 2: 0↔2, 1↔3
    Effect on V: V → V (partners swap, teams unchanged)

    Args:
        state: Packed int64 state

    Returns:
        Transformed packed state
    """
    remaining, leader, trick_len, p0, p1, p2 = unpack_state_single(state)

    # Swap hands: 0↔2, 1↔3
    new_remaining = np.array([
        remaining[2],
        remaining[3],
        remaining[0],
        remaining[1],
    ], dtype=np.uint8)

    # Transform leader: rotate by 2
    new_leader = (leader + 2) % 4

    if trick_len == 0:
        return pack_state_single(new_remaining, new_leader, 0, 7, 7, 7)
    else:
        # Similar complexity as team_swap for mid-trick states
        # For rotation by 2:
        # Old pos i (player (leader+i)%4) → new player ((leader+i)+2)%4
        # New pos for this player: ((leader+i+2) - new_leader) % 4
        #   = ((leader+i+2) - (leader+2)) % 4 = i
        # So positions stay the same! The plays don't need reordering.
        # But the local indices still refer to old hands...
        #
        # Actually for rotation by 2, the trick structure IS preserved
        # because the relative ordering of players in the trick stays same.
        # Player at old position i is (leader+i)%4
        # After rotation, player at new position i is (new_leader+i)%4 = (leader+2+i)%4
        # This is (old_player + 2)%4
        # So new position i is played by the same logical role (rotated by 2)
        #
        # The local indices p0,p1,p2 referred to hands[player][idx]
        # After rotation, hands have shifted:
        # new_hands[0] = old_hands[2], etc.
        # But player at new position i has new hand...
        #
        # Still can't simply preserve local indices.
        # Keep it simple: only full transform for trick_len=0
        return state


def seat_rotate(states: np.ndarray) -> np.ndarray:
    """
    Apply seat rotation symmetry to multiple states (vectorized).

    Only transforms trick_len=0 states correctly.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) transformed packed states
    """
    remaining, leader, trick_len, p0, p1, p2 = schema.unpack_state(states)

    # Swap hands: 0↔2, 1↔3
    new_remaining = np.zeros_like(remaining)
    new_remaining[:, 0] = remaining[:, 2]
    new_remaining[:, 1] = remaining[:, 3]
    new_remaining[:, 2] = remaining[:, 0]
    new_remaining[:, 3] = remaining[:, 1]

    # Transform leader
    new_leader = (leader + 2) % 4

    # Only transform trick_len=0 states
    mask = trick_len == 0

    # Pack results
    result = np.zeros_like(states)
    for i in range(4):
        result += new_remaining[:, i].astype(np.int64) << (i * REMAINING_BITS)
    result += (new_leader.astype(np.int64) & LEADER_MASK) << LEADER_SHIFT
    result += (trick_len.astype(np.int64) & TRICK_LEN_MASK) << TRICK_LEN_SHIFT
    result += (np.int64(7) & PLAY_MASK) << P0_SHIFT
    result += (np.int64(7) & PLAY_MASK) << P1_SHIFT
    result += (np.int64(7) & PLAY_MASK) << P2_SHIFT

    # Keep original for mid-trick states
    result = np.where(mask, result, states)

    return result


# =============================================================================
# Canonical Forms
# =============================================================================

def canonical_form_single(state: int) -> tuple[int, int]:
    """
    Compute canonical form of a single state under symmetry group.

    The canonical form is the lexicographically smallest state
    in the orbit under the Klein four-group (team_swap, seat_rotate).

    Args:
        state: Packed int64 state

    Returns:
        (canonical_state, transform_id) where transform_id is:
        0 = identity
        1 = team_swap
        2 = seat_rotate
        3 = team_swap + seat_rotate
    """
    remaining, leader, trick_len, p0, p1, p2 = unpack_state_single(state)

    # For mid-trick states, return as-is (no simple canonical form)
    if trick_len != 0:
        return state, 0

    # Generate orbit (4 elements for Klein four-group)
    orbit = [
        state,
        team_swap_single(state),
        seat_rotate_single(state),
        team_swap_single(seat_rotate_single(state)),
    ]

    # Find minimum (canonical)
    min_idx = 0
    min_state = orbit[0]
    for i in range(1, 4):
        if orbit[i] < min_state:
            min_state = orbit[i]
            min_idx = i

    return min_state, min_idx


def canonical_form(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute canonical forms for multiple states.

    Args:
        states: (N,) int64 packed states

    Returns:
        (canonical_states, transform_ids) each (N,)
    """
    # Generate all orbit elements
    s0 = states
    s1 = team_swap(states)
    s2 = seat_rotate(states)
    s3 = team_swap(seat_rotate(states))

    # Stack for comparison
    orbit = np.stack([s0, s1, s2, s3], axis=1)  # (N, 4)

    # Find minimum along axis 1
    canonical = orbit.min(axis=1)
    transform_ids = orbit.argmin(axis=1)

    return canonical, transform_ids


# =============================================================================
# Orbit Enumeration
# =============================================================================

def enumerate_orbits(
    states: np.ndarray,
    V: np.ndarray | None = None,
) -> dict[int, list[tuple[int, int | None]]]:
    """
    Group states by their orbits under symmetry.

    Args:
        states: (N,) int64 packed states
        V: Optional (N,) int8 values

    Returns:
        Dict mapping canonical_state -> list of (state, V) tuples
    """
    canonical, _ = canonical_form(states)

    orbits: dict[int, list[tuple[int, int | None]]] = {}

    for i, (state, canon) in enumerate(zip(states, canonical)):
        v = V[i] if V is not None else None
        state_int = int(state)
        canon_int = int(canon)

        if canon_int not in orbits:
            orbits[canon_int] = []
        orbits[canon_int].append((state_int, v))

    return orbits


def orbit_sizes(states: np.ndarray) -> np.ndarray:
    """
    Compute orbit size for each state.

    For states where all 4 symmetry transforms give distinct states,
    orbit size is 4. For states with fixed points, orbit is smaller.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) int orbit sizes (1, 2, or 4)
    """
    s0 = states
    s1 = team_swap(states)
    s2 = seat_rotate(states)
    s3 = team_swap(seat_rotate(states))

    # Count unique elements per state
    orbit = np.stack([s0, s1, s2, s3], axis=1)  # (N, 4)

    # Count unique values along axis 1
    sizes = np.zeros(len(states), dtype=np.int32)
    for i in range(len(states)):
        sizes[i] = len(np.unique(orbit[i]))

    return sizes


def orbit_compression_ratio(states: np.ndarray) -> float:
    """
    Compute compression ratio from symmetry.

    Compression ratio = num_states / num_orbits

    Args:
        states: (N,) int64 packed states

    Returns:
        Compression ratio (1.0 to 4.0)
    """
    canonical, _ = canonical_form(states)
    num_orbits = len(np.unique(canonical))
    return len(states) / num_orbits


# =============================================================================
# V Consistency Checking
# =============================================================================

def check_v_consistency(
    states: np.ndarray,
    V: np.ndarray,
) -> dict[str, float]:
    """
    Check if V values are consistent within orbits.

    For team_swap: V should negate
    For seat_rotate: V should be preserved

    Args:
        states: (N,) int64 packed states
        V: (N,) int8 values

    Returns:
        Dict with consistency metrics
    """
    # Filter to trick_len=0 states only
    _, _, trick_len, _, _, _ = schema.unpack_state(states)
    mask = trick_len == 0
    states_clean = states[mask]
    V_clean = V[mask]

    if len(states_clean) == 0:
        return {'error': 'No trick_len=0 states'}

    # Build state -> V lookup
    state_to_v = dict(zip(states_clean, V_clean))

    # Check team_swap: V should negate
    swapped = team_swap(states_clean)
    team_swap_consistent = 0
    team_swap_total = 0
    for i, (orig, swap) in enumerate(zip(states_clean, swapped)):
        if int(swap) in state_to_v:
            team_swap_total += 1
            expected_v = -V_clean[i]
            actual_v = state_to_v[int(swap)]
            if expected_v == actual_v:
                team_swap_consistent += 1

    # Check seat_rotate: V should be preserved
    rotated = seat_rotate(states_clean)
    seat_rotate_consistent = 0
    seat_rotate_total = 0
    for i, (orig, rot) in enumerate(zip(states_clean, rotated)):
        if int(rot) in state_to_v:
            seat_rotate_total += 1
            expected_v = V_clean[i]
            actual_v = state_to_v[int(rot)]
            if expected_v == actual_v:
                seat_rotate_consistent += 1

    return {
        'team_swap_consistency': team_swap_consistent / max(team_swap_total, 1),
        'team_swap_pairs': team_swap_total,
        'seat_rotate_consistency': seat_rotate_consistent / max(seat_rotate_total, 1),
        'seat_rotate_pairs': seat_rotate_total,
        'clean_states': len(states_clean),
    }
