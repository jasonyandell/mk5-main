"""State space navigation utilities for principal variation tracing.

This module provides functions to navigate the minimax game tree:
- pack_state(): Pack state components into int64
- compute_successor(): Compute state after a move
- get_children(): Get all legal successor states
- trace_principal_variation(): Follow optimal play to terminal
- build_state_lookup(): Create state -> (V, Q) lookup from DataFrame
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from forge.oracle import schema, tables

if TYPE_CHECKING:
    import pandas as pd


# =============================================================================
# State Packing
# =============================================================================

def pack_state(
    remaining: np.ndarray,
    leader: int,
    trick_len: int,
    p0: int = 7,
    p1: int = 7,
    p2: int = 7,
) -> int:
    """
    Pack state components into int64.

    Args:
        remaining: (4,) uint8 - 7-bit masks per player
        leader: current trick leader (0-3)
        trick_len: plays so far in current trick (0-3)
        p0, p1, p2: local indices of plays (7 = no play)

    Returns:
        Packed int64 state
    """
    state = 0
    for p in range(4):
        state |= int(remaining[p]) << (p * 7)
    state |= (leader & 0x3) << 28
    state |= (trick_len & 0x3) << 30
    state |= (p0 & 0x7) << 32
    state |= (p1 & 0x7) << 35
    state |= (p2 & 0x7) << 38
    return state


def unpack_state_single(state: int) -> tuple[np.ndarray, int, int, int, int, int]:
    """
    Unpack a single int64 state.

    Args:
        state: Packed int64 state

    Returns:
        (remaining, leader, trick_len, p0, p1, p2)
    """
    remaining = np.zeros(4, dtype=np.uint8)
    for p in range(4):
        remaining[p] = (state >> (p * 7)) & 0x7F
    leader = (state >> 28) & 0x3
    trick_len = (state >> 30) & 0x3
    p0 = (state >> 32) & 0x7
    p1 = (state >> 35) & 0x7
    p2 = (state >> 38) & 0x7
    return remaining, leader, trick_len, p0, p1, p2


# =============================================================================
# State Transitions
# =============================================================================

def compute_successor(
    state: int,
    local_idx: int,
    seed: int,
    decl_id: int,
) -> int:
    """
    Compute the successor state after playing a domino.

    Args:
        state: Current packed state
        local_idx: Local index (0-6) of domino to play
        seed: Deal seed (for resolving tricks)
        decl_id: Declaration ID

    Returns:
        Successor packed state
    """
    remaining, leader, trick_len, p0, p1, p2 = unpack_state_single(state)
    current_player = (leader + trick_len) % 4

    # Remove domino from player's hand
    new_remaining = remaining.copy()
    # Use 0x7F mask to handle Python's arbitrary precision negative integers
    new_remaining[current_player] &= (~(1 << local_idx)) & 0x7F

    if trick_len < 3:
        # Add play to current trick
        new_plays = [p0, p1, p2]
        new_plays[trick_len] = local_idx
        new_trick_len = trick_len + 1
        return pack_state(new_remaining, leader, new_trick_len, *new_plays)
    else:
        # Completing trick (4th play) - resolve and start new trick
        hands = schema.deal_from_seed(seed)

        # Get global domino IDs for all 4 plays
        domino_ids = (
            hands[leader][p0],
            hands[(leader + 1) % 4][p1],
            hands[(leader + 2) % 4][p2],
            hands[current_player][local_idx],
        )

        # Resolve trick to find winner
        outcome = tables.resolve_trick(domino_ids[0], domino_ids, decl_id)
        new_leader = (leader + outcome.winner_offset) % 4

        return pack_state(new_remaining, new_leader, 0, 7, 7, 7)


def get_legal_moves(q_values: np.ndarray) -> list[int]:
    """
    Get legal move indices from Q-values.

    Args:
        q_values: (7,) Q-values (-128 = illegal)

    Returns:
        List of legal local indices (0-6)
    """
    ILLEGAL = -128
    return [i for i in range(7) if q_values[i] != ILLEGAL]


def get_children(
    state: int,
    q_values: np.ndarray,
    seed: int,
    decl_id: int,
) -> list[tuple[int, int, int]]:
    """
    Get all legal successor states with their move indices and Q-values.

    Args:
        state: Current packed state
        q_values: (7,) Q-values for this state
        seed: Deal seed
        decl_id: Declaration ID

    Returns:
        List of (successor_state, local_idx, q_value) tuples
    """
    children = []
    for local_idx in get_legal_moves(q_values):
        successor = compute_successor(state, local_idx, seed, decl_id)
        children.append((successor, local_idx, q_values[local_idx]))
    return children


# =============================================================================
# State Lookup
# =============================================================================

def build_state_lookup(
    df: "pd.DataFrame",
) -> dict[int, tuple[int, np.ndarray]]:
    """
    Build state -> (V, Q-values) lookup from DataFrame.

    Args:
        df: DataFrame with state, V, q0-q6 columns

    Returns:
        Dict mapping state -> (V, (7,) Q-values array)
    """
    q_cols = ["q0", "q1", "q2", "q3", "q4", "q5", "q6"]
    lookup = {}
    for _, row in df.iterrows():
        state = row["state"]
        v = row["V"]
        q = np.array([row[c] for c in q_cols], dtype=np.int8)
        lookup[state] = (v, q)
    return lookup


def build_state_lookup_fast(
    df: "pd.DataFrame",
) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    """
    Build fast state lookup using numpy arrays.

    Args:
        df: DataFrame with state, V, q0-q6 columns

    Returns:
        (state_to_idx, V_array, Q_array) where:
        - state_to_idx: dict mapping state -> row index
        - V_array: (N,) int8 V values
        - Q_array: (N, 7) int8 Q values
    """
    q_cols = ["q0", "q1", "q2", "q3", "q4", "q5", "q6"]

    states = df["state"].values
    V = df["V"].values
    Q = df[q_cols].values.astype(np.int8)

    state_to_idx = {s: i for i, s in enumerate(states)}
    return state_to_idx, V, Q


# =============================================================================
# Principal Variation Tracing
# =============================================================================

def trace_principal_variation(
    start_state: int,
    seed: int,
    decl_id: int,
    state_to_idx: dict[int, int],
    V: np.ndarray,
    Q: np.ndarray,
    max_depth: int = 100,
) -> list[tuple[int, int, int]]:
    """
    Trace principal variation (optimal play) from a state to terminal.

    Args:
        start_state: Starting packed state
        seed: Deal seed
        decl_id: Declaration ID
        state_to_idx: State -> row index mapping
        V: (N,) V values array
        Q: (N, 7) Q values array
        max_depth: Maximum trace depth (safety limit)

    Returns:
        List of (state, V, best_move) tuples along the PV.
        Terminal state has best_move = -1.
    """
    pv = []
    current_state = start_state

    for _ in range(max_depth):
        if current_state not in state_to_idx:
            break

        idx = state_to_idx[current_state]
        v = V[idx]
        q = Q[idx]

        # Check for terminal (all Q illegal or no remaining dominoes)
        legal = get_legal_moves(q)
        if not legal:
            pv.append((current_state, v, -1))
            break

        # Find optimal move
        # Team 0 maximizes, Team 1 minimizes
        remaining, leader, trick_len, _, _, _ = unpack_state_single(current_state)
        current_player = (leader + trick_len) % 4
        is_team0 = (current_player % 2) == 0

        if is_team0:
            best_idx = max(legal, key=lambda i: q[i])
        else:
            best_idx = min(legal, key=lambda i: q[i])

        pv.append((current_state, v, best_idx))

        # Move to successor
        current_state = compute_successor(current_state, best_idx, seed, decl_id)

    return pv


def trace_to_terminal_outcome(
    start_state: int,
    seed: int,
    decl_id: int,
    state_to_idx: dict[int, int],
    V: np.ndarray,
    Q: np.ndarray,
) -> tuple[int, list[int]]:
    """
    Trace to terminal and return final V and moves made.

    Args:
        start_state: Starting packed state
        seed: Deal seed
        decl_id: Declaration ID
        state_to_idx: State -> row index mapping
        V: V values array
        Q: Q values array

    Returns:
        (terminal_V, moves) where moves is list of local indices played
    """
    pv = trace_principal_variation(
        start_state, seed, decl_id, state_to_idx, V, Q
    )
    if not pv:
        return 0, []

    terminal_v = pv[-1][1]
    moves = [m for _, _, m in pv if m >= 0]
    return terminal_v, moves


# =============================================================================
# Count Domino Tracking
# =============================================================================

def track_count_captures(
    start_state: int,
    seed: int,
    decl_id: int,
    state_to_idx: dict[int, int],
    V: np.ndarray,
    Q: np.ndarray,
) -> dict[int, int]:
    """
    Track which team captures each count domino along the PV.

    Args:
        start_state: Starting packed state
        seed: Deal seed
        decl_id: Declaration ID
        state_to_idx: State -> row index mapping
        V: V values array
        Q: Q values array

    Returns:
        Dict mapping domino_id -> capturing_team (0 or 1)
        Only includes count dominoes (5-count and 10-count).
    """
    from forge.analysis.utils.features import COUNT_DOMINO_IDS

    hands = schema.deal_from_seed(seed)

    # Track which count dominoes are still in play at start
    remaining, _, _, _, _, _ = unpack_state_single(start_state)

    # Build domino -> (player, local_idx) lookup
    domino_to_location = {}
    for p in range(4):
        for local_idx, domino_id in enumerate(hands[p]):
            domino_to_location[domino_id] = (p, local_idx)

    # Check which counts are still in play
    counts_in_play = set()
    for domino_id in COUNT_DOMINO_IDS:
        p, local_idx = domino_to_location[domino_id]
        if (remaining[p] >> local_idx) & 1:
            counts_in_play.add(domino_id)

    if not counts_in_play:
        return {}

    # Trace PV and track trick outcomes
    captures = {}
    pv = trace_principal_variation(
        start_state, seed, decl_id, state_to_idx, V, Q
    )

    # Process each move in the PV
    for i, (state, v, move) in enumerate(pv):
        if move < 0:
            break

        remaining, leader, trick_len, p0, p1, p2 = unpack_state_single(state)
        current_player = (leader + trick_len) % 4

        # Get domino being played
        played_domino = hands[current_player][move]

        # If completing a trick, determine who wins and what counts are captured
        if trick_len == 3:
            # This is the 4th play
            trick_plays = [p0, p1, p2, move]
            trick_dominos = tuple(
                hands[(leader + j) % 4][trick_plays[j]]
                for j in range(4)
            )

            outcome = tables.resolve_trick(trick_dominos[0], trick_dominos, decl_id)
            winner = (leader + outcome.winner_offset) % 4
            winning_team = winner % 2

            # Check which count dominoes are in this trick
            for j, domino_id in enumerate(trick_dominos):
                if domino_id in counts_in_play and domino_id not in captures:
                    captures[domino_id] = winning_team

    return captures


def count_capture_signature(
    captures: dict[int, int],
) -> tuple[int, int]:
    """
    Summarize count captures as (team0_points, team1_points).

    Args:
        captures: Dict mapping domino_id -> capturing_team

    Returns:
        (team0_count_points, team1_count_points)
    """
    team0 = 0
    team1 = 0
    for domino_id, team in captures.items():
        points = tables.DOMINO_COUNT_POINTS[domino_id]
        if team == 0:
            team0 += points
        else:
            team1 += points
    return team0, team1
