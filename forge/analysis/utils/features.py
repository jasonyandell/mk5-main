"""Feature extraction from packed game states.

This module provides functions to extract analytical features from packed state arrays:
- depth(): Total dominoes remaining
- team(), player(): Current team/player to move
- count_locations(): Who holds each count domino
- counts_remaining(): Total count points still in play
- extract_all(): Comprehensive feature DataFrame
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from forge.oracle import schema, tables

if TYPE_CHECKING:
    import pandas as pd


# Count domino info from tables.py
# 5-count: (5,0), (4,1), (3,2) -> domino IDs 15, 10, 6
# 10-count: (5,5), (6,4) -> domino IDs 20, 25
COUNT_DOMINO_IDS = [
    d for d in range(28) if tables.DOMINO_COUNT_POINTS[d] > 0
]
COUNT_DOMINO_POINTS = {d: tables.DOMINO_COUNT_POINTS[d] for d in COUNT_DOMINO_IDS}


def depth(states: np.ndarray) -> np.ndarray:
    """
    Compute total dominoes remaining (game depth).

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) int32 - dominoes remaining (0-28)
    """
    remaining, _, _, _, _, _ = schema.unpack_state(states)
    return schema.remaining_count(remaining)


def player(states: np.ndarray) -> np.ndarray:
    """
    Compute current player to move.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) uint8 - player index (0-3)
    """
    _, leader, trick_len, _, _, _ = schema.unpack_state(states)
    return schema.current_player(leader, trick_len)


def team(states: np.ndarray) -> np.ndarray:
    """
    Compute current team to move.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) bool - True if team 0's turn
    """
    _, leader, trick_len, _, _, _ = schema.unpack_state(states)
    return schema.current_team(leader, trick_len)


def hand_balance(states: np.ndarray) -> np.ndarray:
    """
    Compute team 0's domino advantage.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) int32 - (team 0 dominoes) - (team 1 dominoes)
    """
    remaining, _, _, _, _, _ = schema.unpack_state(states)

    # Count per player
    counts = np.zeros((len(states), 4), dtype=np.int32)
    for p in range(4):
        for bit in range(7):
            counts[:, p] += (remaining[:, p] >> bit) & 1

    # Team balance: (P0 + P2) - (P1 + P3)
    team0 = counts[:, 0] + counts[:, 2]
    team1 = counts[:, 1] + counts[:, 3]
    return team0 - team1


def count_locations(
    states: np.ndarray,
    seed: int,
) -> dict[int, np.ndarray]:
    """
    Track which player holds each count domino (or -1 if played).

    Args:
        states: (N,) int64 packed states
        seed: Deal seed for hand reconstruction

    Returns:
        Dict mapping domino_id -> (N,) int8 array of player indices (-1 = played)
    """
    hands = schema.deal_from_seed(seed)
    remaining, _, _, _, _, _ = schema.unpack_state(states)

    # Build reverse lookup: domino_id -> (player, local_idx)
    domino_location = {}
    for p in range(4):
        for local_idx, domino_id in enumerate(hands[p]):
            domino_location[domino_id] = (p, local_idx)

    result = {}
    for domino_id in COUNT_DOMINO_IDS:
        p, local_idx = domino_location[domino_id]
        # Check if bit local_idx is set in remaining[p]
        has_domino = (remaining[:, p] >> local_idx) & 1
        # Player p if has it, -1 if played
        holder = np.where(has_domino, p, -1).astype(np.int8)
        result[domino_id] = holder

    return result


def counts_remaining(
    states: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Compute total count points still in play.

    Args:
        states: (N,) int64 packed states
        seed: Deal seed

    Returns:
        (N,) int32 - total count points remaining (0-35)
    """
    locations = count_locations(states, seed)
    total = np.zeros(len(states), dtype=np.int32)
    for domino_id, holders in locations.items():
        points = tables.DOMINO_COUNT_POINTS[domino_id]
        # Add points if domino still held (holder >= 0)
        total += np.where(holders >= 0, points, 0)
    return total


def counts_by_team(
    states: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute count points held by each team.

    Args:
        states: (N,) int64 packed states
        seed: Deal seed

    Returns:
        (team0_counts, team1_counts) - each (N,) int32
    """
    locations = count_locations(states, seed)
    team0 = np.zeros(len(states), dtype=np.int32)
    team1 = np.zeros(len(states), dtype=np.int32)

    for domino_id, holders in locations.items():
        points = tables.DOMINO_COUNT_POINTS[domino_id]
        # Team 0: players 0, 2
        team0 += np.where((holders == 0) | (holders == 2), points, 0)
        # Team 1: players 1, 3
        team1 += np.where((holders == 1) | (holders == 3), points, 0)

    return team0, team1


def trick_info(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract current trick information.

    Args:
        states: (N,) int64 packed states

    Returns:
        (leader, trick_len) - each (N,) uint8
    """
    _, leader, trick_len, _, _, _ = schema.unpack_state(states)
    return leader, trick_len


def extract_all(
    states: np.ndarray,
    seed: int,
    decl_id: int,
    V: np.ndarray | None = None,
) -> "pd.DataFrame":
    """
    Extract comprehensive feature DataFrame.

    Args:
        states: (N,) int64 packed states
        seed: Deal seed
        decl_id: Declaration ID
        V: Optional (N,) int8 values to include

    Returns:
        DataFrame with columns:
        - state, seed, decl_id
        - depth, player, team, hand_balance
        - leader, trick_len
        - counts_remaining, team0_counts, team1_counts
        - count_holder_{domino_id} for each count domino
        - V (if provided)
    """
    import pandas as pd

    remaining, leader, trick_len, p0, p1, p2 = schema.unpack_state(states)

    df = pd.DataFrame({
        "state": states,
        "seed": seed,
        "decl_id": decl_id,
        "depth": schema.remaining_count(remaining),
        "player": schema.current_player(leader, trick_len),
        "team": schema.current_team(leader, trick_len).astype(np.int8),
        "hand_balance": hand_balance(states),
        "leader": leader,
        "trick_len": trick_len,
    })

    # Count domino info
    df["counts_remaining"] = counts_remaining(states, seed)
    team0_counts, team1_counts = counts_by_team(states, seed)
    df["team0_counts"] = team0_counts
    df["team1_counts"] = team1_counts

    # Individual count domino locations
    locations = count_locations(states, seed)
    for domino_id, holders in locations.items():
        pips = schema.domino_pips(domino_id)
        col_name = f"count_{pips[0]}{pips[1]}_holder"
        df[col_name] = holders

    if V is not None:
        df["V"] = V

    return df


def q_stats(q_values: np.ndarray) -> "pd.DataFrame":
    """
    Compute Q-value statistics per state.

    Args:
        q_values: (N, 7) int8 Q-values (-128 = illegal)

    Returns:
        DataFrame with columns:
        - n_legal: Number of legal moves
        - q_spread: max(Q) - min(Q) among legal moves
        - q_gap: Best Q - second best Q (0 if <=1 legal)
        - n_optimal: Number of moves tied for best Q
        - best_q: Best Q value
    """
    import pandas as pd

    ILLEGAL = -128
    n = len(q_values)

    # Mask illegal moves
    legal_mask = q_values != ILLEGAL

    n_legal = legal_mask.sum(axis=1)

    # Replace illegal with very negative for max, very positive for min
    q_for_max = np.where(legal_mask, q_values, -1000)
    q_for_min = np.where(legal_mask, q_values, 1000)

    best_q = q_for_max.max(axis=1)
    worst_q = q_for_min.min(axis=1)
    q_spread = best_q - worst_q

    # For q_gap: need second best
    # Sort descending, take difference between 0th and 1st
    q_sorted = np.sort(q_for_max, axis=1)[:, ::-1]  # descending
    q_gap = np.where(n_legal > 1, q_sorted[:, 0] - q_sorted[:, 1], 0)

    # Count optimal moves (ties)
    n_optimal = (q_values == best_q[:, np.newaxis]).sum(axis=1)

    return pd.DataFrame({
        "n_legal": n_legal,
        "q_spread": q_spread,
        "q_gap": q_gap,
        "n_optimal": n_optimal,
        "best_q": best_q,
    })
