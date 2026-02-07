"""
forge.oracle Parquet Schema
===========================

This module documents the parquet file format produced by the oracle solver and provides
utilities for loading/decoding the data.

File Naming
-----------
    seed_{SEED:08d}_decl_{DECL_ID}.parquet

Parquet Columns
---------------
    state : int64
        Packed game state (41 bits used). See STATE_LAYOUT below.

    V : int8
        Minimax value from this state. Range [-42, +42].

        **V Semantics (Team 0 perspective):**
        - V is a minimax **value-to-go**: the expected remaining (Team 0 − Team 1) point differential from this
          state to the end of the hand, assuming perfect play by all players with full information.
        - Positive V = Team 0 is advantaged from here; negative V = Team 1 is advantaged from here.
        - Terminal states have no remaining points to win, so `V(terminal) = 0`.
        - At the initial state (start of hand), V equals the final hand point differential because no points have been
          scored yet.

        **Important**: The packed `state` does not encode "score so far" or trick history, so V cannot represent an
        accumulated score; it must be value-to-go.

    q0..q6 : int8
        Q-values (Q*): Optimal value of playing local domino index 0-6 from current player's hand.

        **Q-value Semantics (Team 0 perspective):**
        - All Q-values are from Team 0's perspective, regardless of whose turn it is
        - -128 indicates illegal move (domino not in hand or can't follow suit)
        - Legal moves have values in [-42, +42]
        - Q-values are in the same units as V (remaining point differential from this move onward)
        - For Team 0's turn: argmax(Q) is optimal
        - For Team 1's turn: argmin(Q) is optimal (they minimize Team 0's value)

Parquet Metadata
----------------
    seed : str (encoded int)
        Deal seed. Use deal_from_seed(seed) to get the 4 hands.

    decl_id : str (encoded int)
        Declaration type (0-9). See DECL_NAMES below.


STATE_LAYOUT (41 bits packed in int64)
--------------------------------------
    **State Packing: 41 bits in int64**

    Bits 0-27:  Remaining hands (4 x 7-bit masks)
        [0:7]   remaining[0] - Player 0's local indices still in hand
        [7:14]  remaining[1] - Player 1's
        [14:21] remaining[2] - Player 2's
        [21:28] remaining[3] - Player 3's

    Bits 28-29: leader (2 bits)
        Current trick leader (0-3)

    Bits 30-31: trick_len (2 bits)
        Plays so far in current trick (0-3)

    Bits 32-40: Current trick plays (3 x 3 bits)
        [32:35] p0 - Leader's local index (0-6, or 7 if N/A)
        [35:38] p1 - Second player's local index
        [38:41] p2 - Third player's local index

    p3 (fourth play) is not stored because the trick resolves immediately
    when the 4th domino is played.

Why Local Indices?
------------------
    Each player holds 7 dominoes (indices 0-6 in their hand).
    The parquet stores LOCAL indices, not global domino IDs (0-27).
    To convert local -> global, use the deal: hands[player][local_idx] -> domino_id

    This makes the state encoding deal-independent (same bit layout for all seeds).


Example Usage
-------------
    from forge.oracle.schema import load_file, unpack_state, DECL_NAMES

    # Load a file
    df, seed, decl_id = load_file("data/shards/seed_00000000_decl_5.parquet")

    # Access columns
    states = df["state"].values   # (N,) int64
    values = df["V"].values       # (N,) int8
    move_values = df[["q0", "q1", "q2", "q3", "q4", "q5", "q6"]].values

    # Decode states
    remaining, leader, trick_len, p0, p1, p2 = unpack_state(states)
    # remaining: (N, 4) - 7-bit masks per player
    # leader, trick_len: (N,) - current trick info
    # p0, p1, p2: (N,) - local indices of plays so far (7 = no play yet)

    # Human-readable declaration
    print(f"Declaration: {DECL_NAMES[decl_id]}")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# =============================================================================
# Declaration Types (re-exported from declarations.py for convenience)
# =============================================================================

from .declarations import DECL_ID_TO_NAME as DECL_NAMES
from .declarations import DECL_NAME_TO_ID

# =============================================================================
# State Bit Layout Constants
# =============================================================================

# Remaining masks (4 players x 7 bits each)
REMAINING_BITS = 7
REMAINING_MASK = 0x7F

# Leader: 2 bits at position 28
LEADER_SHIFT = 28
LEADER_MASK = 0x3

# Trick length: 2 bits at position 30
TRICK_LEN_SHIFT = 30
TRICK_LEN_MASK = 0x3

# Trick plays: 3 bits each
P0_SHIFT = 32
P1_SHIFT = 35
P2_SHIFT = 38
PLAY_MASK = 0x7


# =============================================================================
# Loading
# =============================================================================


def load_file(path: str | Path) -> tuple["pd.DataFrame", int, int]:
    """
    Load an oracle parquet file.

    Returns:
        (df, seed, decl_id) where:
        - df: DataFrame with columns [state, V, q0..q6]
        - seed: int - deal seed
        - decl_id: int - declaration type (0-9)
    """
    import pandas as pd
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = pf.schema_arrow.metadata or {}
    seed = int(meta.get(b"seed", b"0").decode())
    decl_id = int(meta.get(b"decl_id", b"0").decode())

    df = pd.read_parquet(path)
    return df, seed, decl_id


def load_states_only(path: str | Path) -> tuple[np.ndarray, int, int]:
    """
    Load just the states from a parquet file (faster, less memory).

    Returns:
        (states, seed, decl_id) where states is (N,) int64 array
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = pf.schema_arrow.metadata or {}
    seed = int(meta.get(b"seed", b"0").decode())
    decl_id = int(meta.get(b"decl_id", b"0").decode())

    table = pf.read(columns=["state"])
    states = table["state"].to_numpy()
    return states, seed, decl_id


# =============================================================================
# State Unpacking
# =============================================================================


def unpack_state(
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack int64 states into components.

    Args:
        states: (N,) int64 packed states

    Returns:
        remaining: (N, 4) uint8 - 7-bit masks per player (bit i set = has local domino i)
        leader: (N,) uint8 - current trick leader (0-3)
        trick_len: (N,) uint8 - plays so far in current trick (0-3)
        p0: (N,) uint8 - leader's play (0-6, or 7 if none)
        p1: (N,) uint8 - 2nd player's play
        p2: (N,) uint8 - 3rd player's play
    """
    remaining = np.zeros((len(states), 4), dtype=np.uint8)
    for p in range(4):
        remaining[:, p] = (states >> (p * REMAINING_BITS)) & REMAINING_MASK

    leader = ((states >> LEADER_SHIFT) & LEADER_MASK).astype(np.uint8)
    trick_len = ((states >> TRICK_LEN_SHIFT) & TRICK_LEN_MASK).astype(np.uint8)
    p0 = ((states >> P0_SHIFT) & PLAY_MASK).astype(np.uint8)
    p1 = ((states >> P1_SHIFT) & PLAY_MASK).astype(np.uint8)
    p2 = ((states >> P2_SHIFT) & PLAY_MASK).astype(np.uint8)

    return remaining, leader, trick_len, p0, p1, p2


def current_player(leader: np.ndarray, trick_len: np.ndarray) -> np.ndarray:
    """
    Compute whose turn it is.

    Args:
        leader: (N,) - current trick leader
        trick_len: (N,) - plays so far

    Returns:
        player: (N,) - current player (0-3)
    """
    return (leader + trick_len) % 4


def current_team(leader: np.ndarray, trick_len: np.ndarray) -> np.ndarray:
    """
    Compute which team is to play.

    Args:
        leader: (N,) - current trick leader
        trick_len: (N,) - plays so far

    Returns:
        is_team0: (N,) bool - True if team 0's turn
    """
    player = (leader + trick_len) % 4
    return (player % 2) == 0


def remaining_count(remaining: np.ndarray) -> np.ndarray:
    """
    Count total dominoes remaining across all hands.

    Args:
        remaining: (N, 4) - 7-bit masks per player

    Returns:
        count: (N,) - total dominoes remaining (0-28)
    """
    # Popcount for each mask
    total = np.zeros(len(remaining), dtype=np.int32)
    for p in range(4):
        for bit in range(7):
            total += (remaining[:, p] >> bit) & 1
    return total


# =============================================================================
# Deal Reconstruction
# =============================================================================

# Domino IDs 0-27 mapped to (high, low) pips
# ID = h*(h+1)/2 + l for h >= l
DOMINOES: list[tuple[int, int]] = [(h, l) for h in range(7) for l in range(h + 1)]


def deal_from_seed(seed: int) -> list[list[int]]:
    """
    Reconstruct the 4 hands from a seed.

    Args:
        seed: deal seed

    Returns:
        hands: list of 4 lists, each containing 7 domino IDs (0-27), sorted
    """
    import random

    rng = random.Random(seed)
    dominos = list(range(28))
    rng.shuffle(dominos)
    return [sorted(dominos[i * 7 : (i + 1) * 7]) for i in range(4)]


def local_to_global(seed: int, player: int, local_idx: int) -> int:
    """
    Convert a local domino index to global domino ID.

    Args:
        seed: deal seed
        player: player index (0-3)
        local_idx: local index in player's hand (0-6)

    Returns:
        domino_id: global domino ID (0-27)
    """
    hands = deal_from_seed(seed)
    return hands[player][local_idx]


def domino_pips(domino_id: int) -> tuple[int, int]:
    """
    Get the pip values for a domino.

    Args:
        domino_id: global domino ID (0-27)

    Returns:
        (high, low): pip values, high >= low
    """
    return DOMINOES[domino_id]


# =============================================================================
# Value Interpretation
# =============================================================================


def normalize_value(v: np.ndarray) -> np.ndarray:
    """
    Normalize V values to [-1, 1] range.

    Args:
        v: int8 values in [-42, +42]

    Returns:
        normalized: float32 in [-1, +1]
    """
    return v.astype(np.float32) / 42.0
