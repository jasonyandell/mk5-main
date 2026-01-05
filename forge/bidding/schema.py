"""
forge.bidding Parquet Schema
============================

This module documents the parquet file format produced by the continuous bidding
evaluation CLI and provides utilities for loading the data.

File Naming
-----------
    seed_{SEED:08d}.parquet

    Files are stored in train/val/test subdirectories based on seed % 1000:
    - 0-899: train/
    - 900-949: val/
    - 950-999: test/

Parquet Columns
---------------
    seed : int64
        RNG seed used to generate P0's hand via deal_from_seed(seed).

    hand : string
        P0's hand as comma-separated "high-low" pairs (e.g., "6-4,5-5,4-2,3-1,2-0,1-1,0-0").

    n_samples : int32
        Number of Monte Carlo simulations per declaration.

    model_checkpoint : string
        Path to the policy model checkpoint used for simulations.

    timestamp : string
        ISO format timestamp when evaluation was run.

    points_{decl} : list<int8>
        Raw simulation results for declaration {decl}. Array of n_samples values,
        each representing Team 0's final points (0-84) in one simulated game.

        Declarations evaluated: 0-7 and 9 (skips 8=doubles-suit)
        - 0: blanks
        - 1: ones
        - 2: twos
        - 3: threes
        - 4: fours
        - 5: fives
        - 6: sixes
        - 7: doubles-trump
        - 9: notrump

    pmake_{decl}_{bid} : float32
        P(make) for declaration {decl} at bid threshold {bid}.
        Computed as: count(points >= bid) / n_samples

        Bid thresholds: 30, 31, 32, ..., 42 (13 values)

    ci_low_{decl}_{bid} : float32
        Wilson score confidence interval lower bound (95% CI) for pmake_{decl}_{bid}.

    ci_high_{decl}_{bid} : float32
        Wilson score confidence interval upper bound (95% CI) for pmake_{decl}_{bid}.

Column Count
------------
    Total: 365 columns
    - 5 metadata columns (seed, hand, n_samples, model_checkpoint, timestamp)
    - 9 points columns (one per declaration)
    - 351 statistics columns (9 decls × 13 bids × 3 values per bid)

Example Usage
-------------
    from forge.bidding.schema import load_file, DECL_NAMES

    # Load a file
    df, seed, hand = load_file("data/bidding-results/train/seed_00000042.parquet")

    # Access raw points for sixes (decl 6)
    points_sixes = df["points_6"].iloc[0]  # list of int8

    # Access P(make 42) for sixes
    p_make = df["pmake_6_42"].iloc[0]
    ci_low = df["ci_low_6_42"].iloc[0]
    ci_high = df["ci_high_6_42"].iloc[0]

    # Recompute P(make) for custom threshold
    import numpy as np
    points = np.array(points_sixes)
    p_make_35 = (points >= 35).mean()

Relationship to Oracle Schema
-----------------------------
    The oracle schema (forge.oracle.schema) stores perfect-play Q-values computed
    via minimax search. This bidding schema stores Monte Carlo simulation results
    using a trained policy model.

    - Oracle: "What is the optimal value of each move?"
    - Bidding: "How often does the AI win when playing this hand?"

    The bidding evaluation is useful for:
    - Training a bidding model (System 2)
    - Evaluating model quality across diverse hands
    - Understanding hand strength distributions
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from forge.oracle.declarations import DECL_ID_TO_NAME as DECL_NAMES

# Re-export for convenience
__all__ = ["DECL_NAMES", "load_file", "load_points_only", "BID_THRESHOLDS", "EVAL_DECLS"]

# Declarations evaluated (skips 8=doubles-suit)
EVAL_DECLS = [0, 1, 2, 3, 4, 5, 6, 7, 9]

# Bid thresholds
BID_THRESHOLDS = list(range(30, 43))


def load_file(path: str | Path) -> tuple["pd.DataFrame", int, str]:
    """
    Load a bidding evaluation parquet file.

    Returns:
        (df, seed, hand) where:
        - df: DataFrame with all columns
        - seed: int - the seed used to generate the hand
        - hand: str - the hand in "high-low,..." format
    """
    import pandas as pd

    df = pd.read_parquet(path)
    seed = int(df["seed"].iloc[0])
    hand = str(df["hand"].iloc[0])

    return df, seed, hand


def load_points_only(path: str | Path) -> tuple[dict[int, np.ndarray], int, str]:
    """
    Load just the raw points arrays (faster, less memory).

    Returns:
        (points_dict, seed, hand) where:
        - points_dict: {decl_id: np.ndarray} mapping declaration to points array
        - seed: int - the seed
        - hand: str - the hand string
    """
    import pandas as pd

    columns = ["seed", "hand"] + [f"points_{d}" for d in EVAL_DECLS]
    df = pd.read_parquet(path, columns=columns)

    seed = int(df["seed"].iloc[0])
    hand = str(df["hand"].iloc[0])

    points_dict = {}
    for decl in EVAL_DECLS:
        points_dict[decl] = np.array(df[f"points_{decl}"].iloc[0], dtype=np.int8)

    return points_dict, seed, hand


def get_pmake_matrix(df: "pd.DataFrame") -> np.ndarray:
    """
    Extract P(make) values as a 2D matrix.

    Returns:
        np.ndarray of shape (9, 13) where:
        - rows are declarations (in EVAL_DECLS order: 0,1,2,3,4,5,6,7,9)
        - columns are bid thresholds (30-42)
    """
    matrix = np.zeros((len(EVAL_DECLS), len(BID_THRESHOLDS)), dtype=np.float32)

    for i, decl in enumerate(EVAL_DECLS):
        for j, bid in enumerate(BID_THRESHOLDS):
            matrix[i, j] = df[f"pmake_{decl}_{bid}"].iloc[0]

    return matrix
