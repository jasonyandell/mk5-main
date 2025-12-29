#!/usr/bin/env python3
"""
Diagnostic: Verify team extraction and label computation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.rng import deal_from_seed


def main():
    # Load a sample file
    data_dir = Path("data/solver2")
    f = next(data_dir.glob("seed_00000000_*.parquet"))

    pf = pq.ParquetFile(f)
    meta = pf.schema_arrow.metadata or {}
    seed = int(meta.get(b"seed", b"0").decode())
    decl_id = int(meta.get(b"decl_id", b"0").decode())

    print(f"File: {f.name}")
    print(f"Seed: {seed}, Decl: {decl_id}")

    df = pd.read_parquet(f)
    states = df["state"].values.astype(np.int64)

    mv_cols = [f"mv{i}" for i in range(7)]
    q_values = np.stack([df[c].values for c in mv_cols], axis=1)

    hands = deal_from_seed(seed)

    print("\n=== State Field Extraction Test ===")
    print("Checking first 10 states:\n")

    for i in range(min(10, len(states))):
        state = states[i]
        q = q_values[i]

        # Extract fields
        remaining = [(state >> (p * 7)) & 0x7F for p in range(4)]
        leader = (state >> 28) & 0x3
        trick_len = (state >> 30) & 0x3

        current_player = (leader + trick_len) % 4
        team = current_player % 2

        # Find legal moves
        legal = [j for j in range(7) if q[j] != -128]

        # Team-aware optimal
        legal_q = [(j, q[j]) for j in legal]
        if team == 0:
            optimal = max(legal_q, key=lambda x: x[1])
            opt_type = "max"
        else:
            optimal = min(legal_q, key=lambda x: x[1])
            opt_type = "min"

        print(f"State {i}:")
        print(f"  leader={leader}, trick_len={trick_len}")
        print(f"  current_player = ({leader} + {trick_len}) % 4 = {current_player}")
        print(f"  team = {current_player} % 2 = {team}")
        print(f"  Q-values: {q.tolist()}")
        print(f"  Legal moves: {legal}")
        print(f"  Optimal ({opt_type}): idx={optimal[0]}, Q={optimal[1]}")
        print()

    print("\n=== Team Distribution Check ===")
    leaders = ((states >> 28) & 0x3).astype(np.int64)
    trick_lens = ((states >> 30) & 0x3).astype(np.int64)
    current_players = (leaders + trick_lens) % 4
    teams = current_players % 2

    print(f"current_player distribution:")
    for p in range(4):
        count = (current_players == p).sum()
        print(f"  Player {p}: {count:,} ({count/len(states)*100:.1f}%)")

    print(f"\nteam distribution:")
    for t in range(2):
        count = (teams == t).sum()
        print(f"  Team {t}: {count:,} ({count/len(states)*100:.1f}%)")

    print("\n=== Verify Team Assignment ===")
    print("Player 0 -> Team 0:", 0 % 2 == 0)
    print("Player 1 -> Team 1:", 1 % 2 == 1)
    print("Player 2 -> Team 0:", 2 % 2 == 0)
    print("Player 3 -> Team 1:", 3 % 2 == 1)

    # Check token embedding - is_current and is_partner flags
    print("\n=== Token Embedding Check ===")
    print("For current_player=0:")
    print(f"  is_current flags: p0=1, p1=0, p2=0, p3=0")
    print(f"  is_partner flags: p0=0, p1=0, p2=1, p3=0")
    print("For current_player=1:")
    print(f"  is_current flags: p0=0, p1=1, p2=0, p3=0")
    print(f"  is_partner flags: p0=0, p1=0, p2=0, p3=1")

    # The model sees is_current=1 for the acting player
    # But does it know if that player is Team 0 or Team 1?
    print("\n=== Key Insight ===")
    print("The model sees is_current=1 for the acting player's hand tokens.")
    print("But it does NOT explicitly see which team that player is on!")
    print("player_id embedding encodes 0,1,2,3 - the model must learn team=player%2")
    print("")
    print("If the model hasn't learned this mapping, it may treat all positions the same.")


if __name__ == "__main__":
    main()
