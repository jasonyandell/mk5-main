#!/usr/bin/env python3
"""11a: Count Lock Rate Analysis - Memory-efficient version.

Uses terminal V values to infer count outcomes rather than tracing PV.
Key insight: Terminal V = 7 (tricks) + count_points. If we know V, we can
infer roughly how many counts each team captured.

Uses SeedDB for efficient root V queries via DuckDB.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup imports
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from forge.analysis.utils import features
from forge.analysis.utils.seed_db import SeedDB
from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
OUTPUT_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"


def main():
    # Create SeedDB for efficient queries
    db = SeedDB(DATA_DIR)

    # Find all base seeds
    shard_files = list(DATA_DIR.glob("seed_*_opp*_decl_*.parquet"))
    base_seeds = sorted(set(int(f.name.split("_")[1]) for f in shard_files))
    print(f"Found {len(base_seeds)} unique base seeds")

    results = []

    for base_seed in tqdm(base_seeds, desc="Analyzing base seeds"):
        decl_id = base_seed % 10
        p0_hand = deal_from_seed(base_seed)[0]

        # Which counts does P0 hold?
        p0_counts = {d for d in features.COUNT_DOMINO_IDS if d in p0_hand}
        p0_count_points = sum(tables.DOMINO_COUNT_POINTS[d] for d in p0_counts)

        row = {
            "base_seed": base_seed,
            "decl_id": decl_id,
            "p0_count_points": p0_count_points,
            "n_counts_held": len(p0_counts),
        }

        # Track count holdings
        for d in features.COUNT_DOMINO_IDS:
            pips = schema.domino_pips(d)
            row[f"holds_{pips[0]}_{pips[1]}"] = d in p0_counts

        # Load each opponent config - just root V via DuckDB
        v_values = []
        for opp_seed in range(3):
            filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
            try:
                result = db.get_root_v(filename)
                v = result.data
                row[f"V_opp{opp_seed}"] = v
                if v is not None:
                    v_values.append(v)
            except Exception:
                row[f"V_opp{opp_seed}"] = None

        if v_values:
            row["V_mean"] = np.mean(v_values)
            row["V_std"] = np.std(v_values)
            row["V_spread"] = max(v_values) - min(v_values)
            row["V_min"] = min(v_values)
            row["V_max"] = max(v_values)

        results.append(row)

    db.close()

    results_df = pd.DataFrame(results)
    print(f"\nAnalyzed {len(results_df)} base seeds")

    # Analysis
    print("\n=== V DISTRIBUTION ANALYSIS ===")
    print(f"Mean V spread: {results_df['V_spread'].mean():.1f} points")
    print(f"Median V spread: {results_df['V_spread'].median():.1f} points")
    print(f"Max V spread: {results_df['V_spread'].max():.0f} points")
    print(f"Hands with spread > 40: {(results_df['V_spread'] > 40).sum()}")
    print(f"Hands with spread < 10: {(results_df['V_spread'] < 10).sum()}")

    # Correlation between count holdings and V
    print("\n=== COUNT HOLDING VS V ===")
    print(f"Correlation(n_counts_held, V_mean): {results_df['n_counts_held'].corr(results_df['V_mean']):.3f}")
    print(f"Correlation(p0_count_points, V_mean): {results_df['p0_count_points'].corr(results_df['V_mean']):.3f}")
    print(f"Correlation(n_counts_held, V_std): {results_df['n_counts_held'].corr(results_df['V_std']):.3f}")

    # Group by count points held
    print("\n=== V BY COUNT POINTS HELD ===")
    grouped = results_df.groupby("p0_count_points")[["V_mean", "V_std", "V_spread"]].agg(
        ["mean", "count"]
    ).round(2)
    print(grouped.to_string())

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "tables/11a_base_seed_analysis.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/tables/11a_base_seed_analysis.csv")


if __name__ == "__main__":
    main()
