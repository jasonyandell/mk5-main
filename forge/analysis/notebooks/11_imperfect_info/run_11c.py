#!/usr/bin/env python3
"""
11c: Best Move Stability Analysis (Memory-Efficient Serial Version)

Question: Does optimal move change with opponent hands?
Method: % of positions where argmax(Q) is constant across opponent configs

Processes one base_seed at a time, loading only necessary data.
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from forge.analysis.utils import features
from forge.analysis.utils.seed_db import SeedDB

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"

# Skip shards larger than this
MAX_ROWS = 8_000_000


def get_best_move(q_values: np.ndarray) -> np.ndarray:
    """Get argmax(Q) for each state, handling illegals."""
    ILLEGAL = -128
    q_for_max = np.where(q_values != ILLEGAL, q_values, -1000)
    return np.argmax(q_for_max, axis=1)


def analyze_base_seed(db: SeedDB, base_seed: int) -> dict | None:
    """Analyze one base seed by loading 3 configs sequentially."""
    decl_id = base_seed % 10

    # First: collect states from all 3 configs to find common ones
    all_states = []
    for opp_seed in range(3):
        filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        path = DATA_DIR / filename
        if not path.exists():
            return None

        try:
            # Just read the state column to find common states
            result = db.query_columns(files=[filename], columns=['state'])
            if result.data is None or len(result.data) == 0:
                gc.collect()
                return None
            if len(result.data) > MAX_ROWS:
                del result
                gc.collect()
                return None

            states = set(result.data['state'].values)
            all_states.append(states)
            del result
            gc.collect()

        except Exception as e:
            gc.collect()
            return None

    # Find common states
    common_states = all_states[0] & all_states[1] & all_states[2]
    del all_states
    gc.collect()

    if len(common_states) == 0:
        return None

    # Now reload each config and get best moves only for common states
    state_to_best = [{}, {}, {}]
    state_to_depth = {}

    for opp_seed in range(3):
        filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        result = db.query_columns(files=[filename], columns=['state', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'])

        states = result.data['state'].values
        q_cols = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        q_values = np.column_stack([result.data[c].values for c in q_cols]).astype(np.int16)
        del result
        gc.collect()

        # Filter to common states only
        mask = np.isin(states, np.array(list(common_states)))
        common_indices = np.where(mask)[0]

        if len(common_indices) == 0:
            continue

        common_states_arr = states[common_indices]
        common_q = q_values[common_indices]

        best_moves = get_best_move(common_q)
        state_to_best[opp_seed] = dict(zip(common_states_arr, best_moves))

        if opp_seed == 0:
            depths = features.depth(common_states_arr)
            state_to_depth = dict(zip(common_states_arr, depths))

        del states, q_values, common_states_arr, common_q, best_moves
        gc.collect()

    # Check consistency
    n_consistent = 0
    n_total = 0
    depth_consistency = defaultdict(lambda: {'consistent': 0, 'total': 0})

    for state in common_states:
        best0 = state_to_best[0].get(state)
        best1 = state_to_best[1].get(state)
        best2 = state_to_best[2].get(state)

        if best0 is not None and best1 is not None and best2 is not None:
            n_total += 1
            is_consistent = (best0 == best1 == best2)
            if is_consistent:
                n_consistent += 1

            d = state_to_depth.get(state, -1)
            if d >= 0:
                depth_consistency[d]['total'] += 1
                if is_consistent:
                    depth_consistency[d]['consistent'] += 1

    del state_to_best, common_states, state_to_depth
    gc.collect()

    if n_total == 0:
        return None

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'n_common_states': n_total,
        'n_consistent': n_consistent,
        'consistency_rate': n_consistent / n_total,
        'depth_consistency': dict(depth_consistency)
    }


def main():
    print("=" * 60)
    print("BEST MOVE STABILITY ANALYSIS")
    print("=" * 60)

    # Initialize SeedDB
    db = SeedDB(DATA_DIR)

    # Get base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]
    print(f"Total base seeds: {len(base_seeds)}")

    # Analyze serially
    all_results = []
    depth_agg = defaultdict(lambda: {'consistent': 0, 'total': 0})

    for base_seed in tqdm(base_seeds, desc="Analyzing"):
        result = analyze_base_seed(db, base_seed)
        if result:
            all_results.append({
                'base_seed': result['base_seed'],
                'decl_id': result['decl_id'],
                'n_common_states': result['n_common_states'],
                'n_consistent': result['n_consistent'],
                'consistency_rate': result['consistency_rate']
            })
            for d, counts in result['depth_consistency'].items():
                depth_agg[d]['consistent'] += counts['consistent']
                depth_agg[d]['total'] += counts['total']

    db.close()
    print(f"\n✓ Analyzed {len(all_results)} base seeds with common states")

    if len(all_results) == 0:
        print("No common states found - different opponent configs lead to different game trees")
        return

    df = pd.DataFrame(all_results)

    # Statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    total_states = df['n_common_states'].sum()
    total_consistent = df['n_consistent'].sum()
    overall_rate = total_consistent / total_states if total_states > 0 else 0

    print(f"\nTotal common states analyzed: {total_states:,}")
    print(f"States with consistent best move: {total_consistent:,}")
    print(f"Overall consistency rate: {overall_rate:.1%}")

    # By declaration
    print("\n" + "=" * 60)
    print("CONSISTENCY BY DECLARATION")
    print("=" * 60)

    decl_stats = df.groupby('decl_id').agg({
        'n_common_states': 'sum',
        'n_consistent': 'sum'
    })
    decl_stats['rate'] = decl_stats['n_consistent'] / decl_stats['n_common_states']
    print(decl_stats.round(3).to_string())

    # By depth
    depth_data = []
    for d in sorted(depth_agg.keys()):
        counts = depth_agg[d]
        rate = counts['consistent'] / counts['total'] if counts['total'] > 0 else 0
        depth_data.append({
            'depth': d,
            'n_states': counts['total'],
            'n_consistent': counts['consistent'],
            'consistency_rate': rate
        })
    depth_df = pd.DataFrame(depth_data)

    print("\n" + "=" * 60)
    print("CONSISTENCY BY DEPTH")
    print("=" * 60)
    print(depth_df.to_string(index=False))

    # Save results
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11c_best_move_stability_by_seed.csv", index=False)
    depth_df.to_csv(tables_dir / "11c_stability_by_depth.csv", index=False)

    summary = pd.DataFrame([{
        'metric': 'overall_consistency',
        'value': overall_rate,
        'n_states': total_states,
        'n_consistent': total_consistent
    }])
    summary.to_csv(tables_dir / "11c_stability_summary.csv", index=False)
    print("\n✓ Saved all tables")

    # Visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.bar(depth_df['depth'], depth_df['consistency_rate'], color='steelblue', alpha=0.7)
    ax1.axhline(y=overall_rate, color='red', linestyle='--', label=f'Overall: {overall_rate:.1%}')
    ax1.set_xlabel('Depth (dominoes remaining)')
    ax1.set_ylabel('Best Move Consistency Rate')
    ax1.set_title('Best Move Stability by Game Depth')
    ax1.legend()
    ax1.set_ylim(0, 1)

    ax2 = axes[1]
    ax2.hist(df['consistency_rate'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=overall_rate, color='red', linestyle='--', label=f'Mean: {overall_rate:.1%}')
    ax2.set_xlabel('Consistency Rate')
    ax2.set_ylabel('Number of Base Seeds')
    ax2.set_title('Distribution of Consistency Across Seeds')
    ax2.legend()

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11c_best_move_stability.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. OVERALL STABILITY: {overall_rate:.1%} of positions have consistent best move")
    print(f"   - Opponent hands {'do NOT' if overall_rate > 0.9 else 'DO'} significantly affect optimal play")


if __name__ == "__main__":
    main()
