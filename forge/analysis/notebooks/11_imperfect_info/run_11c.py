#!/usr/bin/env python3
"""
11c: Best Move Stability Analysis

Question: Does optimal move change with opponent hands?
Method: % of positions where argmax(Q) is constant across opponent configs

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from forge.analysis.utils import loading, features
from forge.oracle import schema

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Analyze all available base seeds
MAX_STATES_PER_SHARD = 1_000_000  # Sample if larger (increased to get more common states)
np.random.seed(42)  # For reproducibility


def get_best_move(q_values: np.ndarray) -> np.ndarray:
    """Get argmax(Q) for each state, handling illegals."""
    ILLEGAL = -128
    # Replace illegal with very negative for argmax
    q_for_max = np.where(q_values != ILLEGAL, q_values, -1000)
    return np.argmax(q_for_max, axis=1)


def analyze_stability_for_base_seed(base_seed: int) -> dict | None:
    """Analyze best move stability across 3 opponent configs for one base seed.

    Memory-optimized: load one shard at a time, build minimal mappings.
    """
    decl_id = base_seed % 10

    # First pass: get states and best moves for each config, one at a time
    state_to_best = [{}, {}, {}]
    states_set = [set(), set(), set()]

    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None
        try:
            df, _, _ = schema.load_file(path)
            # Skip very large files
            if len(df) > 20_000_000:
                del df
                gc.collect()
                return None

            # Extract only what we need, with sampling for large shards
            n_rows = len(df)
            if n_rows > MAX_STATES_PER_SHARD:
                # Sample randomly
                indices = np.random.choice(n_rows, MAX_STATES_PER_SHARD, replace=False)
                states = df['state'].values[indices]
                q_cols = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
                q_values = df[q_cols].values[indices].astype(np.int16)
            else:
                states = df['state'].values
                q_cols = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
                q_values = df[q_cols].values.astype(np.int16)

            best_moves = get_best_move(q_values)
            state_to_best[opp_seed] = dict(zip(states, best_moves))
            states_set[opp_seed] = set(states)

            # Also save depths from first config
            if opp_seed == 0:
                depths = features.depth(states)
                state_to_depth = dict(zip(states, depths))

            del df, states, q_values, best_moves
            gc.collect()

        except Exception as e:
            print(f"Error loading {path}: {e}")
            gc.collect()
            return None

    # Find common states
    common_states = states_set[0] & states_set[1] & states_set[2]

    # Free memory
    del states_set
    gc.collect()

    if len(common_states) == 0:
        return None

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

            # Track by depth
            d = state_to_depth.get(state, -1)
            if d >= 0:
                depth_consistency[d]['total'] += 1
                if is_consistent:
                    depth_consistency[d]['consistent'] += 1

    # Cleanup
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

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = []
    for f in files:
        parts = f.stem.split('_')
        base_seed = int(parts[1])
        base_seeds.append(base_seed)

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    # Analyze each base seed
    all_results = []
    depth_agg = defaultdict(lambda: {'consistent': 0, 'total': 0})

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_stability_for_base_seed(base_seed)
        if result:
            all_results.append({
                'base_seed': result['base_seed'],
                'decl_id': result['decl_id'],
                'n_common_states': result['n_common_states'],
                'n_consistent': result['n_consistent'],
                'consistency_rate': result['consistency_rate']
            })

            # Aggregate depth consistency
            for d, counts in result['depth_consistency'].items():
                depth_agg[d]['consistent'] += counts['consistent']
                depth_agg[d]['total'] += counts['total']

    print(f"\n✓ Analyzed {len(all_results)} base seeds")

    # Build DataFrames
    df = pd.DataFrame(all_results)

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
    decl_stats = decl_stats.round(3)
    print(decl_stats.to_string())

    # By depth
    print("\n" + "=" * 60)
    print("CONSISTENCY BY DEPTH")
    print("=" * 60)

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
    print("\nConsistency by depth (dominoes remaining):")
    print(depth_df.to_string(index=False))

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Save overall results
    df.to_csv(tables_dir / "11c_best_move_stability_by_seed.csv", index=False)
    print("✓ Saved per-seed results")

    # Save depth breakdown
    depth_df.to_csv(tables_dir / "11c_stability_by_depth.csv", index=False)
    print("✓ Saved depth breakdown")

    # Save summary
    summary = pd.DataFrame([{
        'metric': 'overall_consistency',
        'value': overall_rate,
        'n_states': total_states,
        'n_consistent': total_consistent
    }])
    summary.to_csv(tables_dir / "11c_stability_summary.csv", index=False)
    print("✓ Saved summary")

    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Consistency by depth
    ax1 = axes[0]
    ax1.bar(depth_df['depth'], depth_df['consistency_rate'], color='steelblue', alpha=0.7)
    ax1.axhline(y=overall_rate, color='red', linestyle='--', label=f'Overall: {overall_rate:.1%}')
    ax1.set_xlabel('Depth (dominoes remaining)')
    ax1.set_ylabel('Best Move Consistency Rate')
    ax1.set_title('Best Move Stability by Game Depth')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Right: Histogram of per-seed consistency
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
    print(f"   - Opponent hands do {'NOT ' if overall_rate > 0.9 else ''}significantly affect optimal play")

    # Early vs late game
    early_depths = depth_df[depth_df['depth'] >= 21]
    late_depths = depth_df[depth_df['depth'] <= 7]

    if len(early_depths) > 0 and len(late_depths) > 0:
        early_rate = early_depths['n_consistent'].sum() / early_depths['n_states'].sum()
        late_rate = late_depths['n_consistent'].sum() / late_depths['n_states'].sum()
        print(f"\n2. EARLY GAME (depth 21-28): {early_rate:.1%} consistency")
        print(f"   LATE GAME (depth 1-7): {late_rate:.1%} consistency")
        if early_rate < late_rate:
            print("   → Hidden info matters more early in the hand")
        else:
            print("   → Hidden info impact is uniform across game stages")

    # Most/least stable declarations
    if len(decl_stats) > 1:
        best_decl = decl_stats['rate'].idxmax()
        worst_decl = decl_stats['rate'].idxmin()
        print(f"\n3. DECLARATION EFFECT:")
        print(f"   Most stable: decl {best_decl} ({decl_stats.loc[best_decl, 'rate']:.1%})")
        print(f"   Least stable: decl {worst_decl} ({decl_stats.loc[worst_decl, 'rate']:.1%})")


if __name__ == "__main__":
    main()
