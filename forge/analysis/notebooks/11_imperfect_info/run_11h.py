#!/usr/bin/env python3
"""
11h: Path Divergence Analysis

Question: When do paths diverge across opponent configs?
Method: Compare best moves at each depth across 3 opponent configs
What It Reveals: Early divergence = opponent-dependent strategy

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from forge.analysis.utils import features

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Analyze all available base seeds
MAX_ROWS = 8_000_000
np.random.seed(42)


def get_best_move(q_values: np.ndarray) -> int:
    """Get argmax(Q), handling illegals."""
    ILLEGAL = -128
    q_for_max = np.where(q_values != ILLEGAL, q_values, -1000)
    return int(np.argmax(q_for_max))


def get_best_moves_by_depth(path: Path) -> dict | None:
    """Get best moves organized by depth.

    Returns: {depth: {state: best_action}} for depths 5-28 (skip very low depths)
    """
    try:
        pf = pq.ParquetFile(path)

        # Read in batches to handle large files
        result = defaultdict(dict)

        for batch in pf.iter_batches(batch_size=500000, columns=['state', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']):
            states = batch['state'].to_numpy()
            q_cols = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
            q_values = np.column_stack([batch[c].to_numpy() for c in q_cols]).astype(np.int16)
            depths = features.depth(states)

            # Only keep states at depth 5-28 (skip endgame clutter)
            for state, q, d in zip(states, q_values, depths):
                if 5 <= d <= 28:
                    best = get_best_move(q)
                    result[d][state] = best

            del states, q_values, depths

        gc.collect()
        return dict(result) if result else None

    except Exception as e:
        print(f"Error: {e}")
        gc.collect()
        return None


def analyze_divergence_for_base_seed(base_seed: int) -> dict | None:
    """Compare best moves across 3 configs at each depth."""
    decl_id = base_seed % 10

    # Get best moves for each config
    configs_data = []
    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        data = get_best_moves_by_depth(path)
        if data is None:
            return None
        configs_data.append(data)

    if len(configs_data) != 3:
        return None

    # Find common states at each depth and check consistency
    depth_stats = {}
    first_divergence_depth = None

    for depth in range(28, 0, -1):  # Start from root (28) to endgame (1)
        # Get states present in all 3 configs at this depth
        states_0 = set(configs_data[0].get(depth, {}).keys())
        states_1 = set(configs_data[1].get(depth, {}).keys())
        states_2 = set(configs_data[2].get(depth, {}).keys())

        common = states_0 & states_1 & states_2

        if not common:
            depth_stats[depth] = {'n_common': 0, 'n_consistent': 0, 'rate': None}
            continue

        # Check consistency
        n_consistent = 0
        for state in common:
            a0 = configs_data[0][depth][state]
            a1 = configs_data[1][depth][state]
            a2 = configs_data[2][depth][state]
            if a0 == a1 == a2:
                n_consistent += 1

        rate = n_consistent / len(common)
        depth_stats[depth] = {
            'n_common': len(common),
            'n_consistent': n_consistent,
            'rate': rate
        }

        # Track first divergence (rate < 1.0)
        if first_divergence_depth is None and rate < 1.0:
            first_divergence_depth = depth

    # Summary metrics
    early_game_consistency = []  # depths 20-28
    mid_game_consistency = []    # depths 10-19
    late_game_consistency = []   # depths 1-9

    for d, stats in depth_stats.items():
        if stats['rate'] is not None:
            if d >= 20:
                early_game_consistency.append(stats['rate'])
            elif d >= 10:
                mid_game_consistency.append(stats['rate'])
            else:
                late_game_consistency.append(stats['rate'])

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'first_divergence_depth': first_divergence_depth or 0,
        'early_game_rate': np.mean(early_game_consistency) if early_game_consistency else None,
        'mid_game_rate': np.mean(mid_game_consistency) if mid_game_consistency else None,
        'late_game_rate': np.mean(late_game_consistency) if late_game_consistency else None,
        'depth_stats': depth_stats
    }


def main():
    print("=" * 60)
    print("PATH DIVERGENCE ANALYSIS")
    print("=" * 60)

    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    all_results = []
    depth_agg = defaultdict(lambda: {'total_common': 0, 'total_consistent': 0})

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_divergence_for_base_seed(base_seed)
        if result:
            all_results.append({
                'base_seed': result['base_seed'],
                'decl_id': result['decl_id'],
                'first_divergence_depth': result['first_divergence_depth'],
                'early_game_rate': result['early_game_rate'],
                'mid_game_rate': result['mid_game_rate'],
                'late_game_rate': result['late_game_rate']
            })

            # Aggregate depth stats
            for d, stats in result['depth_stats'].items():
                depth_agg[d]['total_common'] += stats['n_common']
                depth_agg[d]['total_consistent'] += stats['n_consistent']

    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    print(f"\nFirst divergence depth: {df['first_divergence_depth'].mean():.1f} dominoes remaining")
    print(f"Early game consistency (d≥20): {df['early_game_rate'].mean()*100:.1f}%")
    print(f"Mid game consistency (d=10-19): {df['mid_game_rate'].mean()*100:.1f}%")
    print(f"Late game consistency (d<10): {df['late_game_rate'].mean()*100:.1f}%")

    # Depth breakdown
    print("\n" + "=" * 60)
    print("CONSISTENCY BY DEPTH")
    print("=" * 60)

    depth_data = []
    for d in sorted(depth_agg.keys(), reverse=True):
        stats = depth_agg[d]
        if stats['total_common'] > 0:
            rate = stats['total_consistent'] / stats['total_common']
            depth_data.append({
                'depth': d,
                'n_common': stats['total_common'],
                'n_consistent': stats['total_consistent'],
                'rate': rate
            })
            print(f"  Depth {d}: {rate*100:.1f}% ({stats['total_consistent']}/{stats['total_common']})")

    depth_df = pd.DataFrame(depth_data)

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11h_path_divergence_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    depth_df.to_csv(tables_dir / "11h_divergence_by_depth.csv", index=False)
    print("✓ Saved depth breakdown")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Consistency by depth
    ax1 = axes[0, 0]
    ax1.bar(depth_df['depth'], depth_df['rate']*100, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Depth (Dominoes Remaining)')
    ax1.set_ylabel('Best Move Consistency (%)')
    ax1.set_title('Path Consistency by Game Phase')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 100)

    # Top right: First divergence distribution
    ax2 = axes[0, 1]
    ax2.hist(df['first_divergence_depth'], bins=range(0, 30), color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('First Divergence Depth')
    ax2.set_ylabel('Frequency')
    ax2.set_title('When Paths First Diverge')

    # Bottom left: Game phase consistency
    ax3 = axes[1, 0]
    phases = ['Early\n(d≥20)', 'Mid\n(d=10-19)', 'Late\n(d<10)']
    rates = [df['early_game_rate'].mean()*100, df['mid_game_rate'].mean()*100, df['late_game_rate'].mean()*100]
    colors = ['red', 'yellow', 'green']
    ax3.bar(phases, rates, color=colors, alpha=0.7)
    ax3.set_ylabel('Consistency (%)')
    ax3.set_title('Path Consistency by Game Phase')
    ax3.set_ylim(0, 100)

    # Bottom right: Depth vs common states
    ax4 = axes[1, 1]
    ax4.scatter(depth_df['depth'], depth_df['n_common'], s=50, alpha=0.7)
    ax4.set_xlabel('Depth')
    ax4.set_ylabel('Number of Common States')
    ax4.set_title('Common States Across Configs')

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11h_path_divergence.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    early = df['early_game_rate'].mean()*100
    late = df['late_game_rate'].mean()*100

    print(f"\n1. EARLY GAME: {early:.0f}% consistency")
    if early < 30:
        print("   → Strategy is highly opponent-dependent from the start")
    elif early < 60:
        print("   → Moderate opponent influence in opening")
    else:
        print("   → Opening play is relatively robust")

    print(f"\n2. LATE GAME: {late:.0f}% consistency")
    print(f"\n3. IMPROVEMENT: {late - early:+.0f}% from early to late game")


if __name__ == "__main__":
    main()
