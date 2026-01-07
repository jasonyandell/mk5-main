#!/usr/bin/env python3
"""
11i: Basin Convergence Analysis

Question: Do different opponent configs reach same basin (outcome)?
Method: % of configs sharing modal basin (V category)
What It Reveals: Hand dominance vs luck

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
A "basin" is defined as V falling into one of several outcome categories.
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

from forge.analysis.utils import features

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Full analysis
np.random.seed(42)

# Define V basins (outcome categories)
# V ranges from -42 to +42
BASIN_THRESHOLDS = [-20, -5, 5, 20]  # Creates 5 basins
BASIN_NAMES = ['Big Loss (<-20)', 'Loss (-20 to -5)', 'Draw (-5 to 5)',
               'Win (5 to 20)', 'Big Win (>20)']


def get_basin(v: float) -> int:
    """Map V value to basin index."""
    for i, threshold in enumerate(BASIN_THRESHOLDS):
        if v < threshold:
            return i
    return len(BASIN_THRESHOLDS)


def get_root_v_fast(path: Path) -> float | None:
    """Get root state V value without loading entire shard."""
    try:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=10000, columns=['state', 'V']):
            states = batch['state'].to_numpy()
            V = batch['V'].to_numpy()
            depths = features.depth(states)
            root_mask = depths == 28
            if root_mask.any():
                return float(V[root_mask][0])
        return None
    except Exception:
        return None


def analyze_basin_convergence_for_base_seed(base_seed: int) -> dict | None:
    """Analyze basin convergence across 3 opponent configs."""
    decl_id = base_seed % 10

    # Get V values for each config
    V_values = []
    basins = []

    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        V = get_root_v_fast(path)
        if V is None:
            continue

        V_values.append(V)
        basins.append(get_basin(V))

    if len(V_values) != 3:
        return None

    # Basin convergence metrics
    V_mean = np.mean(V_values)
    V_std = np.std(V_values)
    V_spread = max(V_values) - min(V_values)

    # Do all configs land in same basin?
    all_same_basin = (basins[0] == basins[1] == basins[2])

    # Modal basin (most common)
    from collections import Counter
    basin_counts = Counter(basins)
    modal_basin = basin_counts.most_common(1)[0][0]
    modal_basin_count = basin_counts.most_common(1)[0][1]

    # Basin spread (range of basin indices)
    basin_spread = max(basins) - min(basins)

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'V_mean': V_mean,
        'V_std': V_std,
        'V_spread': V_spread,
        'basin_0': basins[0],
        'basin_1': basins[1],
        'basin_2': basins[2],
        'all_same_basin': int(all_same_basin),
        'modal_basin': modal_basin,
        'modal_basin_count': modal_basin_count,
        'basin_spread': basin_spread
    }


def main():
    print("=" * 60)
    print("BASIN CONVERGENCE ANALYSIS")
    print("=" * 60)

    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    all_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_basin_convergence_for_base_seed(base_seed)
        if result:
            all_results.append(result)

    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    # Basin convergence rate
    convergence_rate = df['all_same_basin'].mean()
    print(f"\nBasin convergence rate: {convergence_rate*100:.1f}%")
    print(f"  (% of hands where all 3 configs land in same outcome basin)")

    # V spread statistics
    print(f"\nV spread statistics:")
    print(f"  Mean spread: {df['V_spread'].mean():.1f} points")
    print(f"  Median spread: {df['V_spread'].median():.1f} points")
    print(f"  Spread < 10: {(df['V_spread'] < 10).sum()} ({(df['V_spread'] < 10).mean()*100:.0f}%)")
    print(f"  Spread > 40: {(df['V_spread'] > 40).sum()} ({(df['V_spread'] > 40).mean()*100:.0f}%)")

    # Basin spread distribution
    print(f"\nBasin spread (# of basins crossed):")
    for spread in sorted(df['basin_spread'].unique()):
        count = (df['basin_spread'] == spread).sum()
        pct = count / len(df) * 100
        print(f"  Spread {spread}: {count} ({pct:.0f}%)")

    # Dominance classification
    low_variance = df['V_spread'] < 15
    high_variance = df['V_spread'] > 35

    print(f"\nHand dominance classification:")
    print(f"  Dominant (spread < 15): {low_variance.sum()} ({low_variance.mean()*100:.0f}%)")
    print(f"  Luck-dependent (spread > 35): {high_variance.sum()} ({high_variance.mean()*100:.0f}%)")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11i_basin_convergence_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    summary = pd.DataFrame([{
        'convergence_rate': convergence_rate,
        'mean_v_spread': df['V_spread'].mean(),
        'median_v_spread': df['V_spread'].median(),
        'pct_dominant': low_variance.mean(),
        'pct_luck_dependent': high_variance.mean(),
        'n_samples': len(df)
    }])
    summary.to_csv(tables_dir / "11i_basin_convergence_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: V spread distribution
    ax1 = axes[0, 0]
    ax1.hist(df['V_spread'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=15, color='green', linestyle='--', label='Dominant threshold')
    ax1.axvline(x=35, color='red', linestyle='--', label='Luck threshold')
    ax1.set_xlabel('V Spread (max - min across configs)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Outcome Variance Distribution')
    ax1.legend()

    # Top right: Basin convergence
    ax2 = axes[0, 1]
    labels = ['Converged\n(same basin)', 'Diverged\n(different basins)']
    sizes = [df['all_same_basin'].sum(), len(df) - df['all_same_basin'].sum()]
    colors = ['green', 'red']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax2.set_title('Basin Convergence')

    # Bottom left: Basin spread distribution
    ax3 = axes[1, 0]
    spread_counts = df['basin_spread'].value_counts().sort_index()
    ax3.bar(spread_counts.index, spread_counts.values, color='steelblue', alpha=0.7)
    ax3.set_xlabel('Basin Spread (# of basins crossed)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('How Far Do Outcomes Diverge?')
    ax3.set_xticks(range(5))

    # Bottom right: V mean vs V spread
    ax4 = axes[1, 1]
    colors = ['green' if v < 15 else 'red' if v > 35 else 'gray' for v in df['V_spread']]
    ax4.scatter(df['V_mean'], df['V_spread'], c=colors, alpha=0.6, s=50)
    ax4.set_xlabel('Mean V (expected outcome)')
    ax4.set_ylabel('V Spread (outcome variance)')
    ax4.set_title('Expected Value vs Variance')
    ax4.axhline(y=15, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(y=35, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11i_basin_convergence.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. BASIN CONVERGENCE: {convergence_rate*100:.0f}% of hands reach same basin across configs")

    print(f"\n2. HAND DOMINANCE:")
    if low_variance.mean() > 0.3:
        print(f"   {low_variance.mean()*100:.0f}% of hands are 'dominant' - outcome mostly determined by P0's hand")
    else:
        print(f"   Only {low_variance.mean()*100:.0f}% of hands are 'dominant' - luck plays major role")

    print(f"\n3. INTERPRETATION:")
    if convergence_rate > 0.5:
        print("   → Outcomes are moderately predictable from P0's hand")
    else:
        print("   → Outcomes are highly opponent-dependent (high luck factor)")


if __name__ == "__main__":
    main()
