#!/usr/bin/env python3
"""
11j: Basin Variance Analysis

Question: How many outcome basins are reachable from a hand?
Method: Fix hand, count unique terminal basins across opponent configs
What It Reveals: High variance = risky bid, low variance = safe bid

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
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
from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Full analysis
np.random.seed(42)

# Basin definitions
BASINS = [
    ('Big Loss', -42, -20),
    ('Loss', -20, -5),
    ('Draw', -5, 5),
    ('Win', 5, 20),
    ('Big Win', 20, 43),
]

def get_basin(v: float) -> str:
    """Get basin name for a V value."""
    for name, lo, hi in BASINS:
        if lo <= v < hi:
            return name
    return 'Unknown'


def get_basin_idx(v: float) -> int:
    """Get basin index (0-4) for a V value."""
    for i, (_, lo, hi) in enumerate(BASINS):
        if lo <= v < hi:
            return i
    return -1


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


def extract_hand_features(hand: list[int], trump_suit: int) -> dict:
    """Extract features from a hand."""
    n_doubles = sum(1 for d in hand if schema.domino_pips(d)[0] == schema.domino_pips(d)[1])
    count_points = sum(tables.DOMINO_COUNT_POINTS[d] for d in hand if tables.DOMINO_COUNT_POINTS[d] > 0)
    trump_count = sum(1 for d in hand if trump_suit in schema.domino_pips(d))
    has_trump_double = any(
        schema.domino_pips(d) == (trump_suit, trump_suit) for d in hand
    )
    n_6_high = sum(1 for d in hand if schema.domino_pips(d)[0] == 6)
    total_pips = sum(sum(schema.domino_pips(d)) for d in hand)

    return {
        'n_doubles': n_doubles,
        'count_points': count_points,
        'trump_count': trump_count,
        'has_trump_double': has_trump_double,
        'n_6_high': n_6_high,
        'total_pips': total_pips
    }


def hand_to_str(hand: list[int]) -> str:
    """Convert hand to readable string."""
    dominoes = []
    for d in sorted(hand, reverse=True):
        high, low = schema.domino_pips(d)
        dominoes.append(f"{high}-{low}")
    return " ".join(dominoes)


def analyze_for_base_seed(base_seed: int) -> dict | None:
    """Analyze basin variance for one base seed across opponent configs."""
    decl_id = base_seed % 10
    trump_suit = decl_id
    p0_hand = deal_from_seed(base_seed)[0]

    # Get V values across all 3 opponent configs
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

    V_mean = np.mean(V_values)
    V_std = np.std(V_values)
    V_min = min(V_values)
    V_max = max(V_values)
    V_spread = V_max - V_min

    # Basin metrics
    unique_basins = set(basins)
    n_unique_basins = len(unique_basins)
    basin_converged = n_unique_basins == 1

    # Basin indices for correlation
    basin_indices = [get_basin_idx(v) for v in V_values]
    basin_spread = max(basin_indices) - min(basin_indices)

    # Hand features
    feats = extract_hand_features(p0_hand, trump_suit)

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'hand_str': hand_to_str(p0_hand),
        'V_mean': V_mean,
        'V_std': V_std,
        'V_min': V_min,
        'V_max': V_max,
        'V_spread': V_spread,
        'n_unique_basins': n_unique_basins,
        'basin_converged': basin_converged,
        'basin_spread': basin_spread,
        'basins': ','.join(basins),
        **feats
    }


def main():
    print("=" * 60)
    print("BASIN VARIANCE ANALYSIS")
    print("=" * 60)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    # Collect data
    all_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_for_base_seed(base_seed)
        if result:
            all_results.append(result)

    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("BASIN CONVERGENCE")
    print("=" * 60)

    converged_pct = df['basin_converged'].sum() / len(df) * 100
    print(f"\n  Hands that converge to same basin: {df['basin_converged'].sum()} / {len(df)} ({converged_pct:.1f}%)")

    print("\n  Distribution of unique basins reached:")
    for n in [1, 2, 3]:
        count = (df['n_unique_basins'] == n).sum()
        pct = count / len(df) * 100
        print(f"    {n} basins: {count} ({pct:.1f}%)")

    # Basin spread
    print("\n" + "=" * 60)
    print("BASIN SPREAD ANALYSIS")
    print("=" * 60)

    print("\n  Mean basin spread: {:.2f}".format(df['basin_spread'].mean()))
    print("  Median basin spread: {:.1f}".format(df['basin_spread'].median()))

    print("\n  Basin spread distribution:")
    for spread in range(5):
        count = (df['basin_spread'] == spread).sum()
        pct = count / len(df) * 100
        print(f"    Spread {spread}: {count} ({pct:.1f}%)")

    # High variance vs low variance hands
    print("\n" + "=" * 60)
    print("HIGH VS LOW VARIANCE HANDS")
    print("=" * 60)

    low_var = df[df['basin_converged']]
    high_var = df[~df['basin_converged']]

    print(f"\n  Low variance (converged, n={len(low_var)}):")
    if len(low_var) > 0:
        print(f"    Mean E[V]: {low_var['V_mean'].mean():+.1f}")
        print(f"    Mean doubles: {low_var['n_doubles'].mean():.1f}")
        print(f"    Mean trump count: {low_var['trump_count'].mean():.1f}")
        print(f"    % with trump double: {low_var['has_trump_double'].mean()*100:.0f}%")

    print(f"\n  High variance (diverged, n={len(high_var)}):")
    if len(high_var) > 0:
        print(f"    Mean E[V]: {high_var['V_mean'].mean():+.1f}")
        print(f"    Mean doubles: {high_var['n_doubles'].mean():.1f}")
        print(f"    Mean trump count: {high_var['trump_count'].mean():.1f}")
        print(f"    % with trump double: {high_var['has_trump_double'].mean()*100:.0f}%")

    # Correlations with basin spread
    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS WITH BASIN SPREAD")
    print("=" * 60)

    for col in ['n_doubles', 'trump_count', 'count_points', 'n_6_high', 'total_pips', 'V_mean', 'V_std']:
        corr = df[col].corr(df['basin_spread'])
        print(f"  {col}: {corr:+.3f}")

    # Top low-variance hands
    print("\n" + "=" * 60)
    print("TOP 10 LOWEST VARIANCE HANDS (Safest Bids)")
    print("=" * 60)

    lowest_var = df.nsmallest(10, 'V_spread')
    for i, (_, row) in enumerate(lowest_var.iterrows(), 1):
        print(f"{i:2d}. E[V]={row['V_mean']:+5.1f}, Spread={row['V_spread']:.0f}, "
              f"Basins: {row['basins']}")
        print(f"    {row['hand_str']}")

    # Top high-variance hands
    print("\n" + "=" * 60)
    print("TOP 10 HIGHEST VARIANCE HANDS (Riskiest Bids)")
    print("=" * 60)

    highest_var = df.nlargest(10, 'V_spread')
    for i, (_, row) in enumerate(highest_var.iterrows(), 1):
        print(f"{i:2d}. E[V]={row['V_mean']:+5.1f}, Spread={row['V_spread']:.0f}, "
              f"Basins: {row['basins']}")
        print(f"    {row['hand_str']}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11j_basin_variance_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Summary table
    summary = pd.DataFrame([{
        'n_hands': len(df),
        'converged_pct': converged_pct,
        'mean_unique_basins': df['n_unique_basins'].mean(),
        'mean_basin_spread': df['basin_spread'].mean(),
        'mean_v_spread': df['V_spread'].mean(),
        'low_var_mean_ev': low_var['V_mean'].mean() if len(low_var) > 0 else 0,
        'high_var_mean_ev': high_var['V_mean'].mean() if len(high_var) > 0 else 0,
    }])
    summary.to_csv(tables_dir / "11j_basin_variance_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Basin convergence pie chart
    ax1 = axes[0, 0]
    basin_counts = df['n_unique_basins'].value_counts().sort_index()
    ax1.pie(basin_counts.values,
            labels=[f'{n} basin{"s" if n > 1 else ""}' for n in basin_counts.index],
            autopct='%1.1f%%',
            startangle=90)
    ax1.set_title('Hands by Number of Basins Reached')

    # Top right: E[V] vs V spread
    ax2 = axes[0, 1]
    colors = df['n_unique_basins'].map({1: 'green', 2: 'orange', 3: 'red'})
    ax2.scatter(df['V_mean'], df['V_spread'], c=colors, alpha=0.6, s=40)
    ax2.set_xlabel('E[V]')
    ax2.set_ylabel('V Spread (max - min)')
    ax2.set_title('Expected Value vs Outcome Spread')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='1 basin (converged)'),
        Patch(facecolor='orange', label='2 basins'),
        Patch(facecolor='red', label='3 basins')
    ]
    ax2.legend(handles=legend_elements)

    # Bottom left: Doubles vs basin spread
    ax3 = axes[1, 0]
    for n_dbl in sorted(df['n_doubles'].unique()):
        subset = df[df['n_doubles'] == n_dbl]
        ax3.scatter([n_dbl]*len(subset), subset['basin_spread'],
                   alpha=0.3, s=30, color='blue')
    ax3.set_xlabel('Number of Doubles')
    ax3.set_ylabel('Basin Spread')
    ax3.set_title('Doubles vs Outcome Variance')

    # Add mean line
    means = df.groupby('n_doubles')['basin_spread'].mean()
    ax3.plot(means.index, means.values, 'ro-', linewidth=2, markersize=8,
            label=f'Mean (r={df["n_doubles"].corr(df["basin_spread"]):.2f})')
    ax3.legend()

    # Bottom right: Basin spread histogram
    ax4 = axes[1, 1]
    ax4.hist(df['basin_spread'], bins=range(6), edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Basin Spread')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Basin Spread')
    ax4.set_xticks(range(5))

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11j_basin_variance.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. BASIN CONVERGENCE:")
    print(f"   Only {converged_pct:.1f}% of hands converge to same basin")
    print(f"   → Most hands ({100-converged_pct:.1f}%) cross multiple outcome categories")

    ev_spread_corr = df['V_mean'].corr(df['V_spread'])
    print(f"\n2. E[V] VS SPREAD:")
    print(f"   Correlation: {ev_spread_corr:+.3f}")
    if ev_spread_corr < -0.2:
        print(f"   → Strong hands have LOWER variance (safer bids)")
    elif ev_spread_corr > 0.2:
        print(f"   → Strong hands have HIGHER variance (riskier bids)")
    else:
        print(f"   → E[V] and spread are largely independent")

    doubles_spread_corr = df['n_doubles'].corr(df['basin_spread'])
    print(f"\n3. DOUBLES AND RISK:")
    print(f"   Doubles vs basin spread correlation: {doubles_spread_corr:+.3f}")
    if doubles_spread_corr < -0.2:
        print(f"   → More doubles = LOWER risk (safer bids)")


if __name__ == "__main__":
    main()
