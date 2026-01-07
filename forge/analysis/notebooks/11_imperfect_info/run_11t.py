#!/usr/bin/env python3
"""
11t: Lock Count → Bid Level Correlation

Question: Does # locked counts predict optimal bid?
Method: Correlate count locks with E[V]
What It Reveals: "Lock 4 counts = bid 42"

Uses marginalized data. A "lock" is when Team 0 captures a count in all 3 configs.
Translates E[V] to bid levels based on expected margin.
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from forge.analysis.utils.seed_db import SeedDB
from forge.oracle import tables
from forge.oracle.rng import deal_from_seed

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"
N_BASE_SEEDS = 201  # Full analysis
np.random.seed(42)

# Count domino IDs
COUNT_DOMINO_IDS = [9, 14, 20, 18, 27]  # 3-2, 4-1, 5-0, 5-5, 6-4
COUNT_NAMES = ['3-2', '4-1', '5-0', '5-5', '6-4']


def ev_to_bid_level(ev: float) -> int:
    """Convert expected value to recommended bid level.

    In Texas 42:
    - 30 is minimum bid (need ~30 point margin to make)
    - 31-34 require slight advantage
    - 35-41 require significant advantage
    - 42 requires strong advantage (~42 point margin)

    E[V] is symmetric around 0, so E[V] = +30 means ~30 point advantage.
    """
    if ev < 25:
        return 0  # Don't bid (pass)
    elif ev < 30:
        return 30  # Minimum bid
    elif ev < 33:
        return 31
    elif ev < 36:
        return 34
    elif ev < 40:
        return 38
    else:
        return 42


def analyze_locks_for_base_seed(db: SeedDB, base_seed: int) -> dict | None:
    """Get lock counts and E[V] for one base seed."""
    decl_id = base_seed % 10

    # Get P0's hand to check which counts are held
    p0_hand = deal_from_seed(base_seed)[0]
    p0_count_holdings = {
        name: (d in p0_hand) for d, name in zip(COUNT_DOMINO_IDS, COUNT_NAMES)
    }

    # Get V values across all 3 opponent configs
    V_values = []
    for opp_seed in range(3):
        filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue
        result = db.get_root_v(filename)
        if result.data is None:
            continue
        V_values.append(float(result.data))

    if len(V_values) != 3:
        return None

    V_mean = np.mean(V_values)
    V_min = min(V_values)
    V_max = max(V_values)
    V_spread = V_max - V_min

    # Determine locks (count captured in all 3 configs)
    # A count is "locked" if V > threshold in all configs (proxy for capture)
    # More precisely, we'd trace PV, but we use a simpler heuristic:
    # If min(V) > 0 AND we hold the count, we likely lock it
    # This is imperfect but gives a rough estimate

    # Alternative: count is locked if we hold it AND spread is small
    # (high consistency suggests control)

    # For this analysis, we'll use a simpler approach:
    # Count "reliable captures" based on holding + V characteristics

    n_counts_held = sum(p0_count_holdings.values())
    total_count_points = sum(
        tables.DOMINO_COUNT_POINTS[d] for d in p0_hand
        if tables.DOMINO_COUNT_POINTS[d] > 0
    )

    # Estimate "reliable" count control:
    # High V_mean + low V_spread + holding counts = good control
    # For simplicity: if we hold a count AND V_min > -10, count it as "likely locked"
    likely_locks = 0
    for d, name in zip(COUNT_DOMINO_IDS, COUNT_NAMES):
        if d in p0_hand and V_min > -10:
            likely_locks += 1

    # Alternative metric: expected tricks won
    # In Texas 42, max tricks = 7, each worth ~6 pts (if no counts)
    # E[V] includes count points, so tricks ≈ E[V] / 6
    expected_tricks = max(0, V_mean / 6)

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'V_mean': V_mean,
        'V_min': V_min,
        'V_max': V_max,
        'V_spread': V_spread,
        'n_counts_held': n_counts_held,
        'total_count_points': total_count_points,
        'likely_locks': likely_locks,
        'expected_tricks': expected_tricks,
        'recommended_bid': ev_to_bid_level(V_mean),
        'conservative_bid': ev_to_bid_level(V_min),  # Based on worst case
        **{f'holds_{name}': p0_count_holdings[name] for name in COUNT_NAMES}
    }


def main():
    print("=" * 60)
    print("LOCK COUNT → BID LEVEL CORRELATION")
    print("=" * 60)

    # Initialize SeedDB
    db = SeedDB(DATA_DIR)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    # Collect data
    all_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_locks_for_base_seed(db, base_seed)
        if result:
            all_results.append(result)

    db.close()
    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("E[V] BY COUNT HOLDINGS")
    print("=" * 60)

    # Group by n_counts_held
    by_counts = df.groupby('n_counts_held').agg({
        'V_mean': ['mean', 'std', 'count'],
        'V_spread': 'mean',
        'recommended_bid': 'mean',
        'conservative_bid': 'mean'
    }).round(2)
    print("\n" + by_counts.to_string())

    # Group by total_count_points
    print("\n" + "=" * 60)
    print("E[V] BY COUNT POINTS HELD")
    print("=" * 60)

    by_points = df.groupby('total_count_points').agg({
        'V_mean': ['mean', 'std', 'count'],
        'V_spread': 'mean',
        'recommended_bid': 'mean'
    }).round(2)
    print("\n" + by_points.to_string())

    # Correlations
    print("\n" + "=" * 60)
    print("CORRELATIONS")
    print("=" * 60)

    corr_targets = ['n_counts_held', 'total_count_points', 'likely_locks']
    for target in corr_targets:
        corr_ev = df[target].corr(df['V_mean'])
        corr_bid = df[target].corr(df['recommended_bid'])
        print(f"{target}:")
        print(f"  → E[V]: {corr_ev:+.3f}")
        print(f"  → Bid level: {corr_bid:+.3f}")

    # Bid level distribution
    print("\n" + "=" * 60)
    print("BID LEVEL RECOMMENDATIONS")
    print("=" * 60)

    bid_dist = df['recommended_bid'].value_counts().sort_index()
    print("\nRecommended bids:")
    for bid, count in bid_dist.items():
        pct = count / len(df) * 100
        label = "Pass" if bid == 0 else str(bid)
        print(f"  {label}: {count} ({pct:.0f}%)")

    cons_bid_dist = df['conservative_bid'].value_counts().sort_index()
    print("\nConservative bids (based on V_min):")
    for bid, count in cons_bid_dist.items():
        pct = count / len(df) * 100
        label = "Pass" if bid == 0 else str(bid)
        print(f"  {label}: {count} ({pct:.0f}%)")

    # Bidding heuristic table
    print("\n" + "=" * 60)
    print("BIDDING HEURISTIC TABLE")
    print("=" * 60)

    # Group by counts held, show E[V] range and recommended bid
    heuristic_data = []
    for n in sorted(df['n_counts_held'].unique()):
        subset = df[df['n_counts_held'] == n]
        heuristic_data.append({
            'counts_held': n,
            'avg_ev': subset['V_mean'].mean(),
            'ev_range': f"[{subset['V_mean'].min():.0f}, {subset['V_mean'].max():.0f}]",
            'avg_bid': subset['recommended_bid'].mean(),
            'n_hands': len(subset)
        })

    heuristic_df = pd.DataFrame(heuristic_data)
    print("\nCounts Held → E[V] → Bid Level")
    print(heuristic_df.to_string(index=False))

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11t_lock_count_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    heuristic_df.to_csv(tables_dir / "11t_bidding_heuristics.csv", index=False)
    print("✓ Saved bidding heuristics")

    # Save correlation summary
    corr_data = [{
        'n_counts_vs_ev': df['n_counts_held'].corr(df['V_mean']),
        'count_points_vs_ev': df['total_count_points'].corr(df['V_mean']),
        'counts_vs_bid': df['n_counts_held'].corr(df['recommended_bid']),
        'n_samples': len(df)
    }]
    pd.DataFrame(corr_data).to_csv(tables_dir / "11t_correlations.csv", index=False)
    print("✓ Saved correlations")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: E[V] by counts held
    ax1 = axes[0, 0]
    ev_by_counts = df.groupby('n_counts_held')['V_mean'].agg(['mean', 'std'])
    ax1.bar(ev_by_counts.index, ev_by_counts['mean'],
            yerr=ev_by_counts['std'], capsize=5, color='steelblue', alpha=0.7)
    ax1.axhline(y=30, color='green', linestyle='--', label='Min bid (30)')
    ax1.axhline(y=42, color='red', linestyle='--', label='Max bid (42)')
    ax1.set_xlabel('Counts Held')
    ax1.set_ylabel('E[V]')
    ax1.set_title('Expected Value by Count Holdings')
    ax1.legend()

    # Top right: E[V] vs count points
    ax2 = axes[0, 1]
    ax2.scatter(df['total_count_points'], df['V_mean'], alpha=0.6, s=50)
    z = np.polyfit(df['total_count_points'], df['V_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 25, 100)
    ax2.plot(x_line, p(x_line), 'r-',
             label=f'r = {df["total_count_points"].corr(df["V_mean"]):.2f}')
    ax2.set_xlabel('Count Points Held')
    ax2.set_ylabel('E[V]')
    ax2.set_title('E[V] vs Count Points')
    ax2.legend()

    # Bottom left: Bid recommendations
    ax3 = axes[1, 0]
    bids = [0, 30, 31, 34, 38, 42]
    bid_labels = ['Pass', '30', '31', '34', '38', '42']
    counts = [len(df[df['recommended_bid'] == b]) for b in bids]
    ax3.bar(bid_labels, counts, color='steelblue', alpha=0.7)
    ax3.set_xlabel('Recommended Bid')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Bid Level Distribution')

    # Bottom right: V_spread by counts held
    ax4 = axes[1, 1]
    spread_by_counts = df.groupby('n_counts_held')['V_spread'].mean()
    ax4.bar(spread_by_counts.index, spread_by_counts.values, color='orange', alpha=0.7)
    ax4.set_xlabel('Counts Held')
    ax4.set_ylabel('Average V Spread')
    ax4.set_title('Risk by Count Holdings')

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11t_lock_count_bid_level.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    corr = df['n_counts_held'].corr(df['V_mean'])
    print(f"\n1. CORRELATION: {corr:+.3f}")
    if corr > 0.3:
        print(f"   → More counts held = higher E[V] (moderate relationship)")
    elif corr > 0:
        print(f"   → Weak positive relationship between counts and E[V]")
    else:
        print(f"   → Count holdings don't strongly predict E[V]")

    print("\n2. BIDDING HEURISTICS:")
    for _, row in heuristic_df.iterrows():
        if row['n_hands'] > 0:
            print(f"   {int(row['counts_held'])} counts → E[V]={row['avg_ev']:.0f} → bid ~{row['avg_bid']:.0f}")

    # Mean bid by counts
    print("\n3. SIMPLIFIED RULE:")
    avg_ev_0 = df[df['n_counts_held'] == 0]['V_mean'].mean() if (df['n_counts_held'] == 0).any() else 0
    avg_ev_3 = df[df['n_counts_held'] >= 3]['V_mean'].mean() if (df['n_counts_held'] >= 3).any() else 0
    print(f"   0 counts: E[V] ≈ {avg_ev_0:.0f}")
    print(f"   3+ counts: E[V] ≈ {avg_ev_3:.0f}")
    print(f"   Each count adds ~{(avg_ev_3 - avg_ev_0)/3:.0f} expected points (rough)")


if __name__ == "__main__":
    main()
