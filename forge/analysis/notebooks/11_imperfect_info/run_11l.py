#!/usr/bin/env python3
"""
11l: Lock Rate by Count Value

Question: Are 10-counts easier to lock than 5-counts?
Method: Compare lock rates by count value (5 vs 10 points)
What It Reveals: Which counts matter for bidding

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

DATA_DIR = Path("data/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Full analysis
np.random.seed(42)

# Count dominoes: (id, name, points)
FIVE_POINT_COUNTS = [
    (9, '3-2', 5),   # domino ID 9 = 3-2
    (14, '4-1', 5),  # domino ID 14 = 4-1
    (20, '5-0', 5),  # domino ID 20 = 5-0
]
TEN_POINT_COUNTS = [
    (18, '5-5', 10),  # domino ID 18 = 5-5
    (27, '6-4', 10),  # domino ID 27 = 6-4
]
ALL_COUNTS = FIVE_POINT_COUNTS + TEN_POINT_COUNTS


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


def get_team_ownership(hands: list[list[int]]) -> dict[int, int]:
    """Get team ownership for each count domino.

    Returns: {domino_id: team} where team is 0 (P0+P2) or 1 (P1+P3)
    """
    ownership = {}
    for d_id, _, _ in ALL_COUNTS:
        for player, hand in enumerate(hands):
            if d_id in hand:
                team = 0 if player in [0, 2] else 1
                ownership[d_id] = team
                break
    return ownership


def analyze_lock_rates_for_base_seed(base_seed: int) -> dict | None:
    """Analyze lock rates for counts in one base seed."""
    decl_id = base_seed % 10
    p0_hand = deal_from_seed(base_seed)[0]

    # Collect V and ownership for each opponent config
    config_data = []
    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        V = get_root_v_fast(path)
        if V is None:
            return None

        # Reconstruct full deal for this config
        from forge.oracle.rng import deal_with_fixed_p0
        hands = deal_with_fixed_p0(p0_hand, opp_seed)
        ownership = get_team_ownership(hands)

        config_data.append({'V': V, 'ownership': ownership})

    if len(config_data) != 3:
        return None

    # Analyze each count
    count_results = {}
    for d_id, name, points in ALL_COUNTS:
        # Check if Team 0 owns this count in each config
        team0_owns = [cfg['ownership'].get(d_id, -1) == 0 for cfg in config_data]

        # "Lock" = Team 0 owns in all 3 configs
        locked = all(team0_owns)
        # "Capture rate" = fraction of configs where Team 0 owns
        capture_rate = sum(team0_owns) / 3

        count_results[name] = {
            'owned_count': sum(team0_owns),
            'locked': locked,
            'capture_rate': capture_rate,
            'points': points
        }

    # V statistics
    V_values = [cfg['V'] for cfg in config_data]
    V_mean = np.mean(V_values)
    V_spread = max(V_values) - min(V_values)

    # Aggregate by point value
    five_pt_locks = sum(1 for d, n, p in FIVE_POINT_COUNTS if count_results[n]['locked'])
    ten_pt_locks = sum(1 for d, n, p in TEN_POINT_COUNTS if count_results[n]['locked'])
    five_pt_rate = np.mean([count_results[n]['capture_rate'] for d, n, p in FIVE_POINT_COUNTS])
    ten_pt_rate = np.mean([count_results[n]['capture_rate'] for d, n, p in TEN_POINT_COUNTS])

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'V_mean': V_mean,
        'V_spread': V_spread,
        # Per-count data
        **{f'{name}_locked': count_results[name]['locked'] for d, name, p in ALL_COUNTS},
        **{f'{name}_rate': count_results[name]['capture_rate'] for d, name, p in ALL_COUNTS},
        # Aggregates
        'five_pt_locks': five_pt_locks,
        'ten_pt_locks': ten_pt_locks,
        'five_pt_rate': five_pt_rate,
        'ten_pt_rate': ten_pt_rate,
        'total_locks': five_pt_locks + ten_pt_locks
    }


def main():
    print("=" * 60)
    print("LOCK RATE BY COUNT VALUE")
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
        result = analyze_lock_rates_for_base_seed(base_seed)
        if result:
            all_results.append(result)

    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("LOCK RATES BY COUNT")
    print("=" * 60)

    # Per-count lock rates
    lock_data = []
    for d_id, name, points in ALL_COUNTS:
        lock_rate = df[f'{name}_locked'].mean()
        capture_rate = df[f'{name}_rate'].mean()
        lock_data.append({
            'count': name,
            'points': points,
            'lock_rate': lock_rate,
            'capture_rate': capture_rate,
            'n': len(df)
        })
        print(f"  {name} ({points}pts): Lock={lock_rate:.1%}, Capture={capture_rate:.1%}")

    lock_df = pd.DataFrame(lock_data)

    print("\n" + "=" * 60)
    print("5-POINT vs 10-POINT COUNTS")
    print("=" * 60)

    five_pt_avg_lock = df['five_pt_locks'].mean() / 3  # Normalize to rate
    ten_pt_avg_lock = df['ten_pt_locks'].mean() / 2    # Normalize to rate
    five_pt_avg_capture = df['five_pt_rate'].mean()
    ten_pt_avg_capture = df['ten_pt_rate'].mean()

    print(f"\n5-point counts (3-2, 4-1, 5-0):")
    print(f"  Avg lock rate: {five_pt_avg_lock:.1%}")
    print(f"  Avg capture rate: {five_pt_avg_capture:.1%}")

    print(f"\n10-point counts (5-5, 6-4):")
    print(f"  Avg lock rate: {ten_pt_avg_lock:.1%}")
    print(f"  Avg capture rate: {ten_pt_avg_capture:.1%}")

    diff_lock = ten_pt_avg_lock - five_pt_avg_lock
    diff_capture = ten_pt_avg_capture - five_pt_avg_capture
    print(f"\nDifference (10pt - 5pt):")
    print(f"  Lock rate: {diff_lock:+.1%}")
    print(f"  Capture rate: {diff_capture:+.1%}")

    # Correlation with V
    print("\n" + "=" * 60)
    print("LOCK RATES vs E[V]")
    print("=" * 60)

    corr_total = df['total_locks'].corr(df['V_mean'])
    corr_five = df['five_pt_locks'].corr(df['V_mean'])
    corr_ten = df['ten_pt_locks'].corr(df['V_mean'])

    print(f"\n  total_locks vs E[V]: {corr_total:+.3f}")
    print(f"  five_pt_locks vs E[V]: {corr_five:+.3f}")
    print(f"  ten_pt_locks vs E[V]: {corr_ten:+.3f}")

    # E[V] by lock count
    print("\n" + "=" * 60)
    print("E[V] BY TOTAL LOCKS")
    print("=" * 60)

    by_locks = df.groupby('total_locks').agg({
        'V_mean': ['mean', 'std', 'count'],
        'V_spread': 'mean'
    }).round(2)
    print("\n" + by_locks.to_string())

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11l_lock_by_count_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    lock_df.to_csv(tables_dir / "11l_lock_rates_summary.csv", index=False)
    print("✓ Saved lock rate summary")

    # Summary comparison
    summary = pd.DataFrame([{
        'five_pt_lock_rate': five_pt_avg_lock,
        'ten_pt_lock_rate': ten_pt_avg_lock,
        'five_pt_capture_rate': five_pt_avg_capture,
        'ten_pt_capture_rate': ten_pt_avg_capture,
        'lock_diff': diff_lock,
        'capture_diff': diff_capture,
        'total_locks_vs_ev_corr': corr_total,
        'n_samples': len(df)
    }])
    summary.to_csv(tables_dir / "11l_five_vs_ten_summary.csv", index=False)
    print("✓ Saved 5pt vs 10pt comparison")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Lock rates by count
    ax1 = axes[0, 0]
    colors = ['blue'] * 3 + ['orange'] * 2  # 5pt blue, 10pt orange
    ax1.bar(lock_df['count'], lock_df['lock_rate'] * 100, color=colors, alpha=0.7)
    ax1.set_xlabel('Count Domino')
    ax1.set_ylabel('Lock Rate (%)')
    ax1.set_title('Lock Rate by Count (Blue=5pt, Orange=10pt)')
    ax1.axhline(y=np.mean(lock_df['lock_rate']) * 100, color='red', linestyle='--',
                label=f'Mean: {np.mean(lock_df["lock_rate"])*100:.1f}%')
    ax1.legend()

    # Top right: Capture rates by count
    ax2 = axes[0, 1]
    ax2.bar(lock_df['count'], lock_df['capture_rate'] * 100, color=colors, alpha=0.7)
    ax2.set_xlabel('Count Domino')
    ax2.set_ylabel('Capture Rate (%)')
    ax2.set_title('Capture Rate by Count')

    # Bottom left: 5pt vs 10pt comparison
    ax3 = axes[1, 0]
    categories = ['Lock Rate', 'Capture Rate']
    five_pt_vals = [five_pt_avg_lock * 100, five_pt_avg_capture * 100]
    ten_pt_vals = [ten_pt_avg_lock * 100, ten_pt_avg_capture * 100]
    x = np.arange(len(categories))
    width = 0.35
    ax3.bar(x - width/2, five_pt_vals, width, label='5-point', color='blue', alpha=0.7)
    ax3.bar(x + width/2, ten_pt_vals, width, label='10-point', color='orange', alpha=0.7)
    ax3.set_ylabel('Rate (%)')
    ax3.set_title('5-Point vs 10-Point Counts')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()

    # Bottom right: E[V] by total locks
    ax4 = axes[1, 1]
    ev_by_locks = df.groupby('total_locks')['V_mean'].mean()
    ax4.bar(ev_by_locks.index, ev_by_locks.values, color='steelblue', alpha=0.7)
    ax4.set_xlabel('Total Locked Counts')
    ax4.set_ylabel('E[V]')
    ax4.set_title(f'E[V] by Lock Count (r = {corr_total:.2f})')

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11l_lock_by_count_value.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. LOCK RATE COMPARISON:")
    print(f"   5-point counts: {five_pt_avg_lock:.1%}")
    print(f"   10-point counts: {ten_pt_avg_lock:.1%}")
    if diff_lock > 0.05:
        print(f"   → 10-point counts are EASIER to lock (+{diff_lock:.1%})")
    elif diff_lock < -0.05:
        print(f"   → 5-point counts are EASIER to lock ({diff_lock:.1%})")
    else:
        print(f"   → No significant difference ({diff_lock:+.1%})")

    print(f"\n2. CAPTURE RATE COMPARISON:")
    print(f"   5-point counts: {five_pt_avg_capture:.1%}")
    print(f"   10-point counts: {ten_pt_avg_capture:.1%}")

    print(f"\n3. E[V] CORRELATION:")
    print(f"   total_locks vs E[V]: {corr_total:+.3f}")
    if corr_total > 0.3:
        print(f"   → More locks = higher E[V] (moderate correlation)")
    elif corr_total > 0:
        print(f"   → Weak positive relationship")
    else:
        print(f"   → No or negative relationship")

    print(f"\n4. INDIVIDUAL COUNT RANKINGS (by lock rate):")
    for _, row in lock_df.sort_values('lock_rate', ascending=False).iterrows():
        print(f"   {row['count']} ({row['points']}pt): {row['lock_rate']:.1%}")


if __name__ == "__main__":
    main()
