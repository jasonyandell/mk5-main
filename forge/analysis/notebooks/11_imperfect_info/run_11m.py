#!/usr/bin/env python3
"""
11m: Lock Rate by Trump Length

Question: Does holding trump lock more counts?
Method: Correlate trump length with lock rates
What It Reveals: Trump control → count control?

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from forge.analysis.utils.seed_db import SeedDB
from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed, deal_with_fixed_p0

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"
N_BASE_SEEDS = 201  # Full analysis
np.random.seed(42)

# Count dominoes: (id, name, points)
COUNT_DOMINOES = [
    (9, '3-2', 5),   # domino ID 9 = 3-2
    (14, '4-1', 5),  # domino ID 14 = 4-1
    (20, '5-0', 5),  # domino ID 20 = 5-0
    (18, '5-5', 10),  # domino ID 18 = 5-5
    (27, '6-4', 10),  # domino ID 27 = 6-4
]


def is_trump(domino_id: int, trump_suit: int) -> bool:
    """Check if domino is a trump (either pip matches trump suit)."""
    high, low = schema.domino_pips(domino_id)
    return high == trump_suit or low == trump_suit


def get_trump_count(hand: list[int], trump_suit: int) -> int:
    """Count trump dominoes in hand."""
    return sum(1 for d in hand if is_trump(d, trump_suit))


def has_trump_double(hand: list[int], trump_suit: int) -> bool:
    """Check if hand contains the trump double (e.g., 6-6 when trump is 6)."""
    # Trump double has ID = trump_suit * (trump_suit + 3) // 2 for triangular number
    # But easier: check if any domino is (trump, trump)
    for d in hand:
        high, low = schema.domino_pips(d)
        if high == trump_suit and low == trump_suit:
            return True
    return False


def get_team_ownership(hands: list[list[int]]) -> dict[int, int]:
    """Get team ownership for each count domino.

    Returns: {domino_id: team} where team is 0 (P0+P2) or 1 (P1+P3)
    """
    ownership = {}
    for d_id, _, _ in COUNT_DOMINOES:
        for player, hand in enumerate(hands):
            if d_id in hand:
                team = 0 if player in [0, 2] else 1
                ownership[d_id] = team
                break
    return ownership


def analyze_for_base_seed(db: SeedDB, base_seed: int) -> dict | None:
    """Analyze trump length vs lock rates for one base seed."""
    decl_id = base_seed % 10
    trump_suit = decl_id  # In Texas 42, decl_id IS the trump suit
    p0_hand = deal_from_seed(base_seed)[0]

    # Trump features
    trump_count = get_trump_count(p0_hand, trump_suit)
    trump_double = has_trump_double(p0_hand, trump_suit)

    # Collect V and ownership for each opponent config
    config_data = []
    for opp_seed in range(3):
        filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        path = DATA_DIR / filename
        if not path.exists():
            return None

        result = db.get_root_v(filename)
        if result.data is None:
            return None
        V = float(result.data)

        # Reconstruct full deal for this config
        hands = deal_with_fixed_p0(p0_hand, opp_seed)
        ownership = get_team_ownership(hands)

        config_data.append({'V': V, 'ownership': ownership})

    if len(config_data) != 3:
        return None

    # Analyze each count
    count_results = {}
    total_locks = 0
    for d_id, name, points in COUNT_DOMINOES:
        # Check if Team 0 owns this count in each config
        team0_owns = [cfg['ownership'].get(d_id, -1) == 0 for cfg in config_data]

        # "Lock" = Team 0 owns in all 3 configs
        locked = all(team0_owns)
        if locked:
            total_locks += 1

        # "Capture rate" = fraction of configs where Team 0 owns
        capture_rate = sum(team0_owns) / 3

        count_results[name] = {
            'locked': locked,
            'capture_rate': capture_rate
        }

    # V statistics
    V_values = [cfg['V'] for cfg in config_data]
    V_mean = np.mean(V_values)
    V_spread = max(V_values) - min(V_values)

    # Overall capture rate
    avg_capture_rate = np.mean([count_results[n]['capture_rate'] for _, n, _ in COUNT_DOMINOES])

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'trump_suit': trump_suit,
        'trump_count': trump_count,
        'has_trump_double': trump_double,
        'total_locks': total_locks,
        'avg_capture_rate': avg_capture_rate,
        'V_mean': V_mean,
        'V_spread': V_spread,
        # Per-count lock data
        **{f'{name}_locked': count_results[name]['locked'] for _, name, _ in COUNT_DOMINOES},
        **{f'{name}_rate': count_results[name]['capture_rate'] for _, name, _ in COUNT_DOMINOES},
    }


def main():
    print("=" * 60)
    print("LOCK RATE BY TRUMP LENGTH")
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
        result = analyze_for_base_seed(db, base_seed)
        if result:
            all_results.append(result)

    db.close()
    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    # Analysis by trump count
    print("\n" + "=" * 60)
    print("LOCK RATES BY TRUMP COUNT")
    print("=" * 60)

    by_trump = df.groupby('trump_count').agg({
        'total_locks': ['mean', 'std', 'count'],
        'avg_capture_rate': 'mean',
        'V_mean': 'mean',
        'V_spread': 'mean'
    }).round(3)
    print("\n" + by_trump.to_string())

    # Correlations
    print("\n" + "=" * 60)
    print("CORRELATIONS")
    print("=" * 60)

    corr_locks = df['trump_count'].corr(df['total_locks'])
    corr_capture = df['trump_count'].corr(df['avg_capture_rate'])
    corr_v = df['trump_count'].corr(df['V_mean'])
    corr_spread = df['trump_count'].corr(df['V_spread'])

    print(f"\n  trump_count vs total_locks: {corr_locks:+.3f}")
    print(f"  trump_count vs avg_capture_rate: {corr_capture:+.3f}")
    print(f"  trump_count vs E[V]: {corr_v:+.3f}")
    print(f"  trump_count vs V_spread: {corr_spread:+.3f}")

    # Trump double effect
    print("\n" + "=" * 60)
    print("TRUMP DOUBLE EFFECT")
    print("=" * 60)

    with_double = df[df['has_trump_double']]
    without_double = df[~df['has_trump_double']]

    print(f"\n  With trump double (n={len(with_double)}):")
    if len(with_double) > 0:
        print(f"    Avg locks: {with_double['total_locks'].mean():.2f}")
        print(f"    Avg capture rate: {with_double['avg_capture_rate'].mean():.1%}")
        print(f"    E[V]: {with_double['V_mean'].mean():.1f}")

    print(f"\n  Without trump double (n={len(without_double)}):")
    if len(without_double) > 0:
        print(f"    Avg locks: {without_double['total_locks'].mean():.2f}")
        print(f"    Avg capture rate: {without_double['avg_capture_rate'].mean():.1%}")
        print(f"    E[V]: {without_double['V_mean'].mean():.1f}")

    if len(with_double) > 0 and len(without_double) > 0:
        lock_diff = with_double['total_locks'].mean() - without_double['total_locks'].mean()
        v_diff = with_double['V_mean'].mean() - without_double['V_mean'].mean()
        print(f"\n  Difference (with - without):")
        print(f"    Locks: {lock_diff:+.2f}")
        print(f"    E[V]: {v_diff:+.1f}")

    # Per-count analysis by trump length
    print("\n" + "=" * 60)
    print("PER-COUNT LOCK RATES BY TRUMP LENGTH")
    print("=" * 60)

    for _, name, points in COUNT_DOMINOES:
        corr = df['trump_count'].corr(df[f'{name}_rate'])
        print(f"  {name} ({points}pt) vs trump_count: {corr:+.3f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11m_lock_by_trump_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Summary by trump count
    summary_data = []
    for trump_ct in sorted(df['trump_count'].unique()):
        subset = df[df['trump_count'] == trump_ct]
        summary_data.append({
            'trump_count': trump_ct,
            'n_hands': len(subset),
            'avg_locks': subset['total_locks'].mean(),
            'avg_capture_rate': subset['avg_capture_rate'].mean(),
            'avg_ev': subset['V_mean'].mean(),
            'avg_spread': subset['V_spread'].mean()
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(tables_dir / "11m_lock_by_trump_summary.csv", index=False)
    print("✓ Saved summary by trump count")

    # Correlation summary
    corr_data = [{
        'trump_vs_locks': corr_locks,
        'trump_vs_capture': corr_capture,
        'trump_vs_ev': corr_v,
        'trump_vs_spread': corr_spread,
        'n_samples': len(df)
    }]
    pd.DataFrame(corr_data).to_csv(tables_dir / "11m_correlations.csv", index=False)
    print("✓ Saved correlations")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Lock rate by trump count
    ax1 = axes[0, 0]
    lock_by_trump = df.groupby('trump_count')['total_locks'].mean()
    ax1.bar(lock_by_trump.index, lock_by_trump.values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Trump Count')
    ax1.set_ylabel('Average Locked Counts')
    ax1.set_title(f'Locked Counts by Trump Length (r = {corr_locks:.2f})')
    ax1.set_xticks(range(8))

    # Top right: Capture rate by trump count
    ax2 = axes[0, 1]
    capture_by_trump = df.groupby('trump_count')['avg_capture_rate'].mean()
    ax2.bar(capture_by_trump.index, capture_by_trump.values * 100, color='green', alpha=0.7)
    ax2.set_xlabel('Trump Count')
    ax2.set_ylabel('Average Capture Rate (%)')
    ax2.set_title(f'Capture Rate by Trump Length (r = {corr_capture:.2f})')
    ax2.set_xticks(range(8))

    # Bottom left: E[V] by trump count
    ax3 = axes[1, 0]
    ev_by_trump = df.groupby('trump_count')['V_mean'].mean()
    ax3.bar(ev_by_trump.index, ev_by_trump.values, color='orange', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Trump Count')
    ax3.set_ylabel('E[V]')
    ax3.set_title(f'Expected Value by Trump Length (r = {corr_v:.2f})')
    ax3.set_xticks(range(8))

    # Bottom right: Scatter - trump count vs total locks
    ax4 = axes[1, 1]
    # Add jitter for visibility
    jitter_x = df['trump_count'] + np.random.uniform(-0.2, 0.2, len(df))
    jitter_y = df['total_locks'] + np.random.uniform(-0.2, 0.2, len(df))
    ax4.scatter(jitter_x, jitter_y, alpha=0.5, s=30)
    # Add regression line
    z = np.polyfit(df['trump_count'], df['total_locks'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 7, 100)
    ax4.plot(x_line, p(x_line), 'r-', linewidth=2,
             label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
    ax4.set_xlabel('Trump Count')
    ax4.set_ylabel('Total Locked Counts')
    ax4.set_title('Trump Count vs Locked Counts')
    ax4.legend()

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11m_lock_by_trump.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. TRUMP LENGTH → LOCK RATE:")
    print(f"   Correlation: {corr_locks:+.3f}")
    if corr_locks > 0.3:
        print(f"   → More trump = more locked counts (moderate relationship)")
    elif corr_locks > 0.1:
        print(f"   → Weak positive relationship")
    elif corr_locks > -0.1:
        print(f"   → No significant relationship")
    else:
        print(f"   → Negative relationship (surprising!)")

    print(f"\n2. TRUMP LENGTH → E[V]:")
    print(f"   Correlation: {corr_v:+.3f}")
    if corr_v > 0.3:
        print(f"   → More trump = higher expected value")
    elif corr_v > 0:
        print(f"   → Weak positive relationship")
    else:
        print(f"   → No or negative relationship")

    print(f"\n3. TRUMP DOUBLE EFFECT:")
    if len(with_double) > 0 and len(without_double) > 0:
        lock_diff = with_double['total_locks'].mean() - without_double['total_locks'].mean()
        v_diff = with_double['V_mean'].mean() - without_double['V_mean'].mean()
        if lock_diff > 0.3:
            print(f"   → Trump double adds +{lock_diff:.1f} locks on average")
        elif lock_diff > 0:
            print(f"   → Trump double has small positive effect (+{lock_diff:.2f} locks)")
        else:
            print(f"   → Trump double has no/negative effect ({lock_diff:+.2f} locks)")
        print(f"   → E[V] difference: {v_diff:+.1f}")

    print(f"\n4. LOCK RATE BY TRUMP LENGTH:")
    for trump_ct in sorted(df['trump_count'].unique()):
        subset = df[df['trump_count'] == trump_ct]
        print(f"   {trump_ct} trump: {subset['total_locks'].mean():.1f} locks "
              f"({subset['avg_capture_rate'].mean():.0%} capture), n={len(subset)}")


if __name__ == "__main__":
    main()
