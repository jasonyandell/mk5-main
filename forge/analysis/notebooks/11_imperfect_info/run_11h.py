#!/usr/bin/env python3
"""
11h: Path Divergence Analysis

Question: When do paths diverge across opponent configs?
Method: Compare best moves at each depth across 3 opponent configs
What It Reveals: Early divergence = opponent-dependent strategy

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
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
N_BASE_SEEDS = 201  # Analyze all available base seeds
MAX_ROWS = 8_000_000
np.random.seed(42)


ILLEGAL = -128
MAX_STATES_PER_SEED = 500000  # Sample to keep memory/time bounded


def analyze_divergence_for_base_seed(db: SeedDB, base_seed: int) -> dict | None:
    """Compare best moves across 3 configs at each depth using SQL JOIN."""
    decl_id = base_seed % 10

    # Build file paths
    files = []
    for opp_seed in range(3):
        filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        path = DATA_DIR / filename
        if not path.exists():
            return None
        files.append(str(path))

    # SQL to compute argmax(Q) for each config
    # argmax with CASE WHEN chains - find max legal Q then which index has it
    def argmax_sql(prefix: str) -> str:
        """Generate SQL for argmax of Q values with given prefix."""
        # First compute the max value among legal actions
        max_expr = f"GREATEST(" + ", ".join([
            f"CASE WHEN {prefix}q{i} != {ILLEGAL} THEN {prefix}q{i} ELSE -1000 END"
            for i in range(7)
        ]) + ")"
        # Then find which index matches (first match wins)
        return f"""CASE
            WHEN {prefix}q0 != {ILLEGAL} AND {prefix}q0 = {max_expr} THEN 0
            WHEN {prefix}q1 != {ILLEGAL} AND {prefix}q1 = {max_expr} THEN 1
            WHEN {prefix}q2 != {ILLEGAL} AND {prefix}q2 = {max_expr} THEN 2
            WHEN {prefix}q3 != {ILLEGAL} AND {prefix}q3 = {max_expr} THEN 3
            WHEN {prefix}q4 != {ILLEGAL} AND {prefix}q4 = {max_expr} THEN 4
            WHEN {prefix}q5 != {ILLEGAL} AND {prefix}q5 = {max_expr} THEN 5
            WHEN {prefix}q6 != {ILLEGAL} AND {prefix}q6 = {max_expr} THEN 6
            ELSE -1
        END"""

    sql = f"""
    WITH joined AS (
        SELECT
            a.state,
            depth(a.state) as depth,
            a.q0 as a_q0, a.q1 as a_q1, a.q2 as a_q2, a.q3 as a_q3, a.q4 as a_q4, a.q5 as a_q5, a.q6 as a_q6,
            b.q0 as b_q0, b.q1 as b_q1, b.q2 as b_q2, b.q3 as b_q3, b.q4 as b_q4, b.q5 as b_q5, b.q6 as b_q6,
            c.q0 as c_q0, c.q1 as c_q1, c.q2 as c_q2, c.q3 as c_q3, c.q4 as c_q4, c.q5 as c_q5, c.q6 as c_q6
        FROM read_parquet('{files[0]}') a
        INNER JOIN read_parquet('{files[1]}') b ON a.state = b.state
        INNER JOIN read_parquet('{files[2]}') c ON a.state = c.state
        WHERE depth(a.state) >= 5 AND depth(a.state) <= 28
        USING SAMPLE {MAX_STATES_PER_SEED} ROWS
    ),
    with_best AS (
        SELECT
            depth,
            {argmax_sql('a_')} as best_a,
            {argmax_sql('b_')} as best_b,
            {argmax_sql('c_')} as best_c
        FROM joined
    )
    SELECT
        depth,
        COUNT(*) as n_common,
        SUM(CASE WHEN best_a = best_b AND best_b = best_c THEN 1 ELSE 0 END) as n_consistent
    FROM with_best
    GROUP BY depth
    ORDER BY depth DESC
    """

    try:
        result = db.execute(sql)
        if result.data is None or len(result.data) == 0:
            return None

        df = result.data

        # Build depth stats
        depth_stats = {}
        first_divergence_depth = None

        for _, row in df.iterrows():
            d = int(row['depth'])
            n_common = int(row['n_common'])
            n_consistent = int(row['n_consistent'])
            rate = n_consistent / n_common if n_common > 0 else None

            depth_stats[d] = {
                'n_common': n_common,
                'n_consistent': n_consistent,
                'rate': rate
            }

            if first_divergence_depth is None and rate is not None and rate < 1.0:
                first_divergence_depth = d

        # Summary metrics
        early_game_consistency = []
        mid_game_consistency = []
        late_game_consistency = []

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

    except Exception as e:
        print(f"Error on seed {base_seed}: {e}")
        return None


def main():
    print("=" * 60)
    print("PATH DIVERGENCE ANALYSIS")
    print("=" * 60)

    # Initialize SeedDB
    db = SeedDB(DATA_DIR)

    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    all_results = []
    depth_agg = defaultdict(lambda: {'total_common': 0, 'total_consistent': 0})

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_divergence_for_base_seed(db, base_seed)
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

    db.close()
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
