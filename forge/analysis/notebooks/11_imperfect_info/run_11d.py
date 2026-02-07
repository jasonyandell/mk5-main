#!/usr/bin/env python3
"""
11d: Q-value Variance Analysis

Question: How much do Q-values vary per position across opponent configs?
Method: σ(Q) for each move across opponent configs
What It Reveals: Confidence in move choice under uncertainty

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
MAX_STATES_PER_SHARD = 500_000  # Sample if larger (balance memory vs coverage)
np.random.seed(42)  # For reproducibility


def analyze_q_variance_for_base_seed(db: SeedDB, base_seed: int) -> dict | None:
    """Analyze Q-value variance across 3 opponent configs using SQL JOIN.

    Optimized: Single SQL query joins all 3 files and computes variance in DuckDB.
    """
    decl_id = base_seed % 10
    ILLEGAL = -128

    # Build file paths
    files = [
        str(DATA_DIR / f"seed_{base_seed:08d}_opp{i}_decl_{decl_id}.parquet")
        for i in range(3)
    ]

    # Check files exist
    for f in files:
        if not Path(f).exists():
            return None

    try:
        # Single SQL query: JOIN all 3 files on state, compute variance per slot
        # This is MUCH faster than loading into Python and doing set intersection
        sql = f"""
        WITH joined AS (
            SELECT
                a.state,
                depth(a.state) as depth,
                a.q0 as q0_0, a.q1 as q1_0, a.q2 as q2_0, a.q3 as q3_0, a.q4 as q4_0, a.q5 as q5_0, a.q6 as q6_0,
                b.q0 as q0_1, b.q1 as q1_1, b.q2 as q2_1, b.q3 as q3_1, b.q4 as q4_1, b.q5 as q5_1, b.q6 as q6_1,
                c.q0 as q0_2, c.q1 as q1_2, c.q2 as q2_2, c.q3 as q3_2, c.q4 as q4_2, c.q5 as q5_2, c.q6 as q6_2
            FROM read_parquet('{files[0]}') a
            INNER JOIN read_parquet('{files[1]}') b ON a.state = b.state
            INNER JOIN read_parquet('{files[2]}') c ON a.state = c.state
            USING SAMPLE {MAX_STATES_PER_SHARD} ROWS
        ),
        per_state AS (
            SELECT
                state, depth,
                -- Variance for each slot (only if all 3 are legal, i.e., != -128)
                -- stddev = sqrt((sum of squares)/n - mean^2) for population stddev
                -- Cast to DOUBLE to avoid INT8 overflow when squaring
                CASE WHEN q0_0 != {ILLEGAL} AND q0_1 != {ILLEGAL} AND q0_2 != {ILLEGAL}
                     THEN sqrt((q0_0::DOUBLE*q0_0 + q0_1::DOUBLE*q0_1 + q0_2::DOUBLE*q0_2)/3.0 - power((q0_0 + q0_1 + q0_2)/3.0, 2)) END as std_0,
                CASE WHEN q1_0 != {ILLEGAL} AND q1_1 != {ILLEGAL} AND q1_2 != {ILLEGAL}
                     THEN sqrt((q1_0::DOUBLE*q1_0 + q1_1::DOUBLE*q1_1 + q1_2::DOUBLE*q1_2)/3.0 - power((q1_0 + q1_1 + q1_2)/3.0, 2)) END as std_1,
                CASE WHEN q2_0 != {ILLEGAL} AND q2_1 != {ILLEGAL} AND q2_2 != {ILLEGAL}
                     THEN sqrt((q2_0::DOUBLE*q2_0 + q2_1::DOUBLE*q2_1 + q2_2::DOUBLE*q2_2)/3.0 - power((q2_0 + q2_1 + q2_2)/3.0, 2)) END as std_2,
                CASE WHEN q3_0 != {ILLEGAL} AND q3_1 != {ILLEGAL} AND q3_2 != {ILLEGAL}
                     THEN sqrt((q3_0::DOUBLE*q3_0 + q3_1::DOUBLE*q3_1 + q3_2::DOUBLE*q3_2)/3.0 - power((q3_0 + q3_1 + q3_2)/3.0, 2)) END as std_3,
                CASE WHEN q4_0 != {ILLEGAL} AND q4_1 != {ILLEGAL} AND q4_2 != {ILLEGAL}
                     THEN sqrt((q4_0::DOUBLE*q4_0 + q4_1::DOUBLE*q4_1 + q4_2::DOUBLE*q4_2)/3.0 - power((q4_0 + q4_1 + q4_2)/3.0, 2)) END as std_4,
                CASE WHEN q5_0 != {ILLEGAL} AND q5_1 != {ILLEGAL} AND q5_2 != {ILLEGAL}
                     THEN sqrt((q5_0::DOUBLE*q5_0 + q5_1::DOUBLE*q5_1 + q5_2::DOUBLE*q5_2)/3.0 - power((q5_0 + q5_1 + q5_2)/3.0, 2)) END as std_5,
                CASE WHEN q6_0 != {ILLEGAL} AND q6_1 != {ILLEGAL} AND q6_2 != {ILLEGAL}
                     THEN sqrt((q6_0::DOUBLE*q6_0 + q6_1::DOUBLE*q6_1 + q6_2::DOUBLE*q6_2)/3.0 - power((q6_0 + q6_1 + q6_2)/3.0, 2)) END as std_6
            FROM joined
        )
        SELECT
            depth,
            COUNT(*) as n_states,
            -- Per-slot aggregates
            AVG(std_0) as mean_std_0, COUNT(std_0) as n_0,
            AVG(std_1) as mean_std_1, COUNT(std_1) as n_1,
            AVG(std_2) as mean_std_2, COUNT(std_2) as n_2,
            AVG(std_3) as mean_std_3, COUNT(std_3) as n_3,
            AVG(std_4) as mean_std_4, COUNT(std_4) as n_4,
            AVG(std_5) as mean_std_5, COUNT(std_5) as n_5,
            AVG(std_6) as mean_std_6, COUNT(std_6) as n_6,
            -- High variance rate (any slot > 5)
            SUM(CASE WHEN COALESCE(std_0,0) > 5 OR COALESCE(std_1,0) > 5 OR COALESCE(std_2,0) > 5
                       OR COALESCE(std_3,0) > 5 OR COALESCE(std_4,0) > 5 OR COALESCE(std_5,0) > 5
                       OR COALESCE(std_6,0) > 5 THEN 1 ELSE 0 END) as high_var_count
        FROM per_state
        GROUP BY depth
        ORDER BY depth DESC
        """

        result = db.execute(sql)
        if result.data is None or len(result.data) == 0:
            return None

        df = result.data

        # Aggregate results from SQL output
        n_analyzed = int(df['n_states'].sum())
        if n_analyzed == 0:
            return None

        # Slot stats
        slot_stats = {}
        total_legal_actions = 0
        for slot in range(7):
            n_col = f'n_{slot}'
            mean_col = f'mean_std_{slot}'
            n_obs = int(df[n_col].sum())
            if n_obs > 0:
                # Weighted average across depths
                weighted_mean = (df[mean_col] * df[n_col]).sum() / n_obs
                slot_stats[slot] = {
                    'mean_std': float(weighted_mean),
                    'n_observations': n_obs
                }
                total_legal_actions += n_obs

        # Depth stats
        depth_stats = {}
        for _, row in df.iterrows():
            depth = int(row['depth'])
            # Average across all slots for this depth
            slot_means = [row[f'mean_std_{s}'] for s in range(7) if pd.notna(row[f'mean_std_{s}'])]
            if slot_means:
                depth_stats[depth] = {
                    'mean_std': float(np.mean(slot_means)),
                    'n_observations': int(row['n_states'])
                }

        high_variance_positions = int(df['high_var_count'].sum())

        return {
            'base_seed': base_seed,
            'decl_id': decl_id,
            'n_common_states': n_analyzed,
            'total_legal_actions': total_legal_actions,
            'high_variance_rate': high_variance_positions / n_analyzed if n_analyzed > 0 else 0,
            'slot_stats': slot_stats,
            'depth_stats': depth_stats
        }

    except Exception as e:
        print(f"Error analyzing base_seed {base_seed}: {e}")
        return None


def main():
    print("=" * 60)
    print("Q-VALUE VARIANCE ANALYSIS")
    print("=" * 60)

    # Initialize SeedDB
    db = SeedDB(DATA_DIR)

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

    # Aggregate across all seeds
    all_slot_stds = defaultdict(list)
    all_depth_stds = defaultdict(list)
    all_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_q_variance_for_base_seed(db, base_seed)
        if result:
            all_results.append({
                'base_seed': result['base_seed'],
                'decl_id': result['decl_id'],
                'n_common_states': result['n_common_states'],
                'total_legal_actions': result['total_legal_actions'],
                'high_variance_rate': result['high_variance_rate']
            })

            # Aggregate slot stats
            for slot, stats in result['slot_stats'].items():
                all_slot_stds[slot].extend([stats['mean_std']] * stats['n_observations'])

            # Aggregate depth stats
            for depth, stats in result['depth_stats'].items():
                all_depth_stds[depth].extend([stats['mean_std']] * stats['n_observations'])

    db.close()
    print(f"\n✓ Analyzed {len(all_results)} base seeds")

    # Build results DataFrames
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    total_states = df['n_common_states'].sum()
    total_actions = df['total_legal_actions'].sum()
    mean_high_var_rate = df['high_variance_rate'].mean()

    print(f"\nTotal common states analyzed: {total_states:,}")
    print(f"Total legal actions analyzed: {total_actions:,}")
    print(f"Mean high variance rate (σ>5): {mean_high_var_rate:.1%}")

    # Slot analysis
    print("\n" + "=" * 60)
    print("Q-VALUE VARIANCE BY ACTION SLOT")
    print("=" * 60)

    slot_data = []
    for slot in range(7):
        if all_slot_stds[slot]:
            stds = all_slot_stds[slot]
            slot_data.append({
                'slot': slot,
                'mean_std': np.mean(stds),
                'median_std': np.median(stds),
                'std_of_std': np.std(stds),
                'pct_high_var': sum(1 for s in stds if s > 5) / len(stds),
                'n_observations': len(stds)
            })

    slot_df = pd.DataFrame(slot_data)
    print("\nQ-value standard deviation by action slot:")
    print(slot_df.to_string(index=False))

    # Depth analysis
    print("\n" + "=" * 60)
    print("Q-VALUE VARIANCE BY DEPTH")
    print("=" * 60)

    depth_data = []
    for depth in sorted(all_depth_stds.keys()):
        stds = all_depth_stds[depth]
        if stds:
            depth_data.append({
                'depth': depth,
                'mean_std': np.mean(stds),
                'median_std': np.median(stds),
                'pct_high_var': sum(1 for s in stds if s > 5) / len(stds),
                'n_observations': len(stds)
            })

    depth_df = pd.DataFrame(depth_data)
    print("\nQ-value standard deviation by depth (dominoes remaining):")
    print(depth_df.to_string(index=False))

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Save per-seed results
    df.to_csv(tables_dir / "11d_q_variance_by_seed.csv", index=False)
    print("✓ Saved per-seed results")

    # Save slot breakdown
    slot_df.to_csv(tables_dir / "11d_q_variance_by_slot.csv", index=False)
    print("✓ Saved slot breakdown")

    # Save depth breakdown
    depth_df.to_csv(tables_dir / "11d_q_variance_by_depth.csv", index=False)
    print("✓ Saved depth breakdown")

    # Save summary
    overall_mean_std = np.mean([s for stds in all_slot_stds.values() for s in stds]) if all_slot_stds else 0
    overall_pct_high = np.mean([
        sum(1 for s in stds if s > 5) / len(stds)
        for stds in all_slot_stds.values() if stds
    ]) if all_slot_stds else 0

    summary = pd.DataFrame([{
        'metric': 'overall_mean_q_std',
        'value': overall_mean_std,
        'n_states': total_states,
        'n_actions': total_actions,
        'pct_high_variance': overall_pct_high
    }])
    summary.to_csv(tables_dir / "11d_q_variance_summary.csv", index=False)
    print("✓ Saved summary")

    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Q-variance by depth
    ax1 = axes[0]
    ax1.bar(depth_df['depth'], depth_df['mean_std'], color='steelblue', alpha=0.7)
    ax1.axhline(y=overall_mean_std, color='red', linestyle='--', label=f'Overall: {overall_mean_std:.2f}')
    ax1.set_xlabel('Depth (dominoes remaining)')
    ax1.set_ylabel('Mean σ(Q) across opponent configs')
    ax1.set_title('Q-Value Uncertainty by Game Depth')
    ax1.legend()

    # Right: Q-variance by slot
    ax2 = axes[1]
    ax2.bar(slot_df['slot'], slot_df['mean_std'], color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axhline(y=overall_mean_std, color='red', linestyle='--', label=f'Mean: {overall_mean_std:.2f}')
    ax2.set_xlabel('Action Slot')
    ax2.set_ylabel('Mean σ(Q) across opponent configs')
    ax2.set_title('Q-Value Uncertainty by Action Slot')
    ax2.set_xticks(range(7))
    ax2.legend()

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11d_q_value_variance.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. OVERALL Q-UNCERTAINTY: Mean σ(Q) = {overall_mean_std:.2f} points")
    print(f"   - {overall_pct_high:.1%} of actions have high variance (σ > 5)")

    # Most vs least uncertain depths
    if len(depth_df) > 0:
        most_uncertain_depth = depth_df.loc[depth_df['mean_std'].idxmax()]
        least_uncertain_depth = depth_df.loc[depth_df['mean_std'].idxmin()]
        print(f"\n2. DEPTH EFFECT:")
        print(f"   Most uncertain: depth {int(most_uncertain_depth['depth'])} (σ = {most_uncertain_depth['mean_std']:.2f})")
        print(f"   Least uncertain: depth {int(least_uncertain_depth['depth'])} (σ = {least_uncertain_depth['mean_std']:.2f})")

    # Interpretation
    print(f"\n3. INTERPRETATION:")
    if overall_mean_std < 5:
        print("   Low Q-variance suggests move quality is relatively stable across opponent configs")
    elif overall_mean_std < 10:
        print("   Moderate Q-variance - opponent hands matter somewhat for move evaluation")
    else:
        print("   High Q-variance - opponent hands significantly affect move quality")


if __name__ == "__main__":
    main()
