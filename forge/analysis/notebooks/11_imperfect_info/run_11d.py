#!/usr/bin/env python3
"""
11d: Q-value Variance Analysis

Question: How much do Q-values vary per position across opponent configs?
Method: σ(Q) for each move across opponent configs
What It Reveals: Confidence in move choice under uncertainty

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

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Analyze all available base seeds
MAX_STATES_PER_SHARD = 500_000  # Sample if larger (balance memory vs coverage)
np.random.seed(42)  # For reproducibility


def analyze_q_variance_for_base_seed(base_seed: int) -> dict | None:
    """Analyze Q-value variance across 3 opponent configs for one base seed.

    Memory-optimized: load one shard at a time, build minimal mappings.
    """
    decl_id = base_seed % 10
    ILLEGAL = -128

    # First pass: get states and Q-values for each config
    state_to_q = [{}, {}, {}]
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
                indices = np.random.choice(n_rows, MAX_STATES_PER_SHARD, replace=False)
                states = df['state'].values[indices]
                q_cols = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
                q_values = df[q_cols].values[indices].astype(np.int16)
            else:
                states = df['state'].values
                q_cols = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
                q_values = df[q_cols].values.astype(np.int16)

            state_to_q[opp_seed] = dict(zip(states, [tuple(q) for q in q_values]))
            states_set[opp_seed] = set(states)

            # Save depths from first config
            if opp_seed == 0:
                depths = features.depth(states)
                state_to_depth = dict(zip(states, depths))

            del df, states, q_values
            gc.collect()

        except Exception as e:
            print(f"Error loading {path}: {e}")
            gc.collect()
            return None

    # Find common states
    common_states = states_set[0] & states_set[1] & states_set[2]

    del states_set
    gc.collect()

    if len(common_states) == 0:
        return None

    # Analyze Q-value variance
    q_std_by_slot = defaultdict(list)  # slot -> list of std values
    q_std_by_depth = defaultdict(list)  # depth -> list of (slot, std) tuples
    q_range_by_slot = defaultdict(list)  # slot -> list of ranges

    n_analyzed = 0
    total_legal_actions = 0
    high_variance_positions = 0  # σ(Q) > 5 for at least one action

    for state in common_states:
        q0 = state_to_q[0].get(state)
        q1 = state_to_q[1].get(state)
        q2 = state_to_q[2].get(state)

        if q0 is None or q1 is None or q2 is None:
            continue

        n_analyzed += 1
        depth = state_to_depth.get(state, -1)

        has_high_variance = False

        # For each action slot
        for slot in range(7):
            v0, v1, v2 = q0[slot], q1[slot], q2[slot]

            # Skip illegal actions
            if v0 == ILLEGAL or v1 == ILLEGAL or v2 == ILLEGAL:
                continue

            total_legal_actions += 1
            q_std = np.std([v0, v1, v2])
            q_range = max(v0, v1, v2) - min(v0, v1, v2)

            q_std_by_slot[slot].append(q_std)
            q_range_by_slot[slot].append(q_range)

            if depth >= 0:
                q_std_by_depth[depth].append((slot, q_std))

            if q_std > 5:
                has_high_variance = True

        if has_high_variance:
            high_variance_positions += 1

    del state_to_q, common_states, state_to_depth
    gc.collect()

    if n_analyzed == 0:
        return None

    # Aggregate results
    slot_stats = {}
    for slot in range(7):
        if q_std_by_slot[slot]:
            slot_stats[slot] = {
                'mean_std': np.mean(q_std_by_slot[slot]),
                'median_std': np.median(q_std_by_slot[slot]),
                'max_std': np.max(q_std_by_slot[slot]),
                'mean_range': np.mean(q_range_by_slot[slot]),
                'n_observations': len(q_std_by_slot[slot])
            }

    depth_stats = {}
    for depth in sorted(q_std_by_depth.keys()):
        stds = [s for _, s in q_std_by_depth[depth]]
        if stds:
            depth_stats[depth] = {
                'mean_std': np.mean(stds),
                'median_std': np.median(stds),
                'n_observations': len(stds)
            }

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'n_common_states': n_analyzed,
        'total_legal_actions': total_legal_actions,
        'high_variance_rate': high_variance_positions / n_analyzed if n_analyzed > 0 else 0,
        'slot_stats': slot_stats,
        'depth_stats': depth_stats
    }


def main():
    print("=" * 60)
    print("Q-VALUE VARIANCE ANALYSIS")
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

    # Aggregate across all seeds
    all_slot_stds = defaultdict(list)
    all_depth_stds = defaultdict(list)
    all_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_q_variance_for_base_seed(base_seed)
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
