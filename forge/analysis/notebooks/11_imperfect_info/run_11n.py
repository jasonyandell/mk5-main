#!/usr/bin/env python3
"""
11n: Decision Point Consistency

Question: Are critical decisions the same across opponent configs?
Method: Track which positions have Q-gap > 5 across configs
What It Reveals: Stable critical decisions vs opponent-dependent

A "critical decision" is one where Q-gap > threshold, meaning the choice matters.
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
from forge.oracle import schema

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 50  # Use subset for speed (preliminary)
MAX_STATES_PER_SEED = 5000
Q_GAP_THRESHOLD = 5  # Points - a "critical" decision
np.random.seed(42)


def load_common_states_for_seed(base_seed: int, max_states: int = MAX_STATES_PER_SEED) -> dict | None:
    """Load states that appear in all 3 opponent configs and get their Q values."""
    decl_id = base_seed % 10

    config_data = []

    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        try:
            pf = pq.ParquetFile(path)
            states_q = {}

            for batch in pf.iter_batches(batch_size=50000, columns=['state', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']):
                states = batch['state'].to_numpy()
                Q = np.column_stack([
                    batch['q0'].to_numpy(),
                    batch['q1'].to_numpy(),
                    batch['q2'].to_numpy(),
                    batch['q3'].to_numpy(),
                    batch['q4'].to_numpy(),
                    batch['q5'].to_numpy(),
                    batch['q6'].to_numpy()
                ])

                for i, s in enumerate(states):
                    if s not in states_q:
                        states_q[s] = Q[i]

                if len(states_q) > max_states * 3:
                    break

            config_data.append(states_q)
        except Exception:
            return None

    if len(config_data) != 3:
        return None

    common = set(config_data[0].keys()) & set(config_data[1].keys()) & set(config_data[2].keys())

    if len(common) == 0:
        return None

    common = list(common)[:max_states]

    result = {}
    for s in common:
        result[s] = [config_data[0][s], config_data[1][s], config_data[2][s]]

    return result


def get_q_gap(Q: np.ndarray) -> float:
    """Get Q-gap (best - second-best) for a Q vector."""
    legal = Q[Q > -128]
    if len(legal) < 2:
        return 0.0
    sorted_q = np.sort(legal)[::-1]
    return sorted_q[0] - sorted_q[1]


def get_best_action(Q: np.ndarray) -> int:
    """Get best legal action."""
    legal_mask = Q > -128
    if not legal_mask.any():
        return -1
    Q_legal = Q.astype(np.float32)
    Q_legal[~legal_mask] = -np.inf
    return int(np.argmax(Q_legal))


def analyze_decision_consistency(state_data: dict) -> dict:
    """Analyze decision point consistency across opponent configs.

    For each state:
    - Compute Q-gap in each config
    - A state is "critical in all" if Q-gap > threshold in all 3 configs
    - Check if best move is consistent for critical decisions
    """
    results = {
        'total_states': 0,
        'critical_in_all': 0,  # Critical in all 3 configs
        'critical_in_any': 0,  # Critical in at least 1 config
        'critical_consistent': 0,  # Critical in all AND same best move
        'critical_inconsistent': 0,  # Critical in all BUT different best moves
        'q_gaps': [],  # Mean Q-gap across configs
        'depths': [],
        'critical_depths': [],
        'inconsistent_q_gaps': [],
    }

    for state, Q_list in state_data.items():
        results['total_states'] += 1
        depth = features.depth(np.array([state]))[0]
        results['depths'].append(depth)

        # Compute Q-gap and best action for each config
        q_gaps = [get_q_gap(Q) for Q in Q_list]
        best_actions = [get_best_action(Q) for Q in Q_list]

        mean_gap = np.mean(q_gaps)
        results['q_gaps'].append(mean_gap)

        critical_in_all = all(g > Q_GAP_THRESHOLD for g in q_gaps)
        critical_in_any = any(g > Q_GAP_THRESHOLD for g in q_gaps)

        if critical_in_any:
            results['critical_in_any'] += 1

        if critical_in_all:
            results['critical_in_all'] += 1
            results['critical_depths'].append(depth)

            # Check if best move is consistent
            if len(set(best_actions)) == 1:
                results['critical_consistent'] += 1
            else:
                results['critical_inconsistent'] += 1
                results['inconsistent_q_gaps'].append(mean_gap)

    return results


def main():
    print("=" * 60)
    print("DECISION POINT CONSISTENCY ANALYSIS")
    print(f"Q-gap threshold: {Q_GAP_THRESHOLD} points")
    print("=" * 60)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds (preliminary)...")

    # Collect results
    all_results = []
    total_stats = {
        'total_states': 0,
        'critical_in_all': 0,
        'critical_in_any': 0,
        'critical_consistent': 0,
        'critical_inconsistent': 0,
    }
    all_q_gaps = []
    all_depths = []
    critical_depths = []
    inconsistent_q_gaps = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        state_data = load_common_states_for_seed(base_seed)
        if state_data is None:
            continue

        result = analyze_decision_consistency(state_data)

        for key in ['total_states', 'critical_in_all', 'critical_in_any',
                    'critical_consistent', 'critical_inconsistent']:
            total_stats[key] += result[key]

        all_q_gaps.extend(result['q_gaps'])
        all_depths.extend(result['depths'])
        critical_depths.extend(result['critical_depths'])
        inconsistent_q_gaps.extend(result['inconsistent_q_gaps'])

        # Per-seed summary
        if result['critical_in_all'] > 0:
            consistency_rate = result['critical_consistent'] / result['critical_in_all']
        else:
            consistency_rate = 1.0

        all_results.append({
            'base_seed': base_seed,
            'total_states': result['total_states'],
            'critical_in_all': result['critical_in_all'],
            'critical_in_any': result['critical_in_any'],
            'critical_consistent': result['critical_consistent'],
            'critical_inconsistent': result['critical_inconsistent'],
            'consistency_rate': consistency_rate,
            'mean_q_gap': np.mean(result['q_gaps']) if result['q_gaps'] else 0,
        })

        del state_data
        gc.collect()

    print(f"\n✓ Analyzed {len(all_results)} seeds, {total_stats['total_states']:,} common states")

    # Summary statistics
    print("\n" + "=" * 60)
    print("CRITICAL DECISION ANALYSIS")
    print("=" * 60)

    if total_stats['total_states'] > 0:
        crit_all_pct = total_stats['critical_in_all'] / total_stats['total_states'] * 100
        crit_any_pct = total_stats['critical_in_any'] / total_stats['total_states'] * 100
    else:
        crit_all_pct = crit_any_pct = 0

    print(f"\n  Total common states: {total_stats['total_states']:,}")
    print(f"  Critical in ALL configs (Q-gap>{Q_GAP_THRESHOLD}): {total_stats['critical_in_all']:,} ({crit_all_pct:.1f}%)")
    print(f"  Critical in ANY config: {total_stats['critical_in_any']:,} ({crit_any_pct:.1f}%)")

    print("\n" + "=" * 60)
    print("CONSISTENCY OF CRITICAL DECISIONS")
    print("=" * 60)

    if total_stats['critical_in_all'] > 0:
        consistent_pct = total_stats['critical_consistent'] / total_stats['critical_in_all'] * 100
        inconsistent_pct = total_stats['critical_inconsistent'] / total_stats['critical_in_all'] * 100
        print(f"\n  Critical decisions with SAME best move: {total_stats['critical_consistent']:,} ({consistent_pct:.1f}%)")
        print(f"  Critical decisions with DIFFERENT best moves: {total_stats['critical_inconsistent']:,} ({inconsistent_pct:.1f}%)")
    else:
        consistent_pct = 100.0
        inconsistent_pct = 0.0
        print("\n  No critical decisions found in all configs")

    # Q-gap analysis
    print("\n" + "=" * 60)
    print("Q-GAP ANALYSIS")
    print("=" * 60)

    if all_q_gaps:
        print(f"\n  Mean Q-gap: {np.mean(all_q_gaps):.2f}")
        print(f"  Median Q-gap: {np.median(all_q_gaps):.2f}")
        print(f"  Std Q-gap: {np.std(all_q_gaps):.2f}")
        print(f"  % with Q-gap > 5: {sum(1 for g in all_q_gaps if g > 5) / len(all_q_gaps) * 100:.1f}%")
        print(f"  % with Q-gap > 10: {sum(1 for g in all_q_gaps if g > 10) / len(all_q_gaps) * 100:.1f}%")

    if inconsistent_q_gaps:
        print(f"\n  Mean Q-gap for INCONSISTENT critical decisions: {np.mean(inconsistent_q_gaps):.2f}")

    # Depth analysis
    print("\n" + "=" * 60)
    print("CRITICAL DECISIONS BY DEPTH")
    print("=" * 60)

    if critical_depths:
        depth_counts = {}
        for d in critical_depths:
            depth_counts[d] = depth_counts.get(d, 0) + 1

        for d in sorted(depth_counts.keys()):
            count = depth_counts[d]
            pct = count / len(critical_depths) * 100
            print(f"  Depth {d:2d}: {count:5d} ({pct:5.1f}%)")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed data
    df = pd.DataFrame(all_results)
    df.to_csv(tables_dir / "11n_decision_consistency_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Summary
    summary = pd.DataFrame([{
        'n_seeds': len(all_results),
        'n_states': total_stats['total_states'],
        'q_gap_threshold': Q_GAP_THRESHOLD,
        'critical_in_all': total_stats['critical_in_all'],
        'critical_in_all_pct': crit_all_pct,
        'critical_consistent': total_stats['critical_consistent'],
        'critical_inconsistent': total_stats['critical_inconsistent'],
        'consistency_pct': consistent_pct,
        'mean_q_gap': np.mean(all_q_gaps) if all_q_gaps else 0,
    }])
    summary.to_csv(tables_dir / "11n_decision_consistency_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Q-gap distribution
    ax1 = axes[0, 0]
    ax1.hist(all_q_gaps, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=Q_GAP_THRESHOLD, color='r', linestyle='--', label=f'Threshold={Q_GAP_THRESHOLD}')
    ax1.set_xlabel('Q-gap (best - second-best)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Q-gap Distribution')
    ax1.legend()

    # Top right: Consistency breakdown
    ax2 = axes[0, 1]
    labels = ['Consistent', 'Inconsistent']
    sizes = [total_stats['critical_consistent'], total_stats['critical_inconsistent']]
    colors = ['green', 'red']
    if sum(sizes) > 0:
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Critical Decision Consistency\n(Q-gap > {Q_GAP_THRESHOLD} in all configs)')

    # Bottom left: Critical decisions by depth
    ax3 = axes[1, 0]
    if critical_depths:
        unique_depths = sorted(set(critical_depths))
        counts = [critical_depths.count(d) for d in unique_depths]
        ax3.bar(unique_depths, counts, color='blue', alpha=0.7)
    ax3.set_xlabel('Depth')
    ax3.set_ylabel('Critical Decisions')
    ax3.set_title('Critical Decisions by Game Depth')

    # Bottom right: Per-seed consistency
    ax4 = axes[1, 1]
    if df['critical_in_all'].sum() > 0:
        ax4.scatter(df['critical_in_all'], df['consistency_rate'], alpha=0.6)
        ax4.axhline(y=np.mean(df['consistency_rate']), color='r', linestyle='--',
                   label=f'Mean={np.mean(df["consistency_rate"]):.2f}')
    ax4.set_xlabel('Critical Decisions per Seed')
    ax4.set_ylabel('Consistency Rate')
    ax4.set_title('Per-Seed Consistency')
    ax4.set_ylim([0, 1.1])
    ax4.legend()

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11n_decision_consistency.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS (PRELIMINARY)")
    print("=" * 60)

    print(f"\n1. CRITICAL DECISION FREQUENCY:")
    print(f"   {crit_all_pct:.1f}% of positions are critical in ALL opponent configs")
    print(f"   {crit_any_pct:.1f}% of positions are critical in ANY config")

    print(f"\n2. CONSISTENCY OF CRITICAL DECISIONS:")
    print(f"   {consistent_pct:.1f}% of critical decisions have the SAME best move")
    print(f"   → Critical decisions are {'STABLE' if consistent_pct > 80 else 'OPPONENT-DEPENDENT'}")

    print(f"\n3. IMPLICATIONS:")
    if consistent_pct > 80:
        print(f"   When a decision matters (Q-gap>{Q_GAP_THRESHOLD}), the best move is usually clear")
        print(f"   Focus on identifying critical moments, not opponent reading")
    else:
        print(f"   Many critical decisions depend on opponent hands")
        print(f"   Opponent inference is valuable at key decision points")


if __name__ == "__main__":
    main()
