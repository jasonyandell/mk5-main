#!/usr/bin/env python3
"""
11x: Information Value (Perfect vs Imperfect)

Question: How much does knowing opponent hands help?
Method: Compare Q-values with perfect info (single config) vs imperfect info (average across configs)
What It Reveals: The value of opponent inference

Perfect info: You know which opponent holds which cards
Imperfect info: You only know your own hand (average across possible opponent distributions)
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
N_BASE_SEEDS = 50  # Preliminary
MAX_STATES_PER_SEED = 5000
np.random.seed(42)


def load_common_states_for_seed(base_seed: int, max_states: int = MAX_STATES_PER_SEED) -> dict | None:
    """Load states common to all 3 opponent configs with Q values."""
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


def get_best_action(Q: np.ndarray) -> int:
    """Get best legal action."""
    legal_mask = Q > -128
    if not legal_mask.any():
        return -1
    Q_legal = Q.astype(np.float32)
    Q_legal[~legal_mask] = -np.inf
    return int(np.argmax(Q_legal))


def compute_information_value(state_data: dict) -> dict:
    """Compute the value of information for each state.

    For each state:
    - Perfect info: Best move under each config separately
    - Imperfect info: Best move using average Q across configs

    Information value = difference in outcomes
    """
    results = {
        'states': [],
        'perfect_values': [],  # V if we knew opponent hands
        'imperfect_values': [],  # V if we play average-best
        'info_gains': [],  # Difference
        'depths': [],
        'agreement_rates': [],  # How often perfect and imperfect pick same move
    }

    for state, Q_list in state_data.items():
        depth = features.depth(np.array([state]))[0]
        results['depths'].append(depth)

        # Convert to float for averaging
        Q_float = [Q.astype(np.float32) for Q in Q_list]

        # Perfect info: Best action under each config
        perfect_actions = [get_best_action(Q) for Q in Q_float]
        perfect_values = []
        for i, Q in enumerate(Q_float):
            if perfect_actions[i] >= 0:
                perfect_values.append(Q[perfect_actions[i]])
            else:
                perfect_values.append(0)
        mean_perfect_value = np.mean(perfect_values)

        # Imperfect info: Best action under AVERAGE Q
        # Create legal mask (same across configs)
        avg_Q = np.mean(Q_float, axis=0)
        imperfect_action = get_best_action(avg_Q)

        # Value of imperfect action across configs
        imperfect_values = []
        for Q in Q_float:
            if imperfect_action >= 0 and Q[imperfect_action] > -128:
                imperfect_values.append(Q[imperfect_action])
            else:
                imperfect_values.append(0)
        mean_imperfect_value = np.mean(imperfect_values)

        # Information gain
        info_gain = mean_perfect_value - mean_imperfect_value

        results['states'].append(state)
        results['perfect_values'].append(mean_perfect_value)
        results['imperfect_values'].append(mean_imperfect_value)
        results['info_gains'].append(info_gain)

        # Agreement rate
        agrees = sum(1 for a in perfect_actions if a == imperfect_action)
        results['agreement_rates'].append(agrees / 3)

    return results


def main():
    print("=" * 60)
    print("INFORMATION VALUE ANALYSIS")
    print("Perfect vs Imperfect Information")
    print("=" * 60)

    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds (preliminary)...")

    # Collect results
    all_info_gains = []
    all_depths = []
    all_agreements = []
    per_seed_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        state_data = load_common_states_for_seed(base_seed)
        if state_data is None:
            continue

        result = compute_information_value(state_data)

        all_info_gains.extend(result['info_gains'])
        all_depths.extend(result['depths'])
        all_agreements.extend(result['agreement_rates'])

        per_seed_results.append({
            'base_seed': base_seed,
            'n_states': len(result['states']),
            'mean_info_gain': np.mean(result['info_gains']),
            'median_info_gain': np.median(result['info_gains']),
            'max_info_gain': np.max(result['info_gains']),
            'agreement_rate': np.mean(result['agreement_rates']),
        })

        del state_data
        gc.collect()

    print(f"\n✓ Analyzed {len(per_seed_results)} seeds, {len(all_info_gains):,} states")

    # Summary statistics
    print("\n" + "=" * 60)
    print("INFORMATION VALUE SUMMARY")
    print("=" * 60)

    info_gains = np.array(all_info_gains)
    print(f"\n  Mean information gain: {np.mean(info_gains):.2f} points")
    print(f"  Median information gain: {np.median(info_gains):.2f} points")
    print(f"  Std information gain: {np.std(info_gains):.2f} points")
    print(f"  Max information gain: {np.max(info_gains):.2f} points")

    # What percentage of states benefit from info
    pct_benefit = np.mean(info_gains > 0) * 100
    pct_significant = np.mean(info_gains > 2) * 100
    pct_large = np.mean(info_gains > 5) * 100

    print(f"\n  States where perfect info helps:")
    print(f"    Any benefit (>0): {pct_benefit:.1f}%")
    print(f"    Significant (>2 pts): {pct_significant:.1f}%")
    print(f"    Large (>5 pts): {pct_large:.1f}%")

    # Agreement rate
    print(f"\n  Perfect/Imperfect action agreement: {np.mean(all_agreements)*100:.1f}%")

    # By depth
    print("\n" + "=" * 60)
    print("INFORMATION VALUE BY DEPTH")
    print("=" * 60)

    depth_gains = {}
    for d, g in zip(all_depths, all_info_gains):
        if d not in depth_gains:
            depth_gains[d] = []
        depth_gains[d].append(g)

    for d in sorted(depth_gains.keys()):
        gains = depth_gains[d]
        print(f"  Depth {d:2d}: mean={np.mean(gains):+5.2f}, n={len(gains):,}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed data
    df = pd.DataFrame(per_seed_results)
    df.to_csv(tables_dir / "11x_information_value_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Summary
    summary = pd.DataFrame([{
        'n_seeds': len(per_seed_results),
        'n_states': len(all_info_gains),
        'mean_info_gain': np.mean(info_gains),
        'median_info_gain': np.median(info_gains),
        'max_info_gain': np.max(info_gains),
        'pct_benefit_any': pct_benefit,
        'pct_benefit_significant': pct_significant,
        'pct_benefit_large': pct_large,
        'action_agreement_rate': np.mean(all_agreements) * 100,
    }])
    summary.to_csv(tables_dir / "11x_information_value_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Information gain distribution
    ax1 = axes[0, 0]
    ax1.hist(info_gains, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=np.mean(info_gains), color='r', linestyle='--', label=f'Mean={np.mean(info_gains):.2f}')
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=2)
    ax1.set_xlabel('Information Gain (points)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Value of Knowing Opponent Hands')
    ax1.legend()

    # Top right: Information gain by depth
    ax2 = axes[0, 1]
    depths_sorted = sorted(depth_gains.keys())
    means = [np.mean(depth_gains[d]) for d in depths_sorted]
    ax2.bar(depths_sorted, means, color='blue', alpha=0.7)
    ax2.set_xlabel('Depth')
    ax2.set_ylabel('Mean Information Gain (points)')
    ax2.set_title('Information Value by Game Phase')
    ax2.axhline(y=0, color='k', linestyle='-')

    # Bottom left: Agreement rate vs info gain
    ax3 = axes[1, 0]
    ax3.scatter(df['agreement_rate'], df['mean_info_gain'], alpha=0.6)
    ax3.set_xlabel('Action Agreement Rate')
    ax3.set_ylabel('Mean Information Gain')
    ax3.set_title('Agreement vs Information Value')

    # Bottom right: Cumulative distribution
    ax4 = axes[1, 1]
    sorted_gains = np.sort(info_gains)
    cumulative = np.arange(1, len(sorted_gains) + 1) / len(sorted_gains)
    ax4.plot(sorted_gains, cumulative)
    ax4.axvline(x=0, color='k', linestyle='--', label='No benefit')
    ax4.axvline(x=2, color='r', linestyle='--', label='2pt threshold')
    ax4.axvline(x=5, color='g', linestyle='--', label='5pt threshold')
    ax4.set_xlabel('Information Gain')
    ax4.set_ylabel('Cumulative Fraction')
    ax4.set_title('Cumulative Distribution of Information Value')
    ax4.legend()

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11x_information_value.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS (PRELIMINARY)")
    print("=" * 60)

    print(f"\n1. INFORMATION VALUE:")
    print(f"   Average: {np.mean(info_gains):.2f} points")
    print(f"   → Knowing opponent hands gains ~{np.mean(info_gains):.1f} points on average")

    print(f"\n2. HOW OFTEN DOES IT MATTER?")
    print(f"   {pct_benefit:.0f}% of positions benefit from perfect info")
    print(f"   {pct_significant:.0f}% gain >2 points")
    print(f"   {pct_large:.0f}% gain >5 points")

    print(f"\n3. PRACTICAL IMPLICATION:")
    agreement = np.mean(all_agreements) * 100
    print(f"   Action agreement: {agreement:.0f}%")
    if agreement > 70:
        print(f"   → Most of the time, you'd make the same move anyway")
        print(f"   → Opponent inference provides marginal improvement")
    else:
        print(f"   → Opponent hands significantly affect optimal play")
        print(f"   → Inference skills are valuable")


if __name__ == "__main__":
    main()
