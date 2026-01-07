#!/usr/bin/env python3
"""
11z: Partner Inference (MI) Analysis

Question: Does partner's play reveal their hand?
Method: MI(partner_actions; partner_hand)
What It Reveals: Signaling potential

Approach:
- For each base_seed, P2 (partner) has different hands in different configs
- Find states where P2 acts that are common across configs
- Measure how much P2's action varies with their hand
- Higher variance = actions reveal more about hand (higher MI)
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import entropy

from forge.analysis.utils import features
from forge.oracle.rng import deal_from_seed, deal_with_fixed_p0
from forge.oracle import schema

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Full analysis
MAX_ROWS = 20_000_000  # Increased to allow larger files
np.random.seed(42)


def get_best_move(q_values: np.ndarray) -> np.ndarray:
    """Get argmax(Q) for each state, handling illegals."""
    ILLEGAL = -128
    q_for_max = np.where(q_values != ILLEGAL, q_values, -1000)
    return np.argmax(q_for_max, axis=1)


def get_hand_features(hand: list[int], trump: int) -> dict:
    """Extract features from a hand."""
    hand_pips = [schema.domino_pips(d) for d in hand]

    trump_count = sum(1 for d in hand_pips if d[0] == trump or d[1] == trump)
    doubles = sum(1 for d in hand_pips if d[0] == d[1])
    high_pips = sum(max(d[0], d[1]) for d in hand_pips)
    total_pips = sum(d[0] + d[1] for d in hand_pips)

    # Count domino holdings
    count_ids = set(features.COUNT_DOMINO_IDS)
    count_dominoes = sum(1 for d in hand if d in count_ids)
    count_points = sum(
        features.COUNT_DOMINO_POINTS.get(d, 0)
        for d in hand
    )

    return {
        'trump_count': trump_count,
        'doubles': doubles,
        'high_pips': high_pips,
        'total_pips': total_pips,
        'count_dominoes': count_dominoes,
        'count_points': count_points,
    }


def analyze_partner_inference(base_seed: int) -> dict | None:
    """Analyze partner inference for one base seed.

    Uses pairwise comparisons since 3-way common states are rare due to path divergence.
    """
    decl_id = base_seed % 10
    p0_hand = deal_from_seed(base_seed)[0]

    # Get P2's hand in each config
    p2_hands = []
    p2_features = []
    for opp_seed in range(3):
        hands = deal_with_fixed_p0(p0_hand, opp_seed)
        p2_hands.append(hands[2])  # P2 is partner
        p2_features.append(get_hand_features(hands[2], decl_id))

    # First pass: find P2 states and best moves in each config
    config_data = []
    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        try:
            table = pq.read_table(path, columns=['state', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'])
            if table.num_rows > MAX_ROWS:
                del table
                gc.collect()
                return None

            states = table.column('state').to_numpy()
            q_cols = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
            q_values = np.column_stack([table.column(c).to_numpy() for c in q_cols]).astype(np.int16)
            del table
            gc.collect()

            # Filter to P2 states (player == 2)
            players = features.player(states)
            p2_mask = players == 2
            p2_states = states[p2_mask]
            p2_q = q_values[p2_mask]

            best_moves = get_best_move(p2_q)
            state_to_best = dict(zip(p2_states, best_moves))
            config_data.append({'states': set(p2_states), 'best': state_to_best})

            del states, q_values, p2_states, p2_q
            gc.collect()

        except Exception:
            gc.collect()
            return None

    # Compute pairwise consistency (more robust than 3-way)
    pairwise_consistent = 0
    pairwise_varied = 0
    all_entropies = []

    for i in range(3):
        for j in range(i + 1, 3):
            common = config_data[i]['states'] & config_data[j]['states']
            for state in common:
                a_i = config_data[i]['best'].get(state)
                a_j = config_data[j]['best'].get(state)
                if a_i is not None and a_j is not None:
                    if a_i == a_j:
                        pairwise_consistent += 1
                    else:
                        pairwise_varied += 1

    # Also compute 3-way stats if available
    common_3way = config_data[0]['states'] & config_data[1]['states'] & config_data[2]['states']
    n_3way = len(common_3way)
    n_consistent_3way = 0
    n_varied_3way = 0

    for state in common_3way:
        actions = [config_data[k]['best'].get(state) for k in range(3)]
        if all(a is not None for a in actions):
            unique = len(set(actions))
            if unique == 1:
                n_consistent_3way += 1
            else:
                n_varied_3way += 1
            # Action entropy
            action_counts = np.bincount(actions, minlength=7)
            action_probs = action_counts / action_counts.sum()
            all_entropies.append(entropy(action_probs + 1e-10))

    del config_data
    gc.collect()

    total_pairwise = pairwise_consistent + pairwise_varied
    if total_pairwise == 0:
        return None

    # P2 hand variance across configs
    p2_trump_counts = [f['trump_count'] for f in p2_features]
    p2_doubles = [f['doubles'] for f in p2_features]
    p2_total_pips = [f['total_pips'] for f in p2_features]
    p2_count_pts = [f['count_points'] for f in p2_features]

    return {
        'base_seed': base_seed,
        'n_pairwise_comparisons': total_pairwise,
        'n_3way_common': n_3way,
        'pairwise_consistent': pairwise_consistent,
        'pairwise_varied': pairwise_varied,
        'consistency_rate': pairwise_consistent / total_pairwise,
        'n_consistent_3way': n_consistent_3way,
        'n_varied_3way': n_varied_3way,
        'mean_action_entropy': np.mean(all_entropies) if all_entropies else 0,
        'p2_trump_std': np.std(p2_trump_counts),
        'p2_doubles_std': np.std(p2_doubles),
        'p2_pips_std': np.std(p2_total_pips),
        'p2_count_std': np.std(p2_count_pts),
    }


def compute_mi_proxy(df: pd.DataFrame) -> dict:
    """Compute mutual information proxy metrics."""
    # Average action entropy across hands
    # Higher entropy = more action variety = more information revealed
    mean_entropy = df['mean_action_entropy'].mean()

    # Correlation between hand variance and action variance
    # If hand variance → action variance, then actions reveal hand info
    hand_variance = df['p2_trump_std'] + df['p2_doubles_std']
    action_variance = 1 - df['consistency_rate']

    if hand_variance.std() > 0 and action_variance.std() > 0:
        corr = np.corrcoef(hand_variance, action_variance)[0, 1]
    else:
        corr = 0.0

    return {
        'mean_action_entropy': mean_entropy,
        'hand_action_variance_corr': corr,
    }


def main():
    print("=" * 60)
    print("PARTNER INFERENCE (MI) ANALYSIS")
    print("Does partner's play reveal their hand?")
    print("=" * 60)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    # Collect results
    all_results = []
    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_partner_inference(base_seed)
        if result:
            all_results.append(result)

    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("PARTNER ACTION CONSISTENCY")
    print("=" * 60)

    print(f"\n  Pairwise comparisons per hand:")
    print(f"    Mean: {df['n_pairwise_comparisons'].mean():.0f}")
    print(f"    Median: {df['n_pairwise_comparisons'].median():.0f}")

    print(f"\n  3-way common states per hand:")
    print(f"    Mean: {df['n_3way_common'].mean():.0f}")
    print(f"    Median: {df['n_3way_common'].median():.0f}")

    print(f"\n  Action consistency rate (pairwise):")
    print(f"    Mean: {df['consistency_rate'].mean()*100:.1f}%")
    print(f"    Median: {df['consistency_rate'].median()*100:.1f}%")
    print(f"    Std: {df['consistency_rate'].std()*100:.1f}%")

    print(f"\n  Mean action entropy:")
    print(f"    Mean: {df['mean_action_entropy'].mean():.3f}")
    print(f"    Max theoretical (3 configs): {np.log(3):.3f}")

    # MI proxy
    print("\n" + "=" * 60)
    print("MUTUAL INFORMATION PROXY")
    print("=" * 60)

    mi_metrics = compute_mi_proxy(df)
    print(f"\n  Action entropy (proxy for information revealed):")
    print(f"    {mi_metrics['mean_action_entropy']:.3f} (higher = more revealing)")

    print(f"\n  Hand-action variance correlation:")
    print(f"    {mi_metrics['hand_action_variance_corr']:+.3f}")

    # Breakdown by consistency
    print("\n" + "=" * 60)
    print("CONSISTENCY BREAKDOWN")
    print("=" * 60)

    high_consistency = df[df['consistency_rate'] > 0.9]
    med_consistency = df[(df['consistency_rate'] > 0.5) & (df['consistency_rate'] <= 0.9)]
    low_consistency = df[df['consistency_rate'] <= 0.5]

    print(f"\n  High consistency (>90%): {len(high_consistency)} ({len(high_consistency)/len(df)*100:.1f}%)")
    print(f"  Medium consistency (50-90%): {len(med_consistency)} ({len(med_consistency)/len(df)*100:.1f}%)")
    print(f"  Low consistency (≤50%): {len(low_consistency)} ({len(low_consistency)/len(df)*100:.1f}%)")

    # Correlations
    print("\n" + "=" * 60)
    print("CORRELATIONS WITH HAND VARIANCE")
    print("=" * 60)

    print(f"\n  Consistency vs P2 trump std: {df['consistency_rate'].corr(df['p2_trump_std']):+.3f}")
    print(f"  Consistency vs P2 doubles std: {df['consistency_rate'].corr(df['p2_doubles_std']):+.3f}")
    print(f"  Consistency vs P2 pips std: {df['consistency_rate'].corr(df['p2_pips_std']):+.3f}")
    print(f"  Consistency vs P2 count std: {df['consistency_rate'].corr(df['p2_count_std']):+.3f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11z_partner_inference_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Summary
    summary = pd.DataFrame([{
        'n_hands': len(df),
        'mean_pairwise_comparisons': df['n_pairwise_comparisons'].mean(),
        'mean_3way_common': df['n_3way_common'].mean(),
        'mean_consistency_rate': df['consistency_rate'].mean(),
        'median_consistency_rate': df['consistency_rate'].median(),
        'mean_action_entropy': mi_metrics['mean_action_entropy'],
        'hand_action_corr': mi_metrics['hand_action_variance_corr'],
        'high_consistency_pct': len(high_consistency) / len(df) * 100,
        'med_consistency_pct': len(med_consistency) / len(df) * 100,
        'low_consistency_pct': len(low_consistency) / len(df) * 100,
    }])
    summary.to_csv(tables_dir / "11z_partner_inference_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Consistency rate distribution
    ax1 = axes[0, 0]
    ax1.hist(df['consistency_rate'] * 100, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(df['consistency_rate'].median() * 100, color='red', linestyle='--',
                label=f'Median: {df["consistency_rate"].median()*100:.1f}%')
    ax1.set_xlabel('Action Consistency Rate (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title("P2's Action Consistency Across Opponent Configs")
    ax1.legend()

    # Top right: Action entropy distribution
    ax2 = axes[0, 1]
    ax2.hist(df['mean_action_entropy'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(df['mean_action_entropy'].median(), color='red', linestyle='--',
                label=f'Median: {df["mean_action_entropy"].median():.3f}')
    ax2.set_xlabel('Mean Action Entropy')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Action Entropy (Higher = More Variation)')
    ax2.legend()

    # Bottom left: Consistency vs P2 hand variance
    ax3 = axes[1, 0]
    hand_var = df['p2_trump_std'] + df['p2_doubles_std']
    ax3.scatter(hand_var, df['consistency_rate'] * 100, alpha=0.6, s=40)
    ax3.set_xlabel('P2 Hand Variance (trump_std + doubles_std)')
    ax3.set_ylabel('Action Consistency Rate (%)')
    corr = hand_var.corr(df['consistency_rate'])
    ax3.set_title(f'Consistency vs Hand Variance (r={corr:.2f})')

    # Bottom right: Pairwise comparisons vs consistency
    ax4 = axes[1, 1]
    ax4.scatter(df['n_pairwise_comparisons'], df['consistency_rate'] * 100, alpha=0.6, s=40)
    ax4.set_xlabel('Number of Pairwise Comparisons')
    ax4.set_ylabel('Action Consistency Rate (%)')
    corr = df['n_pairwise_comparisons'].corr(df['consistency_rate'])
    ax4.set_title(f'Sample Size vs Consistency (r={corr:.2f})')

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11z_partner_inference.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. ACTION CONSISTENCY:")
    print(f"   {df['consistency_rate'].mean()*100:.1f}% of P2's actions are consistent across configs")
    var_pct = 100 - df['consistency_rate'].mean()*100
    print(f"   {var_pct:.1f}% of actions vary with P2's hand (reveal information)")

    print(f"\n2. SIGNALING POTENTIAL (MI Proxy):")
    print(f"   Action entropy: {mi_metrics['mean_action_entropy']:.3f}")
    if mi_metrics['mean_action_entropy'] > 0.3:
        print(f"   → HIGH signaling potential: actions strongly reveal hand info")
    elif mi_metrics['mean_action_entropy'] > 0.1:
        print(f"   → MODERATE signaling potential")
    else:
        print(f"   → LOW signaling potential: actions mostly determined by game state")

    print(f"\n3. PARTNER INFERENCE VALUE:")
    if var_pct > 30:
        print(f"   With {var_pct:.0f}% action variance, observing partner can improve play")
        print(f"   → Partner inference is VALUABLE for imperfect-info play")
    elif var_pct > 10:
        print(f"   Moderate action variance ({var_pct:.0f}%) - some inference value")
    else:
        print(f"   Low action variance ({var_pct:.0f}%) - partner actions largely predictable")
        print(f"   → Partner inference has LIMITED value")


if __name__ == "__main__":
    main()
