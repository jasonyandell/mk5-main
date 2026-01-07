#!/usr/bin/env python3
"""
11e: Contest State Distribution Analysis

Question: What's P(team0 captures) for each count domino?
Method: 5-vector of probabilities per hand, using V as proxy for count capture
What It Reveals: The imperfect-info manifold coordinates

Uses marginalized data where P0's hand is fixed but opponents' cards vary.

Approach:
- For each base_seed, reconstruct full deal for all 3 opponent configs
- Track which counts are held by Team 0 (P0+P2) vs Team 1 (P1+P3)
- Use V distribution to estimate capture probabilities
- High V + Team 0 holds count → likely captured
- Low V + Team 0 holds count → possibly lost to opponents
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

from forge.analysis.utils import features
from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed, deal_with_fixed_p0

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Analyze all available base seeds
np.random.seed(42)

# Count domino IDs and their point values
COUNT_DOMINO_IDS = features.COUNT_DOMINO_IDS  # [6, 7, 10, 27, 21]
COUNT_NAMES = ['3-2', '4-1', '5-0', '5-5', '6-4']
COUNT_POINTS = [5, 5, 5, 10, 10]


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


def get_team_count_ownership(hands: list[list[int]]) -> dict[int, int]:
    """Determine which team holds each count domino.

    Returns: {count_domino_id: team} where team is 0 or 1
    """
    ownership = {}
    for count_id in COUNT_DOMINO_IDS:
        # Check each player's hand
        for player_id, hand in enumerate(hands):
            if count_id in hand:
                team = player_id % 2  # P0,P2 = Team 0; P1,P3 = Team 1
                ownership[count_id] = team
                break
    return ownership


def analyze_contest_for_base_seed(base_seed: int) -> dict | None:
    """Analyze count capture probabilities for one base seed.

    Returns metrics about count ownership and V correlation.
    """
    decl_id = base_seed % 10
    p0_hand = deal_from_seed(base_seed)[0]

    # Collect data across 3 opponent configs
    configs = []
    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        V = get_root_v_fast(path)
        if V is None:
            continue

        hands = deal_with_fixed_p0(p0_hand, opp_seed)
        ownership = get_team_count_ownership(hands)

        configs.append({
            'opp_seed': opp_seed,
            'V': V,
            'ownership': ownership
        })

    if len(configs) != 3:
        return None

    # Analyze each count domino
    count_analysis = {}
    V_values = [c['V'] for c in configs]
    V_mean = np.mean(V_values)
    V_std = np.std(V_values)

    for count_id, count_name, points in zip(COUNT_DOMINO_IDS, COUNT_NAMES, COUNT_POINTS):
        # Track ownership across configs
        team0_holds = []  # 1 if Team 0 holds this count, 0 otherwise
        team0_holds_configs = []  # Which configs have Team 0 holding this count

        for c in configs:
            owner = c['ownership'].get(count_id, -1)
            if owner == 0:
                team0_holds.append(1)
                team0_holds_configs.append(c['opp_seed'])
            else:
                team0_holds.append(0)

        # Calculate P(Team 0 holds this count)
        p_team0_holds = np.mean(team0_holds)

        # Calculate mean V when Team 0 holds vs doesn't hold
        v_when_holds = [c['V'] for c in configs if c['ownership'].get(count_id) == 0]
        v_when_not = [c['V'] for c in configs if c['ownership'].get(count_id) != 0]

        v_diff = None
        if v_when_holds and v_when_not:
            v_diff = np.mean(v_when_holds) - np.mean(v_when_not)

        count_analysis[count_name] = {
            'p_team0_holds': p_team0_holds,
            'n_team0_holds': sum(team0_holds),
            'v_when_team0_holds': np.mean(v_when_holds) if v_when_holds else None,
            'v_when_team1_holds': np.mean(v_when_not) if v_when_not else None,
            'v_diff': v_diff,
            'points': points
        }

    # Estimate capture probabilities based on V and ownership
    # Heuristic: If V > 0 and Team 0 holds count, high P(capture)
    # If V < 0 and Team 0 holds count, lower P(capture) - might have been taken

    # For counts held by Team 0:
    # P(capture) ≈ (V + 42) / 84 scaled by ownership
    # This is a rough proxy since V ranges from -42 to +42

    capture_probs = {}
    for count_name in COUNT_NAMES:
        analysis = count_analysis[count_name]
        if analysis['v_when_team0_holds'] is not None:
            # Team 0 sometimes holds this count
            v = analysis['v_when_team0_holds']
            # Higher V → more likely Team 0 captured counts
            # Scale to [0, 1] range
            p_capture_given_hold = np.clip((v + 42) / 84, 0.1, 0.9)

            # Overall P(team0 captures) = P(holds) * P(captures|holds) + P(not holds) * P(captures|not holds)
            # P(captures|not holds) is low but not zero - could steal from opponent
            p_capture_given_not_hold = np.clip((-v + 42) / 84 * 0.2, 0.0, 0.3)  # Reduced probability

            p_capture = (analysis['p_team0_holds'] * p_capture_given_hold +
                        (1 - analysis['p_team0_holds']) * p_capture_given_not_hold)
        else:
            # Team 0 never holds this count
            v = analysis['v_when_team1_holds'] if analysis['v_when_team1_holds'] is not None else V_mean
            # Lower V → opponents kept their counts
            p_capture = np.clip((-v + 42) / 84 * 0.2, 0.0, 0.3)

        capture_probs[count_name] = p_capture

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'V_mean': V_mean,
        'V_std': V_std,
        'V_spread': max(V_values) - min(V_values),
        'count_analysis': count_analysis,
        'capture_probs': capture_probs
    }


def main():
    print("=" * 60)
    print("CONTEST STATE DISTRIBUTION ANALYSIS")
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

    # Aggregate results
    all_results = []
    count_vectors = []  # 5-vectors of capture probabilities

    # Aggregate count ownership statistics
    ownership_stats = {name: {'team0_count': 0, 'total': 0, 'v_sum_team0': 0, 'v_sum_team1': 0,
                              'n_team0': 0, 'n_team1': 0} for name in COUNT_NAMES}

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_contest_for_base_seed(base_seed)
        if result:
            all_results.append({
                'base_seed': result['base_seed'],
                'decl_id': result['decl_id'],
                'V_mean': result['V_mean'],
                'V_std': result['V_std'],
                'V_spread': result['V_spread'],
                **{f'p_capture_{name}': result['capture_probs'][name] for name in COUNT_NAMES}
            })

            # Build 5-vector
            vector = [result['capture_probs'][name] for name in COUNT_NAMES]
            count_vectors.append(vector)

            # Aggregate ownership stats
            for name in COUNT_NAMES:
                analysis = result['count_analysis'][name]
                ownership_stats[name]['total'] += 3  # 3 configs per base_seed
                ownership_stats[name]['team0_count'] += analysis['n_team0_holds']
                if analysis['v_when_team0_holds'] is not None:
                    ownership_stats[name]['v_sum_team0'] += analysis['v_when_team0_holds'] * analysis['n_team0_holds']
                    ownership_stats[name]['n_team0'] += analysis['n_team0_holds']
                if analysis['v_when_team1_holds'] is not None:
                    n_team1 = 3 - analysis['n_team0_holds']
                    ownership_stats[name]['v_sum_team1'] += analysis['v_when_team1_holds'] * n_team1
                    ownership_stats[name]['n_team1'] += n_team1

    print(f"\n✓ Analyzed {len(all_results)} base seeds")

    # Build DataFrames
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    print(f"\nTotal hands analyzed: {len(all_results)}")
    print(f"Mean V across hands: {df['V_mean'].mean():.2f}")
    print(f"Mean V spread: {df['V_spread'].mean():.2f}")

    # Count ownership summary
    print("\n" + "=" * 60)
    print("COUNT DOMINO OWNERSHIP")
    print("=" * 60)

    ownership_data = []
    for name in COUNT_NAMES:
        stats = ownership_stats[name]
        p_team0_holds = stats['team0_count'] / stats['total'] if stats['total'] > 0 else 0
        v_team0 = stats['v_sum_team0'] / stats['n_team0'] if stats['n_team0'] > 0 else np.nan
        v_team1 = stats['v_sum_team1'] / stats['n_team1'] if stats['n_team1'] > 0 else np.nan

        ownership_data.append({
            'count': name,
            'p_team0_holds': p_team0_holds,
            'n_team0': stats['n_team0'],
            'n_team1': stats['n_team1'],
            'v_when_team0_holds': v_team0,
            'v_when_team1_holds': v_team1,
            'v_diff': v_team0 - v_team1 if not np.isnan(v_team0) and not np.isnan(v_team1) else np.nan
        })

    ownership_df = pd.DataFrame(ownership_data)
    print("\nOwnership and V correlation:")
    print(ownership_df.to_string(index=False))

    # 5-vector statistics
    print("\n" + "=" * 60)
    print("CAPTURE PROBABILITY 5-VECTORS")
    print("=" * 60)

    count_vectors_arr = np.array(count_vectors)

    vector_data = []
    for i, name in enumerate(COUNT_NAMES):
        probs = count_vectors_arr[:, i]
        vector_data.append({
            'count': name,
            'mean_p_capture': np.mean(probs),
            'std_p_capture': np.std(probs),
            'min_p_capture': np.min(probs),
            'max_p_capture': np.max(probs),
            'median_p_capture': np.median(probs)
        })

    vector_df = pd.DataFrame(vector_data)
    print("\nCapture probability statistics:")
    print(vector_df.to_string(index=False))

    # Correlation between capture probabilities
    print("\n" + "=" * 60)
    print("CAPTURE PROBABILITY CORRELATIONS")
    print("=" * 60)

    corr_matrix = np.corrcoef(count_vectors_arr.T)
    corr_df = pd.DataFrame(corr_matrix, index=COUNT_NAMES, columns=COUNT_NAMES)
    print("\nCorrelation matrix:")
    print(corr_df.round(3).to_string())

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Save per-seed results with 5-vectors
    df.to_csv(tables_dir / "11e_contest_state_by_seed.csv", index=False)
    print("✓ Saved per-seed results")

    # Save ownership statistics
    ownership_df.to_csv(tables_dir / "11e_count_ownership.csv", index=False)
    print("✓ Saved ownership statistics")

    # Save 5-vector statistics
    vector_df.to_csv(tables_dir / "11e_capture_probabilities.csv", index=False)
    print("✓ Saved capture probability statistics")

    # Save correlation matrix
    corr_df.to_csv(tables_dir / "11e_capture_correlations.csv")
    print("✓ Saved correlation matrix")

    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Capture probabilities by count
    ax1 = axes[0, 0]
    x = range(len(COUNT_NAMES))
    means = [vector_df.loc[i, 'mean_p_capture'] for i in range(len(COUNT_NAMES))]
    stds = [vector_df.loc[i, 'std_p_capture'] for i in range(len(COUNT_NAMES))]
    ax1.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(COUNT_NAMES)
    ax1.set_xlabel('Count Domino')
    ax1.set_ylabel('P(Team 0 Captures)')
    ax1.set_title('Estimated Capture Probabilities by Count')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
    ax1.legend()

    # Top right: V difference when Team 0 vs Team 1 holds count
    ax2 = axes[0, 1]
    v_diffs = ownership_df['v_diff'].values
    colors = ['green' if v > 0 else 'red' for v in v_diffs]
    ax2.bar(x, v_diffs, color=colors, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(COUNT_NAMES)
    ax2.set_xlabel('Count Domino')
    ax2.set_ylabel('V Difference (Team0 holds - Team1 holds)')
    ax2.set_title('Expected Value Difference by Count Ownership')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Bottom left: Distribution of capture probability vectors
    ax3 = axes[1, 0]
    for i, name in enumerate(COUNT_NAMES):
        probs = count_vectors_arr[:, i]
        ax3.hist(probs, bins=20, alpha=0.5, label=name)
    ax3.set_xlabel('P(Team 0 Captures)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Capture Probabilities')
    ax3.legend()

    # Bottom right: Correlation heatmap
    ax4 = axes[1, 1]
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(COUNT_NAMES)))
    ax4.set_yticks(range(len(COUNT_NAMES)))
    ax4.set_xticklabels(COUNT_NAMES)
    ax4.set_yticklabels(COUNT_NAMES)
    ax4.set_title('Capture Probability Correlations')
    for i in range(len(COUNT_NAMES)):
        for j in range(len(COUNT_NAMES)):
            ax4.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11e_contest_state_distribution.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\n1. OWNERSHIP EFFECT ON V:")
    for _, row in ownership_df.iterrows():
        if not np.isnan(row['v_diff']):
            effect = "better" if row['v_diff'] > 0 else "worse"
            print(f"   {row['count']}: Team 0 is {abs(row['v_diff']):.1f} points {effect} when holding")

    print("\n2. CAPTURE PROBABILITY ESTIMATES:")
    for _, row in vector_df.iterrows():
        print(f"   {row['count']}: P(Team 0 captures) = {row['mean_p_capture']:.2f} ± {row['std_p_capture']:.2f}")

    # Most contested vs most controlled
    mean_captures = vector_df['mean_p_capture'].values
    most_controlled = COUNT_NAMES[np.argmax(np.abs(mean_captures - 0.5))]
    most_contested = COUNT_NAMES[np.argmin(np.abs(mean_captures - 0.5))]

    print(f"\n3. CONTROL STATUS:")
    print(f"   Most controlled: {most_controlled} (furthest from 50/50)")
    print(f"   Most contested: {most_contested} (closest to 50/50)")

    # Correlation insights
    print("\n4. COUNT CORRELATIONS:")
    for i in range(len(COUNT_NAMES)):
        for j in range(i+1, len(COUNT_NAMES)):
            corr = corr_matrix[i, j]
            if abs(corr) > 0.3:
                print(f"   {COUNT_NAMES[i]} ↔ {COUNT_NAMES[j]}: {corr:.2f}")


if __name__ == "__main__":
    main()
