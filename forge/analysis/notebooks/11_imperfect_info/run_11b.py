#!/usr/bin/env python3
"""
11b: V Distribution Per Hand Analysis

Compute E[V] and σ(V) for hands across many seeds.
Since there are 1.18M possible hands, we analyze hand FEATURES
rather than individual hands.
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from forge.analysis.utils import loading, features, navigation
from forge.oracle import schema, tables

DATA_DIR = "/mnt/d/shards-standard/"
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_SHARDS = 200  # More shards for better statistics


def extract_hand_features(hand: list[int], decl_id: int) -> dict:
    """
    Extract features from a hand for analysis.

    Args:
        hand: List of 7 domino IDs
        decl_id: Declaration type

    Returns:
        Dict of features
    """
    # Count points in hand
    count_points = sum(tables.DOMINO_COUNT_POINTS[d] for d in hand)

    # Count doubles
    n_doubles = sum(1 for d in hand if tables.DOMINO_IS_DOUBLE[d])

    # Count trump (dominoes in called suit)
    n_trump = sum(1 for d in hand if tables.is_in_called_suit(d, decl_id))

    # High pip count (sum of high pips)
    high_pips = sum(tables.DOMINO_HIGH[d] for d in hand)

    # Total pip sum
    total_pips = sum(tables.DOMINO_SUM[d] for d in hand)

    # Count per suit (pip 0-6)
    suit_counts = [0] * 7
    for d in hand:
        h, l = tables.DOMINO_HIGH[d], tables.DOMINO_LOW[d]
        suit_counts[h] += 1
        if h != l:
            suit_counts[l] += 1

    # Longest suit
    longest_suit = max(suit_counts)

    # Number of suits represented
    n_suits = sum(1 for c in suit_counts if c > 0)

    return {
        'count_points': count_points,
        'n_doubles': n_doubles,
        'n_trump': n_trump,
        'high_pips': high_pips,
        'total_pips': total_pips,
        'longest_suit': longest_suit,
        'n_suits': n_suits,
    }


def analyze_hand_v(shard_path):
    """Analyze initial V for a single shard."""
    try:
        df, seed, decl_id = schema.load_file(shard_path)

        # Skip large shards
        if len(df) > 30_000_000:
            del df
            gc.collect()
            return None

        # Find initial state (depth = 28)
        depths = features.depth(df['state'].values)
        initial_mask = depths == 28

        if not initial_mask.any():
            del df
            gc.collect()
            return None

        initial_V = df.loc[initial_mask, 'V'].values[0]

        # Get hands for all 4 players
        hands = schema.deal_from_seed(seed)

        # Extract features for each player's hand
        results = []
        for player in range(4):
            hand = hands[player]
            team = player % 2

            # Adjust V for player's perspective
            # V is from Team 0's perspective
            player_V = initial_V if team == 0 else -initial_V

            hand_feats = extract_hand_features(hand, decl_id)
            hand_feats['seed'] = seed
            hand_feats['decl_id'] = decl_id
            hand_feats['player'] = player
            hand_feats['team'] = team
            hand_feats['V'] = player_V
            hand_feats['hand'] = tuple(sorted(hand))

            results.append(hand_feats)

        del df
        gc.collect()

        return results

    except Exception as e:
        print(f"Error: {e}")
        gc.collect()
        return None


def main():
    print("=" * 60)
    print("V DISTRIBUTION PER HAND ANALYSIS")
    print("=" * 60)

    # Find shards
    shard_files = loading.find_shard_files(DATA_DIR)
    print(f"Total shards available: {len(shard_files)}")
    sample_files = shard_files[:N_SHARDS]
    print(f"Analyzing {len(sample_files)} shards...")

    # Collect data
    all_results = []
    for path in tqdm(sample_files, desc="Processing"):
        results = analyze_hand_v(path)
        if results:
            all_results.extend(results)

    print(f"\n✓ Collected {len(all_results)} hand-V pairs")

    # Build DataFrame
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)

    print(f"\nOverall V distribution:")
    print(f"  Mean V: {df['V'].mean():.2f}")
    print(f"  Std V:  {df['V'].std():.2f}")
    print(f"  Min V:  {df['V'].min()}")
    print(f"  Max V:  {df['V'].max()}")

    # Group by hand features and compute statistics
    print("\n" + "=" * 60)
    print("V BY HAND FEATURES")
    print("=" * 60)

    # By count points held
    print("\n1. V by Count Points Held:")
    count_groups = df.groupby('count_points').agg({
        'V': ['mean', 'std', 'count']
    }).round(2)
    count_groups.columns = ['E[V]', 'σ(V)', 'n']
    print(count_groups.to_string())

    # By trump count
    print("\n2. V by Trump Count:")
    trump_groups = df.groupby('n_trump').agg({
        'V': ['mean', 'std', 'count']
    }).round(2)
    trump_groups.columns = ['E[V]', 'σ(V)', 'n']
    print(trump_groups.to_string())

    # By doubles count
    print("\n3. V by Doubles Count:")
    doubles_groups = df.groupby('n_doubles').agg({
        'V': ['mean', 'std', 'count']
    }).round(2)
    doubles_groups.columns = ['E[V]', 'σ(V)', 'n']
    print(doubles_groups.to_string())

    # By longest suit
    print("\n4. V by Longest Suit:")
    suit_groups = df.groupby('longest_suit').agg({
        'V': ['mean', 'std', 'count']
    }).round(2)
    suit_groups.columns = ['E[V]', 'σ(V)', 'n']
    print(suit_groups.to_string())

    # Correlation analysis
    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS WITH V")
    print("=" * 60)

    feature_cols = ['count_points', 'n_trump', 'n_doubles', 'high_pips',
                    'total_pips', 'longest_suit', 'n_suits']
    correlations = df[feature_cols + ['V']].corr()['V'].drop('V').sort_values(ascending=False)
    print("\nCorrelation with V:")
    for feat, corr in correlations.items():
        print(f"  {feat:15s}: {corr:+.3f}")

    # Multi-feature analysis
    print("\n" + "=" * 60)
    print("COMBINED FEATURE ANALYSIS")
    print("=" * 60)

    # Create feature bins
    df['count_bin'] = pd.cut(df['count_points'], bins=[-1, 0, 10, 20, 35],
                              labels=['0', '5-10', '15-20', '25-35'])
    df['trump_bin'] = pd.cut(df['n_trump'], bins=[-1, 1, 3, 7],
                              labels=['0-1', '2-3', '4+'])

    combo = df.groupby(['count_bin', 'trump_bin']).agg({
        'V': ['mean', 'std', 'count']
    }).round(2)
    combo.columns = ['E[V]', 'σ(V)', 'n']
    print("\nV by Count Points × Trump Count:")
    print(combo.to_string())

    # Save detailed results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Save feature correlations
    corr_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation_with_V': correlations.values
    })
    corr_df.to_csv(tables_dir / "11b_v_feature_correlations.csv", index=False)
    print(f"✓ Saved feature correlations")

    # Save grouped statistics
    count_groups.to_csv(tables_dir / "11b_v_by_count_points.csv")
    trump_groups.to_csv(tables_dir / "11b_v_by_trump.csv")
    doubles_groups.to_csv(tables_dir / "11b_v_by_doubles.csv")
    print(f"✓ Saved grouped statistics")

    # Key findings summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    best_predictor = correlations.idxmax()
    best_corr = correlations.max()
    print(f"\n1. BEST PREDICTOR OF V: {best_predictor} (r = {best_corr:.3f})")

    # Effect sizes
    low_count = df[df['count_points'] == 0]['V'].mean()
    high_count = df[df['count_points'] >= 20]['V'].mean() if len(df[df['count_points'] >= 20]) > 0 else df[df['count_points'] >= 15]['V'].mean()
    print(f"\n2. COUNT POINTS EFFECT:")
    print(f"   0 count points: E[V] = {low_count:.1f}")
    print(f"   20+ count points: E[V] = {high_count:.1f}")
    print(f"   Difference: {high_count - low_count:.1f} points")

    low_trump = df[df['n_trump'] <= 1]['V'].mean()
    high_trump = df[df['n_trump'] >= 4]['V'].mean() if len(df[df['n_trump'] >= 4]) > 0 else df[df['n_trump'] >= 3]['V'].mean()
    print(f"\n3. TRUMP COUNT EFFECT:")
    print(f"   0-1 trump: E[V] = {low_trump:.1f}")
    print(f"   4+ trump: E[V] = {high_trump:.1f}")
    print(f"   Difference: {high_trump - low_trump:.1f} points")

    # Variance analysis
    print(f"\n4. VARIANCE ANALYSIS:")
    overall_std = df['V'].std()
    print(f"   Overall σ(V): {overall_std:.1f}")

    for feat in ['count_points', 'n_trump']:
        within_std = df.groupby(feat)['V'].std().mean()
        explained = 1 - (within_std / overall_std) ** 2
        print(f"   {feat} explains {explained*100:.1f}% of variance")


if __name__ == "__main__":
    main()
