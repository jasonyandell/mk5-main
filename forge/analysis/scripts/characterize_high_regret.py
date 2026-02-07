#!/usr/bin/env python3
"""Characterize high-regret states from the regret distribution analysis.

Analyzes patterns in the 1,912 high-regret samples:
- Trick number (game progress)
- Trump holdings
- Declaration types
- Position in trick
- Card patterns
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

from forge.oracle.declarations import DECL_ID_TO_NAME

# Token feature indices
FEAT_HIGH_PIP = 0
FEAT_LOW_PIP = 1
FEAT_IS_DOUBLE = 2
FEAT_COUNT_VALUE = 3
FEAT_TRUMP_RANK = 4
FEAT_NORMALIZED_PLAYER = 5
FEAT_IS_CURRENT = 6
FEAT_IS_PARTNER = 7
FEAT_IS_REMAINING = 8
FEAT_TOKEN_TYPE = 9
FEAT_DECL_ID = 10
FEAT_NORMALIZED_LEADER = 11

COUNT_VALUE_TO_POINTS = {0: 0, 1: 5, 2: 10}
RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"


def load_data():
    """Load high-regret samples and validation tokens."""
    hr_df = pd.read_parquet(RESULTS_DIR / "tables/high_regret_samples.parquet")

    # Load validation tokens for the high-regret samples
    tokens = np.load(Path(PROJECT_ROOT) / "data/tokenized-full/val/tokens.npy")
    masks = np.load(Path(PROJECT_ROOT) / "data/tokenized-full/val/masks.npy")

    return hr_df, tokens, masks


def extract_game_features(tokens: np.ndarray, sample_idx: int) -> dict:
    """Extract game features from token array for a single sample."""
    tok = tokens[sample_idx]  # (32, 12)

    # Context token (position 0)
    decl_id = tok[0, FEAT_DECL_ID]
    normalized_leader = tok[0, FEAT_NORMALIZED_LEADER]

    # Count trick tokens (positions 29-31)
    trick_len = 0
    for i in range(29, 32):
        if tok[i, FEAT_TOKEN_TYPE] >= 5:  # Token types 5,6,7 are trick plays
            trick_len += 1

    # Current player's hand (normalized_player == 0, is_remaining == 1)
    my_hand = []
    my_trump_count = 0
    my_doubles = 0
    my_count_points = 0

    for i in range(1, 29):  # Hand tokens
        if tok[i, FEAT_IS_CURRENT] == 1 and tok[i, FEAT_IS_REMAINING] == 1:
            high, low = tok[i, FEAT_HIGH_PIP], tok[i, FEAT_LOW_PIP]
            is_double = tok[i, FEAT_IS_DOUBLE]
            trump_rank = tok[i, FEAT_TRUMP_RANK]
            count_val = tok[i, FEAT_COUNT_VALUE]

            my_hand.append((high, low))
            if trump_rank < 7:  # Is a trump
                my_trump_count += 1
            if is_double:
                my_doubles += 1
            my_count_points += COUNT_VALUE_TO_POINTS.get(count_val, 0)

    # Count remaining dominoes per player
    remaining_by_player = [0, 0, 0, 0]
    for i in range(1, 29):
        if tok[i, FEAT_IS_REMAINING] == 1:
            player = tok[i, FEAT_NORMALIZED_PLAYER]
            remaining_by_player[player] += 1

    total_remaining = sum(remaining_by_player)
    trick_number = 7 - (total_remaining // 4)  # 7 tricks total, 4 dominoes per trick

    return {
        'decl_id': int(decl_id),
        'decl_name': DECL_ID_TO_NAME[int(decl_id)],
        'trick_number': trick_number,
        'trick_len': trick_len,
        'normalized_leader': int(normalized_leader),
        'hand_size': len(my_hand),
        'my_trump_count': my_trump_count,
        'my_doubles': my_doubles,
        'my_count_points': my_count_points,
        'total_remaining': total_remaining,
        'my_hand': my_hand,
    }


def analyze_high_regret(hr_df: pd.DataFrame, tokens: np.ndarray) -> pd.DataFrame:
    """Add game features to high-regret dataframe."""
    features = []
    for idx, row in hr_df.iterrows():
        sample_idx = row['sample_idx']
        feat = extract_game_features(tokens, sample_idx)
        feat['regret'] = row['regret']
        feat['model_pick'] = row['model_pick']
        feat['oracle_best'] = row['oracle_best']
        feat['team'] = row['team']
        feat['player'] = row['player']
        features.append(feat)

    return pd.DataFrame(features)


def print_analysis(df: pd.DataFrame):
    """Print comprehensive analysis."""
    print("=" * 70)
    print("HIGH-REGRET STATE CHARACTERIZATION")
    print(f"Total high-regret samples: {len(df):,}")
    print("=" * 70)

    # By trick number
    print("\n### BY TRICK NUMBER ###")
    trick_stats = df.groupby('trick_number').agg({
        'regret': ['count', 'mean', 'max'],
    }).round(2)
    trick_stats.columns = ['count', 'mean_regret', 'max_regret']
    trick_stats['pct'] = (trick_stats['count'] / len(df) * 100).round(1)
    print(trick_stats.to_string())

    # By declaration
    print("\n### BY DECLARATION ###")
    decl_stats = df.groupby('decl_name').agg({
        'regret': ['count', 'mean', 'max'],
    }).round(2)
    decl_stats.columns = ['count', 'mean_regret', 'max_regret']
    decl_stats['pct'] = (decl_stats['count'] / len(df) * 100).round(1)
    print(decl_stats.sort_values('count', ascending=False).to_string())

    # By position in trick
    print("\n### BY POSITION IN TRICK ###")
    pos_names = {0: 'lead', 1: '2nd', 2: '3rd', 3: '4th'}
    df['position_name'] = df['trick_len'].map(pos_names)
    pos_stats = df.groupby('position_name').agg({
        'regret': ['count', 'mean', 'max'],
    }).round(2)
    pos_stats.columns = ['count', 'mean_regret', 'max_regret']
    pos_stats['pct'] = (pos_stats['count'] / len(df) * 100).round(1)
    print(pos_stats.to_string())

    # By trump holdings
    print("\n### BY TRUMP HOLDINGS ###")
    trump_stats = df.groupby('my_trump_count').agg({
        'regret': ['count', 'mean', 'max'],
    }).round(2)
    trump_stats.columns = ['count', 'mean_regret', 'max_regret']
    trump_stats['pct'] = (trump_stats['count'] / len(df) * 100).round(1)
    print(trump_stats.to_string())

    # By hand size (game progress)
    print("\n### BY HAND SIZE (REMAINING CARDS) ###")
    hand_stats = df.groupby('hand_size').agg({
        'regret': ['count', 'mean', 'max'],
    }).round(2)
    hand_stats.columns = ['count', 'mean_regret', 'max_regret']
    hand_stats['pct'] = (hand_stats['count'] / len(df) * 100).round(1)
    print(hand_stats.to_string())

    # By doubles
    print("\n### BY DOUBLES IN HAND ###")
    double_stats = df.groupby('my_doubles').agg({
        'regret': ['count', 'mean', 'max'],
    }).round(2)
    double_stats.columns = ['count', 'mean_regret', 'max_regret']
    double_stats['pct'] = (double_stats['count'] / len(df) * 100).round(1)
    print(double_stats.to_string())

    # By count points in hand
    print("\n### BY COUNT POINTS IN HAND ###")
    count_stats = df.groupby('my_count_points').agg({
        'regret': ['count', 'mean', 'max'],
    }).round(2)
    count_stats.columns = ['count', 'mean_regret', 'max_regret']
    count_stats['pct'] = (count_stats['count'] / len(df) * 100).round(1)
    print(count_stats.to_string())

    # Extreme cases
    print("\n### EXTREME REGRET CASES (>= 40 points) ###")
    extreme = df[df['regret'] >= 40].sort_values('regret', ascending=False)
    print(f"Count: {len(extreme)}")
    for _, row in extreme.head(10).iterrows():
        print(f"  regret={row['regret']:.0f} trick={row['trick_number']} decl={row['decl_name']} "
              f"pos={row['trick_len']} trumps={row['my_trump_count']} hand={row['my_hand']}")

    # Correlations
    print("\n### CORRELATIONS WITH REGRET ###")
    numeric_cols = ['trick_number', 'trick_len', 'hand_size', 'my_trump_count',
                    'my_doubles', 'my_count_points', 'team']
    for col in numeric_cols:
        corr = df['regret'].corr(df[col])
        print(f"  {col:20s}: r = {corr:+.3f}")


def plot_distributions(df: pd.DataFrame, output_dir: Path):
    """Create visualization of high-regret distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Trick number distribution
    ax = axes[0, 0]
    trick_counts = df['trick_number'].value_counts().sort_index()
    ax.bar(trick_counts.index, trick_counts.values, color='steelblue', edgecolor='white')
    ax.set_xlabel('Trick Number')
    ax.set_ylabel('Count')
    ax.set_title('High-Regret by Trick Number')
    ax.set_xticks(range(1, 8))

    # Declaration distribution
    ax = axes[0, 1]
    decl_counts = df['decl_name'].value_counts()
    ax.barh(range(len(decl_counts)), decl_counts.values, color='coral')
    ax.set_yticks(range(len(decl_counts)))
    ax.set_yticklabels(decl_counts.index)
    ax.set_xlabel('Count')
    ax.set_title('High-Regret by Declaration')

    # Position in trick
    ax = axes[0, 2]
    pos_counts = df['trick_len'].value_counts().sort_index()
    pos_names = ['Lead', '2nd', '3rd', '4th']
    ax.bar(pos_names[:len(pos_counts)], pos_counts.values, color='forestgreen', edgecolor='white')
    ax.set_xlabel('Position in Trick')
    ax.set_ylabel('Count')
    ax.set_title('High-Regret by Position')

    # Trump holdings
    ax = axes[1, 0]
    trump_counts = df['my_trump_count'].value_counts().sort_index()
    ax.bar(trump_counts.index, trump_counts.values, color='purple', edgecolor='white')
    ax.set_xlabel('Trump Count in Hand')
    ax.set_ylabel('Count')
    ax.set_title('High-Regret by Trump Holdings')

    # Regret vs trick number (box plot)
    ax = axes[1, 1]
    trick_groups = [df[df['trick_number'] == t]['regret'].values for t in range(1, 8)]
    ax.boxplot(trick_groups, labels=range(1, 8))
    ax.set_xlabel('Trick Number')
    ax.set_ylabel('Regret (points)')
    ax.set_title('Regret Distribution by Trick')

    # Hand size distribution
    ax = axes[1, 2]
    hand_counts = df['hand_size'].value_counts().sort_index()
    ax.bar(hand_counts.index, hand_counts.values, color='orange', edgecolor='white')
    ax.set_xlabel('Cards Remaining in Hand')
    ax.set_ylabel('Count')
    ax.set_title('High-Regret by Hand Size')

    plt.tight_layout()
    plt.savefig(output_dir / "figures/high_regret_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_dir}/figures/high_regret_distributions.png")


def main():
    print("Loading data...")
    hr_df, tokens, masks = load_data()

    print("Extracting game features...")
    analysis_df = analyze_high_regret(hr_df, tokens)

    # Print analysis
    print_analysis(analysis_df)

    # Save enriched dataframe
    output_path = RESULTS_DIR / "tables/high_regret_characterized.parquet"
    analysis_df.to_parquet(output_path)
    print(f"\nEnriched data saved to {output_path}")

    # Plot distributions
    plot_distributions(analysis_df, RESULTS_DIR)


if __name__ == "__main__":
    main()
