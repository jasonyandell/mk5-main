#!/usr/bin/env python3
"""
11g: Hand Features → Count Locks

Question: What hand features predict count locks?
Method: Regression with features: suit length, high pips, doubles → lock rate
What It Reveals: The "napkin bidding formula" for count control

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
A count is "locked" if Team 0 captures it in all 3 opponent configurations.
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from forge.analysis.utils import features
from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed, deal_with_fixed_p0

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Full analysis
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
    """Determine which team holds each count domino."""
    ownership = {}
    for count_id in COUNT_DOMINO_IDS:
        for player_id, hand in enumerate(hands):
            if count_id in hand:
                team = player_id % 2
                ownership[count_id] = team
                break
    return ownership


def extract_hand_features(hand: list[int], trump_suit: int) -> dict:
    """Extract bidding-relevant features from a hand."""
    features_dict = {}

    # Basic counts
    n_doubles = sum(1 for d in hand if tables.DOMINO_IS_DOUBLE[d])

    # Trump suit length
    if trump_suit <= 6:
        trump_count = sum(1 for d in hand if tables.domino_contains_pip(d, trump_suit))
    else:
        trump_count = 0

    # Pip counts
    pip_counts = [sum(1 for d in hand if tables.domino_contains_pip(d, pip)) for pip in range(7)]
    max_pip_count = max(pip_counts)

    # High dominoes
    n_6_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 6)
    n_5_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 5)

    # Count holdings
    count_points = sum(tables.DOMINO_COUNT_POINTS[d] for d in hand)
    n_count_dominoes = sum(1 for d in hand if tables.DOMINO_COUNT_POINTS[d] > 0)

    # Total pips
    total_pips = sum(tables.DOMINO_HIGH[d] + tables.DOMINO_LOW[d] for d in hand)

    # Trump double
    has_trump_double = any(
        tables.DOMINO_IS_DOUBLE[d] and tables.DOMINO_HIGH[d] == trump_suit
        for d in hand
    ) if trump_suit <= 6 else False

    # Count-specific features: does P0 hold each count?
    holds_count = {}
    for count_id, count_name in zip(COUNT_DOMINO_IDS, COUNT_NAMES):
        holds_count[f'holds_{count_name}'] = int(count_id in hand)

    # Suit lengths for count suits (5-suit for 5-0, 5-5; 6-suit for 6-4; etc.)
    suit_5_length = pip_counts[5]  # For 5-0, 5-5
    suit_6_length = pip_counts[6]  # For 6-4
    suit_4_length = pip_counts[4]  # For 4-1
    suit_3_length = pip_counts[3]  # For 3-2

    features_dict.update({
        'n_doubles': n_doubles,
        'trump_count': trump_count,
        'n_6_high': n_6_high,
        'n_5_high': n_5_high,
        'count_points': count_points,
        'n_count_dominoes': n_count_dominoes,
        'total_pips': total_pips,
        'has_trump_double': int(has_trump_double),
        'max_suit_length': max_pip_count,
        'suit_5_length': suit_5_length,
        'suit_6_length': suit_6_length,
        'suit_4_length': suit_4_length,
        'suit_3_length': suit_3_length,
        **holds_count
    })

    return features_dict


def analyze_count_locks_for_base_seed(base_seed: int) -> dict | None:
    """Analyze count lock rates for one base seed."""
    decl_id = base_seed % 10
    trump_suit = decl_id
    p0_hand = deal_from_seed(base_seed)[0]

    # Track count captures across 3 opponent configs
    count_captures = {name: [] for name in COUNT_NAMES}  # 1 if Team 0 captures, 0 otherwise
    V_values = []

    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        V = get_root_v_fast(path)
        if V is None:
            continue
        V_values.append(V)

        hands = deal_with_fixed_p0(p0_hand, opp_seed)
        ownership = get_team_count_ownership(hands)

        # Estimate capture based on V and ownership
        # High V + Team 0 holds → likely captured
        # Low V + Team 0 holds → possibly lost
        for count_id, count_name in zip(COUNT_DOMINO_IDS, COUNT_NAMES):
            team_holds = ownership.get(count_id, 1)  # Default to Team 1 if not found

            if team_holds == 0:  # Team 0 holds
                # Higher V → more likely to have captured
                capture_prob = np.clip((V + 42) / 84, 0.2, 0.9)
            else:  # Team 1 holds
                # Lower V → opponent kept their count
                capture_prob = np.clip((V + 42) / 84 * 0.3, 0.0, 0.4)

            # Binary decision: captured if prob > 0.5
            captured = 1 if capture_prob > 0.5 else 0
            count_captures[count_name].append(captured)

    if len(V_values) != 3:
        return None

    # Calculate lock rates (proportion of configs where count was captured)
    lock_rates = {}
    for count_name in COUNT_NAMES:
        captures = count_captures[count_name]
        lock_rates[f'lock_rate_{count_name}'] = np.mean(captures)
        lock_rates[f'locked_{count_name}'] = int(all(c == 1 for c in captures))

    # Total locks
    total_locked = sum(lock_rates[f'locked_{name}'] for name in COUNT_NAMES)
    total_lock_rate = np.mean([lock_rates[f'lock_rate_{name}'] for name in COUNT_NAMES])

    # Extract hand features
    hand_features = extract_hand_features(p0_hand, trump_suit)

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'V_mean': np.mean(V_values),
        'total_locked': total_locked,
        'total_lock_rate': total_lock_rate,
        **lock_rates,
        **hand_features
    }


def main():
    print("=" * 60)
    print("HAND FEATURES → COUNT LOCKS")
    print("=" * 60)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    # Collect data
    all_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_count_locks_for_base_seed(base_seed)
        if result:
            all_results.append(result)

    print(f"\n✓ Analyzed {len(all_results)} hands")

    # Build DataFrame
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    print(f"\nTotal lock rates by count:")
    for count_name in COUNT_NAMES:
        rate = df[f'lock_rate_{count_name}'].mean()
        locked_pct = df[f'locked_{count_name}'].mean() * 100
        print(f"  {count_name}: {rate:.2f} avg lock rate, {locked_pct:.0f}% fully locked")

    print(f"\nOverall: {df['total_lock_rate'].mean():.2f} avg, {df['total_locked'].mean():.1f} counts locked per hand")

    # Feature correlations with total_lock_rate
    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS WITH LOCK RATE")
    print("=" * 60)

    basic_features = ['n_doubles', 'trump_count', 'n_6_high', 'n_5_high',
                      'count_points', 'total_pips', 'has_trump_double', 'max_suit_length']

    correlations = []
    for feat in basic_features:
        corr = df[feat].corr(df['total_lock_rate'])
        correlations.append({'feature': feat, 'correlation': corr, 'abs_corr': abs(corr)})
        print(f"  {feat}: {corr:+.3f}")

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

    # Per-count analysis: what predicts locking each specific count?
    print("\n" + "=" * 60)
    print("PER-COUNT LOCK PREDICTORS")
    print("=" * 60)

    count_predictors = []
    for count_name in COUNT_NAMES:
        target = f'lock_rate_{count_name}'
        holds_col = f'holds_{count_name}'

        # Correlation with holding the count
        corr_holds = df[holds_col].corr(df[target]) if holds_col in df.columns else np.nan

        # Best predictor among basic features
        best_feat = None
        best_corr = 0
        for feat in basic_features:
            c = df[feat].corr(df[target])
            if abs(c) > abs(best_corr):
                best_corr = c
                best_feat = feat

        count_predictors.append({
            'count': count_name,
            'corr_with_holding': corr_holds,
            'best_feature': best_feat,
            'best_feature_corr': best_corr
        })

        print(f"\n{count_name}:")
        print(f"  Holding it: {corr_holds:+.3f}")
        print(f"  Best feature: {best_feat} ({best_corr:+.3f})")

    count_pred_df = pd.DataFrame(count_predictors)

    # Regression for total lock rate
    print("\n" + "=" * 60)
    print("REGRESSION: Features → Total Lock Rate")
    print("=" * 60)

    X = df[basic_features].values
    y = df['total_lock_rate'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(df)), scoring='r2')

    print(f"\nR² Score: {model.score(X_scaled, y):.3f}")
    print(f"CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    print("\nFeature Coefficients:")
    coef_data = []
    for feat, coef in zip(basic_features, model.coef_):
        coef_data.append({'feature': feat, 'coefficient': coef})
        print(f"  {feat}: {coef:+.3f}")

    coef_df = pd.DataFrame(coef_data)

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11g_count_locks_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    corr_df.to_csv(tables_dir / "11g_lock_correlations.csv", index=False)
    print("✓ Saved correlations")

    count_pred_df.to_csv(tables_dir / "11g_per_count_predictors.csv", index=False)
    print("✓ Saved per-count predictors")

    coef_df.to_csv(tables_dir / "11g_regression_coefficients.csv", index=False)
    print("✓ Saved regression coefficients")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Lock rates by count
    ax1 = axes[0, 0]
    lock_rates = [df[f'lock_rate_{name}'].mean() for name in COUNT_NAMES]
    ax1.bar(COUNT_NAMES, lock_rates, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Average Lock Rate')
    ax1.set_title('Lock Rate by Count Domino')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    # Top right: Feature correlations
    ax2 = axes[0, 1]
    sorted_corr = corr_df.sort_values('correlation')
    colors = ['green' if c > 0 else 'red' for c in sorted_corr['correlation']]
    ax2.barh(range(len(sorted_corr)), sorted_corr['correlation'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(sorted_corr)))
    ax2.set_yticklabels(sorted_corr['feature'])
    ax2.set_xlabel('Correlation with Lock Rate')
    ax2.set_title('Feature Correlations')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Bottom left: Holding vs Lock Rate per count
    ax3 = axes[1, 0]
    hold_corrs = count_pred_df['corr_with_holding'].values
    ax3.bar(COUNT_NAMES, hold_corrs, color='steelblue', alpha=0.7)
    ax3.set_ylabel('Correlation: Holding → Lock Rate')
    ax3.set_title('Does Holding the Count Predict Locking It?')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Bottom right: Regression coefficients
    ax4 = axes[1, 1]
    sorted_coef = coef_df.sort_values('coefficient')
    colors = ['green' if c > 0 else 'red' for c in sorted_coef['coefficient']]
    ax4.barh(range(len(sorted_coef)), sorted_coef['coefficient'], color=colors, alpha=0.7)
    ax4.set_yticks(range(len(sorted_coef)))
    ax4.set_yticklabels(sorted_coef['feature'])
    ax4.set_xlabel('Standardized Coefficient')
    ax4.set_title(f'Regression Coefficients (R²={model.score(X_scaled, y):.2f})')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11g_hand_features_to_locks.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\n1. LOCK RATES:")
    for count_name in COUNT_NAMES:
        rate = df[f'lock_rate_{count_name}'].mean()
        print(f"   {count_name}: {rate:.2f}")

    print("\n2. TOP PREDICTORS:")
    for _, row in corr_df.head(3).iterrows():
        direction = "↑" if row['correlation'] > 0 else "↓"
        print(f"   {row['feature']}: {direction} ({row['correlation']:+.3f})")

    print("\n3. HOLDING → LOCKING:")
    for _, row in count_pred_df.iterrows():
        strength = "strong" if abs(row['corr_with_holding']) > 0.5 else "weak"
        print(f"   {row['count']}: {strength} ({row['corr_with_holding']:+.3f})")


if __name__ == "__main__":
    main()
