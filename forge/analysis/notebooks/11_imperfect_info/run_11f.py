#!/usr/bin/env python3
"""
11f: Hand Features → E[V] Regression

Question: What predicts E[V]?
Method: Regression with features: trump count, high dominoes, doubles → E[V]
What It Reveals: Explicit bidding heuristics ("napkin formula")

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from forge.analysis.utils.seed_db import SeedDB
from forge.oracle import tables
from forge.oracle.rng import deal_from_seed

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"
N_BASE_SEEDS = 201  # Analyze all available base seeds
np.random.seed(42)


def extract_hand_features(hand: list[int], trump_suit: int) -> dict:
    """Extract bidding-relevant features from a hand.

    Args:
        hand: List of 7 domino IDs
        trump_suit: The declared trump suit (0-6 pip value, or 7 for follow-me)

    Returns:
        Dictionary of feature values
    """
    features_dict = {}

    # Basic counts
    n_doubles = sum(1 for d in hand if tables.DOMINO_IS_DOUBLE[d])

    # Trump suit length (dominoes matching trump)
    if trump_suit <= 6:
        trump_count = sum(1 for d in hand if tables.domino_contains_pip(d, trump_suit))
    else:
        # Follow-me (trump=7) - no traditional trump suit
        trump_count = 0

    # Pip counts (how many dominoes contain each pip)
    pip_counts = [sum(1 for d in hand if tables.domino_contains_pip(d, pip)) for pip in range(7)]
    max_pip_count = max(pip_counts)
    features_dict['max_suit_length'] = max_pip_count

    # High dominoes
    n_6_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 6)
    n_5_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 5)
    n_4_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 4)

    # Count holdings
    count_points = sum(tables.DOMINO_COUNT_POINTS[d] for d in hand)
    n_count_dominoes = sum(1 for d in hand if tables.DOMINO_COUNT_POINTS[d] > 0)

    # Total pips (rough hand strength)
    total_pips = sum(tables.DOMINO_HIGH[d] + tables.DOMINO_LOW[d] for d in hand)

    # Trump-specific features
    has_trump_double = any(
        tables.DOMINO_IS_DOUBLE[d] and tables.DOMINO_HIGH[d] == trump_suit
        for d in hand
    ) if trump_suit <= 6 else False

    # Features dictionary
    features_dict.update({
        'n_doubles': n_doubles,
        'trump_count': trump_count,
        'n_6_high': n_6_high,
        'n_5_high': n_5_high,
        'n_4_high': n_4_high,
        'count_points': count_points,
        'n_count_dominoes': n_count_dominoes,
        'total_pips': total_pips,
        'has_trump_double': int(has_trump_double),
        'pip_0_count': pip_counts[0],
        'pip_1_count': pip_counts[1],
        'pip_2_count': pip_counts[2],
        'pip_3_count': pip_counts[3],
        'pip_4_count': pip_counts[4],
        'pip_5_count': pip_counts[5],
        'pip_6_count': pip_counts[6],
    })

    return features_dict


def analyze_hand_for_base_seed(db: SeedDB, base_seed: int) -> dict | None:
    """Extract hand features and E[V] for one base seed."""
    decl_id = base_seed % 10  # Declaration from seed
    trump_suit = decl_id  # In standard games, decl_id maps to trump suit

    # Get P0's hand
    p0_hand = deal_from_seed(base_seed)[0]

    # Get V values across all 3 opponent configs
    V_values = []
    for opp_seed in range(3):
        filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        path = DATA_DIR / filename
        if not path.exists():
            return None
        result = db.get_root_v(filename)
        if result.data is None:
            continue
        V_values.append(float(result.data))

    if len(V_values) != 3:
        return None

    # Extract hand features
    hand_features = extract_hand_features(p0_hand, trump_suit)

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'trump_suit': trump_suit,
        'V_mean': np.mean(V_values),
        'V_std': np.std(V_values),
        'V_min': min(V_values),
        'V_max': max(V_values),
        **hand_features
    }


def main():
    print("=" * 60)
    print("HAND FEATURES → E[V] REGRESSION")
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

    # Collect data
    all_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_hand_for_base_seed(db, base_seed)
        if result:
            all_results.append(result)

    db.close()
    print(f"\n✓ Analyzed {len(all_results)} hands")

    # Build DataFrame
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    print(f"\nTarget: E[V] (V_mean)")
    print(f"  Mean: {df['V_mean'].mean():.2f}")
    print(f"  Std: {df['V_mean'].std():.2f}")
    print(f"  Range: [{df['V_mean'].min():.1f}, {df['V_mean'].max():.1f}]")

    # Define feature columns
    basic_features = ['n_doubles', 'trump_count', 'n_6_high', 'n_5_high',
                      'count_points', 'total_pips', 'has_trump_double', 'max_suit_length']

    pip_features = ['pip_0_count', 'pip_1_count', 'pip_2_count', 'pip_3_count',
                    'pip_4_count', 'pip_5_count', 'pip_6_count']

    all_features = basic_features + pip_features

    # Correlation analysis
    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS WITH E[V]")
    print("=" * 60)

    correlations = []
    for feat in basic_features:
        corr = df[feat].corr(df['V_mean'])
        correlations.append({'feature': feat, 'correlation': corr, 'abs_corr': abs(corr)})
        print(f"  {feat}: {corr:+.3f}")

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

    # Regression analysis
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION")
    print("=" * 60)

    X = df[basic_features].values
    y = df['V_mean'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit regression
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    print(f"\nR² Score: {model.score(X_scaled, y):.3f}")
    print(f"CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importances (coefficients)
    print("\nFeature Coefficients (standardized):")
    coef_data = []
    for feat, coef in zip(basic_features, model.coef_):
        coef_data.append({'feature': feat, 'coefficient': coef, 'abs_coef': abs(coef)})
        print(f"  {feat}: {coef:+.3f}")

    coef_df = pd.DataFrame(coef_data).sort_values('abs_coef', ascending=False)

    print(f"\nIntercept: {model.intercept_:.2f}")

    # Unstandardized regression for interpretability
    print("\n" + "=" * 60)
    print("NAPKIN FORMULA (Unstandardized)")
    print("=" * 60)

    model_raw = LinearRegression()
    model_raw.fit(X, y)

    print(f"\nE[V] ≈ {model_raw.intercept_:.1f}")
    for feat, coef in sorted(zip(basic_features, model_raw.coef_), key=lambda x: -abs(x[1])):
        if abs(coef) > 0.5:
            sign = '+' if coef > 0 else ''
            print(f"       {sign}{coef:.1f} × {feat}")

    # Analysis by trump suit
    print("\n" + "=" * 60)
    print("ANALYSIS BY TRUMP SUIT")
    print("=" * 60)

    trump_stats = df.groupby('decl_id').agg({
        'V_mean': ['mean', 'std', 'count'],
        'trump_count': 'mean',
        'count_points': 'mean'
    }).round(2)
    print(trump_stats.to_string())

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Save per-seed data
    df.to_csv(tables_dir / "11f_hand_features_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Save correlations
    corr_df.to_csv(tables_dir / "11f_feature_correlations.csv", index=False)
    print("✓ Saved correlations")

    # Save regression coefficients
    coef_df.to_csv(tables_dir / "11f_regression_coefficients.csv", index=False)
    print("✓ Saved coefficients")

    # Save napkin formula
    napkin_data = [{
        'intercept': model_raw.intercept_,
        'r2': model.score(X_scaled, y),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }]
    for feat, coef in zip(basic_features, model_raw.coef_):
        napkin_data[0][f'coef_{feat}'] = coef

    pd.DataFrame(napkin_data).to_csv(tables_dir / "11f_napkin_formula.csv", index=False)
    print("✓ Saved napkin formula")

    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Feature correlations
    ax1 = axes[0, 0]
    sorted_corr = corr_df.sort_values('correlation')
    colors = ['green' if c > 0 else 'red' for c in sorted_corr['correlation']]
    ax1.barh(range(len(sorted_corr)), sorted_corr['correlation'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(sorted_corr)))
    ax1.set_yticklabels(sorted_corr['feature'])
    ax1.set_xlabel('Correlation with E[V]')
    ax1.set_title('Feature Correlations')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Top right: Predicted vs Actual
    ax2 = axes[0, 1]
    y_pred = model.predict(X_scaled)
    ax2.scatter(y, y_pred, alpha=0.5, s=30)
    ax2.plot([-40, 40], [-40, 40], 'r--', label='Perfect fit')
    ax2.set_xlabel('Actual E[V]')
    ax2.set_ylabel('Predicted E[V]')
    ax2.set_title(f'Predicted vs Actual (R² = {model.score(X_scaled, y):.3f})')
    ax2.legend()

    # Bottom left: Top feature vs E[V]
    ax3 = axes[1, 0]
    top_feature = corr_df.iloc[0]['feature']
    ax3.scatter(df[top_feature], df['V_mean'], alpha=0.5, s=30)
    z = np.polyfit(df[top_feature], df['V_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[top_feature].min(), df[top_feature].max(), 100)
    ax3.plot(x_line, p(x_line), 'r-', label=f'r = {corr_df.iloc[0]["correlation"]:.2f}')
    ax3.set_xlabel(top_feature)
    ax3.set_ylabel('E[V]')
    ax3.set_title(f'Most Correlated Feature: {top_feature}')
    ax3.legend()

    # Bottom right: Regression coefficients
    ax4 = axes[1, 1]
    sorted_coef = coef_df.sort_values('coefficient')
    colors = ['green' if c > 0 else 'red' for c in sorted_coef['coefficient']]
    ax4.barh(range(len(sorted_coef)), sorted_coef['coefficient'], color=colors, alpha=0.7)
    ax4.set_yticks(range(len(sorted_coef)))
    ax4.set_yticklabels(sorted_coef['feature'])
    ax4.set_xlabel('Standardized Coefficient')
    ax4.set_title('Regression Coefficients')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11f_hand_features_to_ev.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\n1. MODEL FIT:")
    print(f"   R² = {model.score(X_scaled, y):.3f} ({100*model.score(X_scaled, y):.1f}% of variance explained)")
    if model.score(X_scaled, y) < 0.3:
        print("   → Hand features explain only a small portion of E[V]")
        print("   → Most variance comes from opponent hands (unobserved)")
    elif model.score(X_scaled, y) < 0.6:
        print("   → Hand features moderately predict E[V]")
        print("   → Still significant uncertainty from opponent hands")
    else:
        print("   → Hand features strongly predict E[V]")
        print("   → Bidding can be largely determined from own hand")

    print("\n2. TOP PREDICTORS (by |correlation|):")
    for _, row in corr_df.head(5).iterrows():
        direction = "↑" if row['correlation'] > 0 else "↓"
        print(f"   {row['feature']}: {direction} ({row['correlation']:+.3f})")

    print("\n3. NAPKIN FORMULA:")
    print(f"   E[V] ≈ {model_raw.intercept_:.1f}")
    for feat, coef in sorted(zip(basic_features, model_raw.coef_), key=lambda x: -abs(x[1]))[:4]:
        if abs(coef) > 0.5:
            sign = '+' if coef > 0 else ''
            print(f"         {sign}{coef:.1f} × {feat}")


if __name__ == "__main__":
    main()
