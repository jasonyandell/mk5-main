#!/usr/bin/env python3
"""
11s: Hand Features → σ(V) Regression

Question: What predicts outcome variance?
Method: Regression: trump count, high dominoes, doubles → σ(V)
What It Reveals: Risk assessment heuristics

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
Reuses feature extraction from 11f but targets σ(V) instead of E[V].
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from forge.analysis.utils import features
from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 10  # Quick analysis (use 201 for full run)
np.random.seed(42)


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
    features_dict['max_suit_length'] = max_pip_count

    # High dominoes
    n_6_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 6)
    n_5_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 5)
    n_4_high = sum(1 for d in hand if tables.DOMINO_HIGH[d] == 4)

    # Count holdings
    count_points = sum(tables.DOMINO_COUNT_POINTS[d] for d in hand)
    n_count_dominoes = sum(1 for d in hand if tables.DOMINO_COUNT_POINTS[d] > 0)

    # Total pips
    total_pips = sum(tables.DOMINO_HIGH[d] + tables.DOMINO_LOW[d] for d in hand)

    # Trump-specific features
    has_trump_double = any(
        tables.DOMINO_IS_DOUBLE[d] and tables.DOMINO_HIGH[d] == trump_suit
        for d in hand
    ) if trump_suit <= 6 else False

    # Void suits (suits with 0 dominoes)
    n_voids = sum(1 for c in pip_counts if c == 0)

    # Singleton suits
    n_singletons = sum(1 for c in pip_counts if c == 1)

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
        'n_voids': n_voids,
        'n_singletons': n_singletons,
    })

    return features_dict


def analyze_hand_for_base_seed(base_seed: int) -> dict | None:
    """Extract hand features and V statistics for one base seed."""
    decl_id = base_seed % 10
    trump_suit = decl_id

    # Get P0's hand
    p0_hand = deal_from_seed(base_seed)[0]

    # Get V values across all 3 opponent configs
    V_values = []
    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None
        V = get_root_v_fast(path)
        if V is None:
            continue
        V_values.append(V)

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
        'V_spread': max(V_values) - min(V_values),
        'V_min': min(V_values),
        'V_max': max(V_values),
        **hand_features
    }


def main():
    print("=" * 60)
    print("σ(V) vs HAND FEATURES REGRESSION")
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
        result = analyze_hand_for_base_seed(base_seed)
        if result:
            all_results.append(result)

    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    print(f"\nTarget: σ(V) (V_std)")
    print(f"  Mean: {df['V_std'].mean():.2f}")
    print(f"  Std: {df['V_std'].std():.2f}")
    print(f"  Range: [{df['V_std'].min():.1f}, {df['V_std'].max():.1f}]")

    # Also show V_spread as alternative measure
    print(f"\nAlternative: V_spread (max - min)")
    print(f"  Mean: {df['V_spread'].mean():.2f}")
    print(f"  Range: [{df['V_spread'].min():.1f}, {df['V_spread'].max():.1f}]")

    # Define feature columns
    basic_features = ['n_doubles', 'trump_count', 'n_6_high', 'n_5_high',
                      'count_points', 'total_pips', 'has_trump_double',
                      'max_suit_length', 'n_voids', 'n_singletons']

    # Correlation analysis
    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS WITH σ(V)")
    print("=" * 60)

    correlations = []
    for feat in basic_features:
        corr = df[feat].corr(df['V_std'])
        correlations.append({'feature': feat, 'correlation': corr, 'abs_corr': abs(corr)})
        print(f"  {feat}: {corr:+.3f}")

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

    # Also correlate with V_spread
    print("\n  --- V_spread correlations ---")
    spread_correlations = []
    for feat in basic_features:
        corr = df[feat].corr(df['V_spread'])
        spread_correlations.append({'feature': feat, 'correlation': corr, 'abs_corr': abs(corr)})
        print(f"  {feat}: {corr:+.3f}")

    spread_corr_df = pd.DataFrame(spread_correlations).sort_values('abs_corr', ascending=False)

    # Regression analysis
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION (target: σ(V))")
    print("=" * 60)

    X = df[basic_features].values
    y_std = df['V_std'].values
    y_spread = df['V_spread'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit regression for σ(V)
    model_std = Ridge(alpha=1.0)
    model_std.fit(X_scaled, y_std)

    cv_scores_std = cross_val_score(model_std, X_scaled, y_std, cv=5, scoring='r2')

    print(f"\nR² Score (σ(V)): {model_std.score(X_scaled, y_std):.3f}")
    print(f"CV R² Score: {cv_scores_std.mean():.3f} ± {cv_scores_std.std():.3f}")

    # Fit regression for V_spread
    model_spread = Ridge(alpha=1.0)
    model_spread.fit(X_scaled, y_spread)

    cv_scores_spread = cross_val_score(model_spread, X_scaled, y_spread, cv=5, scoring='r2')

    print(f"\nR² Score (V_spread): {model_spread.score(X_scaled, y_spread):.3f}")
    print(f"CV R² Score: {cv_scores_spread.mean():.3f} ± {cv_scores_spread.std():.3f}")

    # Feature coefficients for σ(V)
    print("\nFeature Coefficients for σ(V) (standardized):")
    coef_data = []
    for feat, coef in zip(basic_features, model_std.coef_):
        coef_data.append({'feature': feat, 'coefficient': coef, 'abs_coef': abs(coef)})
        print(f"  {feat}: {coef:+.3f}")

    coef_df = pd.DataFrame(coef_data).sort_values('abs_coef', ascending=False)

    # Unstandardized formula
    print("\n" + "=" * 60)
    print("RISK FORMULA (Unstandardized)")
    print("=" * 60)

    model_raw = LinearRegression()
    model_raw.fit(X, y_spread)  # Use spread for interpretability

    print(f"\nV_spread ≈ {model_raw.intercept_:.1f}")
    for feat, coef in sorted(zip(basic_features, model_raw.coef_), key=lambda x: -abs(x[1])):
        if abs(coef) > 0.5:
            sign = '+' if coef > 0 else ''
            print(f"          {sign}{coef:.1f} × {feat}")

    # Risk classification
    print("\n" + "=" * 60)
    print("RISK CLASSIFICATION")
    print("=" * 60)

    low_risk = df['V_spread'] < 20
    med_risk = (df['V_spread'] >= 20) & (df['V_spread'] <= 45)
    high_risk = df['V_spread'] > 45

    print(f"\nLow risk (spread < 20): {low_risk.sum()} ({low_risk.mean()*100:.0f}%)")
    print(f"Medium risk (20-45): {med_risk.sum()} ({med_risk.mean()*100:.0f}%)")
    print(f"High risk (spread > 45): {high_risk.sum()} ({high_risk.mean()*100:.0f}%)")

    # Feature profiles by risk
    if low_risk.sum() > 0:
        print(f"\nLow risk hands avg features:")
        for feat in ['n_doubles', 'trump_count', 'total_pips']:
            print(f"  {feat}: {df[low_risk][feat].mean():.1f}")

    if high_risk.sum() > 0:
        print(f"\nHigh risk hands avg features:")
        for feat in ['n_doubles', 'trump_count', 'total_pips']:
            print(f"  {feat}: {df[high_risk][feat].mean():.1f}")

    # E[V] vs σ(V) relationship
    print("\n" + "=" * 60)
    print("E[V] vs σ(V) RELATIONSHIP")
    print("=" * 60)

    ev_std_corr = df['V_mean'].corr(df['V_std'])
    ev_spread_corr = df['V_mean'].corr(df['V_spread'])
    print(f"\nCorrelation E[V] vs σ(V): {ev_std_corr:+.3f}")
    print(f"Correlation E[V] vs V_spread: {ev_spread_corr:+.3f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11s_sigma_v_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    corr_df.to_csv(tables_dir / "11s_sigma_correlations.csv", index=False)
    print("✓ Saved correlations")

    coef_df.to_csv(tables_dir / "11s_regression_coefficients.csv", index=False)
    print("✓ Saved coefficients")

    summary_data = [{
        'r2_std': model_std.score(X_scaled, y_std),
        'cv_r2_std_mean': cv_scores_std.mean(),
        'cv_r2_std_std': cv_scores_std.std(),
        'r2_spread': model_spread.score(X_scaled, y_spread),
        'cv_r2_spread_mean': cv_scores_spread.mean(),
        'cv_r2_spread_std': cv_scores_spread.std(),
        'ev_std_corr': ev_std_corr,
        'n_samples': len(df)
    }]
    pd.DataFrame(summary_data).to_csv(tables_dir / "11s_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Feature correlations with σ(V)
    ax1 = axes[0, 0]
    sorted_corr = corr_df.sort_values('correlation')
    colors = ['green' if c > 0 else 'red' for c in sorted_corr['correlation']]
    ax1.barh(range(len(sorted_corr)), sorted_corr['correlation'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(sorted_corr)))
    ax1.set_yticklabels(sorted_corr['feature'])
    ax1.set_xlabel('Correlation with σ(V)')
    ax1.set_title('Feature Correlations with Outcome Variance')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Top right: E[V] vs σ(V)
    ax2 = axes[0, 1]
    ax2.scatter(df['V_mean'], df['V_std'], alpha=0.6, s=50)
    ax2.set_xlabel('E[V] (Expected Outcome)')
    ax2.set_ylabel('σ(V) (Outcome Variance)')
    ax2.set_title(f'Expected Value vs Risk (r = {ev_std_corr:.2f})')

    # Bottom left: V_spread distribution
    ax3 = axes[1, 0]
    ax3.hist(df['V_spread'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=20, color='green', linestyle='--', label='Low risk threshold')
    ax3.axvline(x=45, color='red', linestyle='--', label='High risk threshold')
    ax3.set_xlabel('V Spread (max - min)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Risk Distribution')
    ax3.legend()

    # Bottom right: Top feature vs V_spread
    ax4 = axes[1, 1]
    top_feature = spread_corr_df.iloc[0]['feature']
    ax4.scatter(df[top_feature], df['V_spread'], alpha=0.6, s=50)
    z = np.polyfit(df[top_feature], df['V_spread'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[top_feature].min(), df[top_feature].max(), 100)
    ax4.plot(x_line, p(x_line), 'r-', label=f'r = {spread_corr_df.iloc[0]["correlation"]:.2f}')
    ax4.set_xlabel(top_feature)
    ax4.set_ylabel('V Spread')
    ax4.set_title(f'Most Correlated: {top_feature}')
    ax4.legend()

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11s_sigma_v_regression.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\n1. MODEL FIT:")
    print(f"   R² = {model_std.score(X_scaled, y_std):.3f} (hand features explain {100*model_std.score(X_scaled, y_std):.0f}% of σ(V))")
    if model_std.score(X_scaled, y_std) < 0.2:
        print("   → Risk is hard to predict from hand alone")
    else:
        print("   → Some risk prediction possible")

    print("\n2. TOP RISK PREDICTORS:")
    for _, row in corr_df.head(3).iterrows():
        direction = "↑" if row['correlation'] > 0 else "↓"
        print(f"   {row['feature']}: {direction} ({row['correlation']:+.3f})")

    print(f"\n3. E[V] vs σ(V) RELATIONSHIP:")
    if abs(ev_std_corr) < 0.2:
        print(f"   r = {ev_std_corr:.2f} → Expected value and risk are independent")
        print("   → High EV hands can be low or high risk")
    elif ev_std_corr < -0.2:
        print(f"   r = {ev_std_corr:.2f} → Better hands tend to have LOWER variance")
    else:
        print(f"   r = {ev_std_corr:.2f} → Better hands tend to have HIGHER variance")


if __name__ == "__main__":
    main()
