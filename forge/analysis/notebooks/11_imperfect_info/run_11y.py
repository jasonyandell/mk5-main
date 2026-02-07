#!/usr/bin/env python3
"""
11y: Reducible Uncertainty Decomposition

Question: What % of variance is opponent-dependent?
Method: ANOVA-style decomposition: V = f(my_hand) + g(opponent_hands) + ε
What It Reveals: Skill vs luck ratio

This gives the definitive answer to "how much is luck vs skill in Texas 42"
using the marginalized oracle data.
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from forge.analysis.utils import features
from forge.analysis.utils.seed_db import SeedDB
from forge.oracle.rng import deal_from_seed
from forge.oracle import schema

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
np.random.seed(42)


def load_root_v(db: SeedDB, filename: str) -> float | None:
    """Load root V (depth 28) from a shard using SeedDB."""
    try:
        result = db.get_root_v(filename)
        return float(result.data) if result.data is not None else None
    except Exception:
        return None


def get_hand_features(base_seed: int, decl_id: int) -> dict:
    """Extract hand features for P0."""
    hands = deal_from_seed(base_seed)
    p0_hand_ids = hands[0]  # List of domino IDs (integers)
    trump = decl_id

    # Convert domino IDs to pip tuples
    p0_hand = [schema.domino_pips(d) for d in p0_hand_ids]

    # Feature extraction
    n_doubles = sum(1 for d in p0_hand if d[0] == d[1])
    trump_count = sum(1 for d in p0_hand if trump in d)
    has_trump_double = any(d == (trump, trump) for d in p0_hand)

    # Count points
    count_dominoes = [(3, 2), (4, 1), (5, 0), (5, 5), (6, 4)]
    count_points = 0
    for cd in count_dominoes:
        cd_norm = tuple(sorted(cd))
        for d in p0_hand:
            if tuple(sorted(d)) == cd_norm:
                if cd in [(5, 5), (6, 4)]:
                    count_points += 10
                else:
                    count_points += 5

    # High dominoes
    n_6_high = sum(1 for d in p0_hand if 6 in d)
    total_pips = sum(d[0] + d[1] for d in p0_hand)

    return {
        'n_doubles': n_doubles,
        'trump_count': trump_count,
        'has_trump_double': int(has_trump_double),
        'count_points': count_points,
        'n_6_high': n_6_high,
        'total_pips': total_pips,
    }


def main():
    print("=" * 60)
    print("REDUCIBLE UNCERTAINTY DECOMPOSITION")
    print("Skill vs Luck Analysis")
    print("=" * 60)

    # Initialize SeedDB
    db = SeedDB(DATA_DIR)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    print(f"Using all {len(base_seeds)} seeds for full analysis...")

    # Collect all V values
    all_data = []

    for base_seed in tqdm(base_seeds, desc="Loading"):
        decl_id = base_seed % 10

        v_values = []
        for opp_seed in range(3):
            filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
            filepath = DATA_DIR / filename
            if not filepath.exists():
                continue
            v = load_root_v(db, filename)
            if v is not None:
                v_values.append(v)

        if len(v_values) == 3:
            hand_features = get_hand_features(base_seed, decl_id)
            for opp_seed, v in enumerate(v_values):
                all_data.append({
                    'base_seed': base_seed,
                    'opp_seed': opp_seed,
                    'V': v,
                    **hand_features
                })

        gc.collect()

    db.close()
    print(f"\n✓ Loaded {len(all_data)} observations ({len(all_data)//3} hands × 3 configs)")

    if len(all_data) == 0:
        print("No data loaded")
        return

    df = pd.DataFrame(all_data)

    # ANOVA decomposition
    print("\n" + "=" * 60)
    print("ANOVA VARIANCE DECOMPOSITION")
    print("=" * 60)

    # Grand mean
    grand_mean = df['V'].mean()

    # Hand means (mean V per base_seed)
    hand_means = df.groupby('base_seed')['V'].mean()
    df['hand_mean'] = df['base_seed'].map(hand_means)

    # Compute sum of squares
    # Total SS
    ss_total = ((df['V'] - grand_mean) ** 2).sum()

    # Between-hand SS (variance explained by hand)
    n_configs = 3
    ss_between = n_configs * ((hand_means - grand_mean) ** 2).sum()

    # Within-hand SS (variance within same hand across configs = luck)
    ss_within = ((df['V'] - df['hand_mean']) ** 2).sum()

    # Variance ratios
    pct_hand = ss_between / ss_total * 100
    pct_opponent = ss_within / ss_total * 100

    print(f"\n  Total variance: {df['V'].var():.1f}")
    print(f"  Grand mean V: {grand_mean:.1f}")

    print(f"\n  Sum of Squares:")
    print(f"    Total SS: {ss_total:,.0f}")
    print(f"    Between-hand SS: {ss_between:,.0f}")
    print(f"    Within-hand SS: {ss_within:,.0f}")

    print(f"\n  VARIANCE DECOMPOSITION:")
    print(f"    Hand (bidding skill): {pct_hand:.1f}%")
    print(f"    Opponent (luck):      {pct_opponent:.1f}%")

    # Hand-level analysis
    print("\n" + "=" * 60)
    print("HAND-LEVEL VARIANCE ANALYSIS")
    print("=" * 60)

    hand_df = df.groupby('base_seed').agg({
        'V': ['mean', 'std', lambda x: x.max() - x.min()],
        'n_doubles': 'first',
        'trump_count': 'first',
        'has_trump_double': 'first',
        'count_points': 'first',
        'n_6_high': 'first',
        'total_pips': 'first',
    }).reset_index()
    hand_df.columns = ['base_seed', 'V_mean', 'V_std', 'V_spread',
                       'n_doubles', 'trump_count', 'has_trump_double',
                       'count_points', 'n_6_high', 'total_pips']

    print(f"\n  Hand-level statistics:")
    print(f"    Mean E[V]: {hand_df['V_mean'].mean():.1f}")
    print(f"    Std of E[V] across hands: {hand_df['V_mean'].std():.1f}")
    print(f"    Mean σ(V) within hand: {hand_df['V_std'].mean():.1f}")
    print(f"    Mean V spread: {hand_df['V_spread'].mean():.1f}")

    # Feature prediction of between-hand variance
    print("\n" + "=" * 60)
    print("FEATURE PREDICTION OF HAND COMPONENT")
    print("=" * 60)

    feature_cols = ['n_doubles', 'trump_count', 'has_trump_double',
                    'count_points', 'n_6_high', 'total_pips']

    # Correlation with E[V]
    print(f"\n  Feature correlations with E[V] (hand component):")
    for col in feature_cols:
        corr = hand_df[col].corr(hand_df['V_mean'])
        print(f"    {col}: {corr:+.3f}")

    # Regression
    X = hand_df[feature_cols].values
    y = hand_df['V_mean'].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    print(f"\n  Regression R² (features → E[V]): {r2:.3f}")
    print(f"    → Hand features explain {r2*100:.0f}% of between-hand variance")
    print(f"    → Total explained: {pct_hand * r2 / 100:.1f}% of total variance")

    # Feature prediction of within-hand variance
    print("\n" + "=" * 60)
    print("FEATURE PREDICTION OF LUCK COMPONENT")
    print("=" * 60)

    # Correlation with V_std (within-hand variance)
    print(f"\n  Feature correlations with σ(V) (luck component):")
    for col in feature_cols:
        corr = hand_df[col].corr(hand_df['V_std'])
        print(f"    {col}: {corr:+.3f}")

    # Reducible uncertainty
    print("\n" + "=" * 60)
    print("REDUCIBLE VS IRREDUCIBLE UNCERTAINTY")
    print("=" * 60)

    # The "skill" in this game comes from:
    # 1. Bidding correctly (choosing to bid on good hands)
    # 2. Playing optimally given your hand

    # Between-hand variance is "reducible" - better bidding reduces exposure to bad hands
    # Within-hand variance is "irreducible" - you can't control what opponents hold

    # But SOME within-hand variance might be reducible if you play better
    # Let's estimate this by looking at whether hand features predict within-hand variance

    X_luck = hand_df[feature_cols].values
    y_luck = hand_df['V_std'].values

    model_luck = LinearRegression()
    model_luck.fit(X_luck, y_luck)
    r2_luck = model_luck.score(X_luck, y_luck)

    print(f"\n  Within-hand variance (luck) predictability:")
    print(f"    R² (features → σ(V)): {r2_luck:.3f}")
    print(f"    → Hand features explain {r2_luck*100:.0f}% of luck component variance")

    # Final decomposition
    print("\n" + "=" * 60)
    print("FINAL SKILL VS LUCK DECOMPOSITION")
    print("=" * 60)

    print(f"\n  TOTAL VARIANCE = 100%")
    print(f"")
    print(f"  ┌─ HAND COMPONENT: {pct_hand:.1f}%")
    print(f"  │   └─ Predictable from features: {pct_hand * r2:.1f}% (true skill)")
    print(f"  │   └─ Unpredictable hand effect: {pct_hand * (1-r2):.1f}%")
    print(f"  │")
    print(f"  └─ OPPONENT COMPONENT: {pct_opponent:.1f}%")
    print(f"      └─ Correlated with hand: {pct_opponent * r2_luck:.1f}% (reducible)")
    print(f"      └─ Pure luck: {pct_opponent * (1-r2_luck):.1f}%")

    # Summary metrics
    true_skill = pct_hand * r2
    reducible_luck = pct_opponent * r2_luck
    pure_luck = pct_opponent * (1 - r2_luck)
    uncontrolled = pct_hand * (1 - r2) + reducible_luck

    print(f"\n  SUMMARY:")
    print(f"    True skill (bidding on good hands): {true_skill:.1f}%")
    print(f"    Pure luck (opponent distribution): {pure_luck:.1f}%")
    print(f"    Mixed (hand effect + correlated luck): {uncontrolled:.1f}%")

    skill_ratio = true_skill / (true_skill + pure_luck)
    print(f"\n  SKILL VS LUCK RATIO: {skill_ratio*100:.0f}% skill / {(1-skill_ratio)*100:.0f}% luck")

    # Category breakdown
    print("\n" + "=" * 60)
    print("UNCERTAINTY BY HAND QUALITY")
    print("=" * 60)

    # Bin hands by E[V]
    hand_df['ev_bin'] = pd.cut(hand_df['V_mean'],
                               bins=[-100, -10, 10, 25, 100],
                               labels=['Very Weak (<-10)', 'Weak (-10 to 10)',
                                      'Good (10-25)', 'Strong (>25)'])

    bin_stats = hand_df.groupby('ev_bin', observed=True).agg({
        'V_mean': 'mean',
        'V_std': 'mean',
        'V_spread': 'mean',
        'base_seed': 'count'
    }).round(1)
    bin_stats.columns = ['Mean E[V]', 'Mean σ(V)', 'Mean Spread', 'Count']

    print(f"\n{bin_stats.to_string()}")

    # E[V] vs σ(V) relationship
    ev_sigma_corr = hand_df['V_mean'].corr(hand_df['V_std'])
    print(f"\n  Correlation E[V] vs σ(V): {ev_sigma_corr:+.3f}")

    if ev_sigma_corr < 0:
        print(f"  → Better hands have LESS luck component (more predictable)")
    else:
        print(f"  → Better hands have MORE luck component (riskier)")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    hand_df.to_csv(tables_dir / "11y_uncertainty_by_hand.csv", index=False)
    print("✓ Saved per-hand data")

    # Summary
    summary = pd.DataFrame([{
        'n_hands': len(hand_df),
        'n_observations': len(df),
        'grand_mean_v': grand_mean,
        'total_variance': df['V'].var(),
        'pct_hand_component': pct_hand,
        'pct_opponent_component': pct_opponent,
        'r2_features_to_ev': r2,
        'r2_features_to_sigma': r2_luck,
        'true_skill_pct': true_skill,
        'pure_luck_pct': pure_luck,
        'skill_luck_ratio': skill_ratio,
        'ev_sigma_correlation': ev_sigma_corr,
    }])
    summary.to_csv(tables_dir / "11y_uncertainty_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Variance decomposition pie chart
    ax1 = axes[0, 0]
    sizes = [pct_hand, pct_opponent]
    labels = [f'Hand\n(Skill)\n{pct_hand:.0f}%', f'Opponent\n(Luck)\n{pct_opponent:.0f}%']
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax1.set_title('Variance Decomposition')

    # Top right: E[V] vs σ(V) scatter
    ax2 = axes[0, 1]
    ax2.scatter(hand_df['V_mean'], hand_df['V_std'], alpha=0.5, s=30)
    ax2.set_xlabel('E[V] (Hand Quality)')
    ax2.set_ylabel('σ(V) (Luck Component)')
    ax2.set_title(f'Hand Quality vs Luck (r={ev_sigma_corr:.2f})')

    # Add trend line
    z = np.polyfit(hand_df['V_mean'], hand_df['V_std'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(hand_df['V_mean'].min(), hand_df['V_mean'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r--', alpha=0.8)

    # Bottom left: Skill vs Luck detailed breakdown
    ax3 = axes[1, 0]
    categories = ['True Skill', 'Mixed\n(Hand+Luck)', 'Pure Luck']
    values = [true_skill, uncontrolled, pure_luck]
    colors = ['#27ae60', '#f39c12', '#c0392b']
    bars = ax3.bar(categories, values, color=colors, alpha=0.8)
    ax3.set_ylabel('% of Total Variance')
    ax3.set_title('Detailed Uncertainty Decomposition')
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center')

    # Bottom right: σ(V) distribution by hand quality
    ax4 = axes[1, 1]
    for bin_name in ['Strong (>25)', 'Good (10-25)', 'Weak (-10 to 10)', 'Very Weak (<-10)']:
        subset = hand_df[hand_df['ev_bin'] == bin_name]
        if len(subset) > 0:
            ax4.hist(subset['V_std'], bins=20, alpha=0.5, label=bin_name)
    ax4.set_xlabel('σ(V) (Within-hand Variance)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Luck Component by Hand Quality')
    ax4.legend()

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11y_reducible_uncertainty.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. SKILL VS LUCK SPLIT:")
    print(f"   {pct_hand:.0f}% of variance is from hand quality (controllable via bidding)")
    print(f"   {pct_opponent:.0f}% of variance is from opponent distribution (luck)")

    print(f"\n2. TRUE SKILL COMPONENT:")
    print(f"   Hand features explain {r2*100:.0f}% of hand-level variance")
    print(f"   → True skill (predictable from hand): {true_skill:.1f}% of total")

    print(f"\n3. THE LUCK PARADOX:")
    if ev_sigma_corr < -0.2:
        print(f"   Strong hands have LESS luck ({ev_sigma_corr:+.2f} correlation)")
        print(f"   → Good bidding reduces variance AND improves expected value")
        print(f"   → This is the core skill in Texas 42")
    elif ev_sigma_corr > 0.2:
        print(f"   Strong hands have MORE luck ({ev_sigma_corr:+.2f} correlation)")
        print(f"   → High-EV hands are also high-variance (risky)")
    else:
        print(f"   Skill and luck are independent ({ev_sigma_corr:+.2f} correlation)")

    print(f"\n4. BOTTOM LINE:")
    print(f"   Skill/Luck ratio: {skill_ratio*100:.0f}/{(1-skill_ratio)*100:.0f}")
    if skill_ratio > 0.5:
        print(f"   → Texas 42 is MORE SKILL than luck")
    else:
        print(f"   → Texas 42 is MORE LUCK than skill")


if __name__ == "__main__":
    main()
