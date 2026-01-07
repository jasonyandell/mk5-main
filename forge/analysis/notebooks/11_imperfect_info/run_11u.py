#!/usr/bin/env python3
"""
11u: Hand Ranking by Risk-Adjusted Value

Question: Which hands are objectively strongest?
Method: Rank by E[V] - λ×σ(V) for various λ (risk aversion levels)
What It Reveals: Optimal bidding order considering both expected value and risk

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
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
from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Full analysis
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


def hand_to_str(hand: list[int]) -> str:
    """Convert hand to readable string."""
    dominoes = []
    for d in sorted(hand, reverse=True):
        high, low = schema.domino_pips(d)
        dominoes.append(f"{high}-{low}")
    return " ".join(dominoes)


def extract_hand_features(hand: list[int], trump_suit: int) -> dict:
    """Extract features from a hand."""
    n_doubles = sum(1 for d in hand if schema.domino_pips(d)[0] == schema.domino_pips(d)[1])

    count_points = sum(tables.DOMINO_COUNT_POINTS[d] for d in hand if tables.DOMINO_COUNT_POINTS[d] > 0)

    trump_count = sum(1 for d in hand if trump_suit in schema.domino_pips(d))

    has_trump_double = any(
        schema.domino_pips(d) == (trump_suit, trump_suit) for d in hand
    )

    n_6_high = sum(1 for d in hand if schema.domino_pips(d)[0] == 6)

    total_pips = sum(sum(schema.domino_pips(d)) for d in hand)

    return {
        'n_doubles': n_doubles,
        'count_points': count_points,
        'trump_count': trump_count,
        'has_trump_double': has_trump_double,
        'n_6_high': n_6_high,
        'total_pips': total_pips
    }


def analyze_for_base_seed(base_seed: int) -> dict | None:
    """Analyze one base seed across all opponent configs."""
    decl_id = base_seed % 10
    trump_suit = decl_id
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

    V_mean = np.mean(V_values)
    V_std = np.std(V_values)
    V_min = min(V_values)
    V_max = max(V_values)
    V_spread = V_max - V_min

    # Hand features
    feats = extract_hand_features(p0_hand, trump_suit)

    # Risk-adjusted utilities for various λ
    # λ = 0: Risk-neutral
    # λ = 0.5: Mildly risk-averse
    # λ = 1: Standard risk penalty
    # λ = 2: Highly risk-averse
    util_0 = V_mean
    util_05 = V_mean - 0.5 * V_std
    util_1 = V_mean - 1.0 * V_std
    util_2 = V_mean - 2.0 * V_std

    return {
        'base_seed': base_seed,
        'decl_id': decl_id,
        'hand_str': hand_to_str(p0_hand),
        'V_mean': V_mean,
        'V_std': V_std,
        'V_min': V_min,
        'V_max': V_max,
        'V_spread': V_spread,
        'util_lambda_0': util_0,
        'util_lambda_05': util_05,
        'util_lambda_1': util_1,
        'util_lambda_2': util_2,
        **feats
    }


def main():
    print("=" * 60)
    print("HAND RANKING BY RISK-ADJUSTED VALUE")
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
        result = analyze_for_base_seed(base_seed)
        if result:
            all_results.append(result)

    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    # Ranking analysis
    print("\n" + "=" * 60)
    print("TOP 10 HANDS BY E[V] (λ=0, Risk-Neutral)")
    print("=" * 60)

    top_ev = df.nlargest(10, 'util_lambda_0')
    for i, (_, row) in enumerate(top_ev.iterrows(), 1):
        print(f"{i:2d}. E[V]={row['V_mean']:+5.1f}, σ={row['V_std']:4.1f}, "
              f"Utility={row['util_lambda_0']:+5.1f} | {row['hand_str']}")

    print("\n" + "=" * 60)
    print("TOP 10 HANDS BY RISK-ADJUSTED VALUE (λ=1)")
    print("=" * 60)

    top_risk_adj = df.nlargest(10, 'util_lambda_1')
    for i, (_, row) in enumerate(top_risk_adj.iterrows(), 1):
        print(f"{i:2d}. E[V]={row['V_mean']:+5.1f}, σ={row['V_std']:4.1f}, "
              f"Utility={row['util_lambda_1']:+5.1f} | {row['hand_str']}")

    print("\n" + "=" * 60)
    print("TOP 10 HANDS BY RISK-ADJUSTED VALUE (λ=2, Highly Risk-Averse)")
    print("=" * 60)

    top_very_risk = df.nlargest(10, 'util_lambda_2')
    for i, (_, row) in enumerate(top_very_risk.iterrows(), 1):
        print(f"{i:2d}. E[V]={row['V_mean']:+5.1f}, σ={row['V_std']:4.1f}, "
              f"Utility={row['util_lambda_2']:+5.1f} | {row['hand_str']}")

    # Ranking changes
    print("\n" + "=" * 60)
    print("RANKING STABILITY ACROSS λ VALUES")
    print("=" * 60)

    df['rank_0'] = df['util_lambda_0'].rank(ascending=False)
    df['rank_05'] = df['util_lambda_05'].rank(ascending=False)
    df['rank_1'] = df['util_lambda_1'].rank(ascending=False)
    df['rank_2'] = df['util_lambda_2'].rank(ascending=False)

    # Correlation between rankings
    rank_corr_0_1 = df['rank_0'].corr(df['rank_1'], method='spearman')
    rank_corr_0_2 = df['rank_0'].corr(df['rank_2'], method='spearman')
    rank_corr_1_2 = df['rank_1'].corr(df['rank_2'], method='spearman')

    print(f"\n  Spearman rank correlations:")
    print(f"    λ=0 vs λ=1: {rank_corr_0_1:.3f}")
    print(f"    λ=0 vs λ=2: {rank_corr_0_2:.3f}")
    print(f"    λ=1 vs λ=2: {rank_corr_1_2:.3f}")

    # Hands that move most in ranking
    df['rank_change_0_to_1'] = df['rank_0'] - df['rank_1']

    print("\n  Hands that IMPROVE most when accounting for risk:")
    most_improved = df.nlargest(5, 'rank_change_0_to_1')
    for _, row in most_improved.iterrows():
        print(f"    +{row['rank_change_0_to_1']:.0f} ranks: E[V]={row['V_mean']:+.0f}, σ={row['V_std']:.0f}")

    print("\n  Hands that WORSEN most when accounting for risk:")
    most_worsened = df.nsmallest(5, 'rank_change_0_to_1')
    for _, row in most_worsened.iterrows():
        print(f"    {row['rank_change_0_to_1']:+.0f} ranks: E[V]={row['V_mean']:+.0f}, σ={row['V_std']:.0f}")

    # Dominated hands analysis
    print("\n" + "=" * 60)
    print("DOMINATED HANDS ANALYSIS")
    print("=" * 60)

    # A hand is dominated if there exists another hand with higher E[V] AND lower σ(V)
    n_dominated = 0
    dominated_hands = []

    for idx, row in df.iterrows():
        # Check if any other hand dominates this one
        dominators = df[
            (df['V_mean'] > row['V_mean']) &
            (df['V_std'] < row['V_std'])
        ]
        if len(dominators) > 0:
            n_dominated += 1
            dominated_hands.append({
                'idx': idx,
                'V_mean': row['V_mean'],
                'V_std': row['V_std'],
                'n_dominators': len(dominators)
            })

    print(f"\n  Total dominated hands: {n_dominated} / {len(df)} ({n_dominated/len(df)*100:.1f}%)")
    print(f"  Non-dominated (Pareto-optimal) hands: {len(df) - n_dominated}")

    # Pareto frontier characteristics
    pareto_df = df[~df.index.isin([d['idx'] for d in dominated_hands])]
    print(f"\n  Pareto-optimal hand statistics:")
    print(f"    Mean E[V]: {pareto_df['V_mean'].mean():+.1f}")
    print(f"    Mean σ(V): {pareto_df['V_std'].mean():.1f}")
    print(f"    Mean doubles: {pareto_df['n_doubles'].mean():.1f}")
    print(f"    Mean trump count: {pareto_df['trump_count'].mean():.1f}")

    # Bidding threshold analysis
    print("\n" + "=" * 60)
    print("BIDDING THRESHOLDS")
    print("=" * 60)

    # What utility threshold suggests bidding?
    # Bid if E[V] > 25 (need ~25 point margin to make 30)
    for lambda_val, col in [(0, 'util_lambda_0'), (1, 'util_lambda_1'), (2, 'util_lambda_2')]:
        bid_hands = df[df[col] >= 25]
        bid_pct = len(bid_hands) / len(df) * 100
        print(f"\n  λ={lambda_val}: {len(bid_hands)} hands ({bid_pct:.0f}%) would bid (utility ≥ 25)")
        if len(bid_hands) > 0:
            print(f"    Avg E[V]: {bid_hands['V_mean'].mean():+.1f}")
            print(f"    Avg σ(V): {bid_hands['V_std'].mean():.1f}")
            print(f"    Avg doubles: {bid_hands['n_doubles'].mean():.1f}")

    # Feature correlations with rankings
    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS WITH UTILITY")
    print("=" * 60)

    for col in ['n_doubles', 'trump_count', 'count_points', 'n_6_high', 'total_pips']:
        corr_0 = df[col].corr(df['util_lambda_0'])
        corr_1 = df[col].corr(df['util_lambda_1'])
        print(f"  {col}: λ=0: {corr_0:+.3f}, λ=1: {corr_1:+.3f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Full rankings
    df_sorted = df.sort_values('util_lambda_1', ascending=False)
    df_sorted.to_csv(tables_dir / "11u_hand_rankings.csv", index=False)
    print("✓ Saved full rankings")

    # Top hands summary
    top_summary = pd.DataFrame({
        'rank_lambda_0': df.nlargest(20, 'util_lambda_0').index.tolist(),
        'rank_lambda_1': df.nlargest(20, 'util_lambda_1').index.tolist(),
        'rank_lambda_2': df.nlargest(20, 'util_lambda_2').index.tolist(),
    })
    top_summary.to_csv(tables_dir / "11u_top_hands.csv", index=False)
    print("✓ Saved top hands")

    # Summary statistics
    summary = pd.DataFrame([{
        'n_hands': len(df),
        'n_dominated': n_dominated,
        'n_pareto_optimal': len(df) - n_dominated,
        'pct_bid_lambda_0': (df['util_lambda_0'] >= 25).sum() / len(df) * 100,
        'pct_bid_lambda_1': (df['util_lambda_1'] >= 25).sum() / len(df) * 100,
        'pct_bid_lambda_2': (df['util_lambda_2'] >= 25).sum() / len(df) * 100,
        'rank_corr_0_vs_1': rank_corr_0_1,
        'rank_corr_0_vs_2': rank_corr_0_2,
        'mean_ev': df['V_mean'].mean(),
        'mean_sigma': df['V_std'].mean()
    }])
    summary.to_csv(tables_dir / "11u_ranking_summary.csv", index=False)
    print("✓ Saved summary statistics")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: E[V] vs σ(V) scatter with Pareto frontier
    ax1 = axes[0, 0]
    non_pareto = df[df.index.isin([d['idx'] for d in dominated_hands])]
    ax1.scatter(pareto_df['V_std'], pareto_df['V_mean'],
                color='green', alpha=0.8, s=60, label='Pareto-optimal', zorder=3)
    ax1.scatter(non_pareto['V_std'], non_pareto['V_mean'],
                color='red', alpha=0.4, s=40, label='Dominated', zorder=2)
    ax1.axhline(y=25, color='blue', linestyle='--', label='Bid threshold')
    ax1.set_xlabel('σ(V) - Risk')
    ax1.set_ylabel('E[V] - Expected Value')
    ax1.set_title('Hand Quality: E[V] vs Risk')
    ax1.legend()

    # Top right: Utility distributions for different λ
    ax2 = axes[0, 1]
    ax2.hist(df['util_lambda_0'], bins=20, alpha=0.5, label='λ=0 (risk-neutral)', color='blue')
    ax2.hist(df['util_lambda_1'], bins=20, alpha=0.5, label='λ=1 (standard)', color='green')
    ax2.hist(df['util_lambda_2'], bins=20, alpha=0.5, label='λ=2 (risk-averse)', color='red')
    ax2.axvline(x=25, color='black', linestyle='--', label='Bid threshold')
    ax2.set_xlabel('Utility (E[V] - λσ(V))')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Utility Distribution by Risk Aversion')
    ax2.legend()

    # Bottom left: Rank changes
    ax3 = axes[1, 0]
    ax3.scatter(df['rank_0'], df['rank_1'], alpha=0.5, s=30)
    ax3.plot([0, 200], [0, 200], 'r--', label='No change')
    ax3.set_xlabel('Rank (λ=0, risk-neutral)')
    ax3.set_ylabel('Rank (λ=1, risk-adjusted)')
    ax3.set_title(f'Rank Change: Risk-Neutral vs Risk-Adjusted (ρ={rank_corr_0_1:.2f})')
    ax3.legend()

    # Bottom right: Feature importance for utility
    ax4 = axes[1, 1]
    features_list = ['n_doubles', 'trump_count', 'count_points', 'n_6_high', 'total_pips']
    corrs_0 = [df[f].corr(df['util_lambda_0']) for f in features_list]
    corrs_1 = [df[f].corr(df['util_lambda_1']) for f in features_list]
    x = np.arange(len(features_list))
    width = 0.35
    ax4.bar(x - width/2, corrs_0, width, label='λ=0', color='blue', alpha=0.7)
    ax4.bar(x + width/2, corrs_1, width, label='λ=1', color='green', alpha=0.7)
    ax4.set_ylabel('Correlation with Utility')
    ax4.set_title('Feature Importance by Risk Aversion')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Doubles', 'Trumps', 'Count Pts', '6-Highs', 'Total Pips'], rotation=15)
    ax4.legend()
    ax4.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11u_hand_ranking.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. RANKING STABILITY:")
    print(f"   λ=0 vs λ=1 rank correlation: {rank_corr_0_1:.3f}")
    if rank_corr_0_1 > 0.9:
        print(f"   → Rankings are VERY STABLE across risk preferences")
    elif rank_corr_0_1 > 0.7:
        print(f"   → Rankings are MODERATELY STABLE")
    else:
        print(f"   → Rankings CHANGE SIGNIFICANTLY with risk aversion")

    print(f"\n2. DOMINATED HANDS:")
    print(f"   {n_dominated} / {len(df)} ({n_dominated/len(df)*100:.1f}%) are dominated")
    print(f"   → {len(df) - n_dominated} hands are Pareto-optimal (no better E[V] with lower risk)")

    print(f"\n3. BIDDING RECOMMENDATIONS:")
    for lambda_val, col in [(0, 'util_lambda_0'), (1, 'util_lambda_1')]:
        bid_pct = (df[col] >= 25).sum() / len(df) * 100
        print(f"   λ={lambda_val}: {bid_pct:.0f}% of hands justify bidding")

    print(f"\n4. RISK-RETURN RELATIONSHIP:")
    ev_sigma_corr = df['V_mean'].corr(df['V_std'])
    print(f"   E[V] vs σ(V) correlation: {ev_sigma_corr:+.3f}")
    if ev_sigma_corr < -0.3:
        print(f"   → NEGATIVE correlation: Good hands are SAFER (no risk-return tradeoff!)")
    elif ev_sigma_corr < 0.3:
        print(f"   → Weak correlation: Risk and return are independent")
    else:
        print(f"   → Positive correlation: Better hands are riskier")


if __name__ == "__main__":
    main()
