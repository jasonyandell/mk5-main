#!/usr/bin/env python3
"""
11r: Manifold Collapse Analysis

Question: How many effective dimensions per hand?
Method: Measure intrinsic dimensionality when hand is fixed
What It Reveals: Strong hands collapse to lower dimensionality

Approach:
- For each hand, build depth × config matrix of V values
- Compute variance decomposition (between-config, between-depth, residual)
- Measure effective rank/participation ratio
- Correlate with hand strength (E[V]) to test: "strong hands collapse"
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from forge.analysis.utils.seed_db import SeedDB

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"
N_BASE_SEEDS = 100  # Expanded from 50 for better statistics
np.random.seed(42)

# Depth levels for analysis
DEPTH_LEVELS = [28, 24, 20, 16, 12, 8, 4, 1]


def load_depth_v_means(db: SeedDB, path: Path) -> dict | None:
    """Load mean V at each depth level via SQL GROUP BY."""
    depth_list = ",".join(str(d) for d in DEPTH_LEVELS)

    sql = f"""
    SELECT depth(state) as depth, AVG(CAST(V AS DOUBLE)) as mean_v
    FROM read_parquet('{path}')
    WHERE depth(state) IN ({depth_list})
    GROUP BY depth(state)
    """

    try:
        result = db.execute(sql)
        df = result.data
        if df is None or len(df) == 0:
            return None

        result_dict = {}
        for _, row in df.iterrows():
            result_dict[int(row['depth'])] = float(row['mean_v'])

        return result_dict if len(result_dict) >= 4 else None
    except Exception:
        return None


def compute_manifold_metrics(db: SeedDB, base_seed: int) -> dict | None:
    """Compute manifold collapse metrics for one hand."""
    decl_id = base_seed % 10

    # Load depth-V profiles for all 3 configs
    profiles = []
    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None
        profile = load_depth_v_means(db, path)
        if profile is None:
            return None
        profiles.append(profile)

    if len(profiles) != 3:
        return None

    # Find common depths
    common_depths = set(profiles[0].keys())
    for p in profiles[1:]:
        common_depths &= set(p.keys())

    if len(common_depths) < 4:
        return None

    sorted_depths = sorted(common_depths, reverse=True)

    # Build depth × config matrix (n_depths × 3)
    matrix = np.zeros((len(sorted_depths), 3))
    for i, d in enumerate(sorted_depths):
        for j, p in enumerate(profiles):
            matrix[i, j] = p[d]

    # Basic statistics
    root_vs = matrix[0, :]  # V at depth 28
    mean_v = np.mean(root_vs)
    std_v = np.std(root_vs)
    v_spread = np.max(root_vs) - np.min(root_vs)

    # Variance decomposition (ANOVA-style)
    grand_mean = np.mean(matrix)

    # Between-config variance (how much V varies by opponent config)
    config_means = np.mean(matrix, axis=0)  # Mean per config
    between_config_var = np.var(config_means)

    # Between-depth variance (how much V varies by depth)
    depth_means = np.mean(matrix, axis=1)  # Mean per depth
    between_depth_var = np.var(depth_means)

    # Total variance
    total_var = np.var(matrix)

    # Residual variance (interaction term)
    residual_var = max(0, total_var - between_config_var - between_depth_var)

    # Effective rank via participation ratio of singular values
    # SVD of centered matrix
    centered = matrix - grand_mean
    try:
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        S_sq = S ** 2
        if np.sum(S_sq) > 0:
            participation_ratio = (np.sum(S_sq) ** 2) / np.sum(S_sq ** 2)
        else:
            participation_ratio = 1.0
    except Exception:
        participation_ratio = 1.0

    # Effective dimensionality measures
    # 1. Config dimension: variance explained by config differences
    config_dim_ratio = between_config_var / total_var if total_var > 0 else 0

    # 2. Depth dimension: variance explained by depth progression
    depth_dim_ratio = between_depth_var / total_var if total_var > 0 else 0

    # 3. Collapse metric: 1 - config_dim (higher = more collapsed)
    collapse_score = 1 - config_dim_ratio

    # 4. Trajectory coherence: correlation of depth profiles across configs
    correlations = []
    for i in range(3):
        for j in range(i + 1, 3):
            if np.std(matrix[:, i]) > 0 and np.std(matrix[:, j]) > 0:
                corr = np.corrcoef(matrix[:, i], matrix[:, j])[0, 1]
                correlations.append(corr)
    mean_trajectory_corr = np.mean(correlations) if correlations else 0

    return {
        'base_seed': base_seed,
        'mean_v': mean_v,
        'std_v': std_v,
        'v_spread': v_spread,
        'total_var': total_var,
        'between_config_var': between_config_var,
        'between_depth_var': between_depth_var,
        'residual_var': residual_var,
        'config_dim_ratio': config_dim_ratio,
        'depth_dim_ratio': depth_dim_ratio,
        'collapse_score': collapse_score,
        'participation_ratio': participation_ratio,
        'mean_trajectory_corr': mean_trajectory_corr,
        'n_depths': len(sorted_depths),
    }


def main():
    print("=" * 60)
    print("MANIFOLD COLLAPSE ANALYSIS")
    print("Intrinsic Dimensionality per Hand")
    print("=" * 60)

    # Initialize SeedDB
    db = SeedDB(DATA_DIR)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    # Collect results
    all_results = []
    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = compute_manifold_metrics(db, base_seed)
        if result:
            all_results.append(result)

    db.close()
    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("MANIFOLD COLLAPSE SUMMARY")
    print("=" * 60)

    print(f"\n  Variance Decomposition:")
    print(f"    Total variance: {df['total_var'].mean():.1f}")
    print(f"    Between-config variance: {df['between_config_var'].mean():.1f} ({df['config_dim_ratio'].mean()*100:.1f}%)")
    print(f"    Between-depth variance: {df['between_depth_var'].mean():.1f} ({df['depth_dim_ratio'].mean()*100:.1f}%)")
    print(f"    Residual variance: {df['residual_var'].mean():.1f}")

    print(f"\n  Effective Dimensionality:")
    print(f"    Mean participation ratio: {df['participation_ratio'].mean():.2f}")
    print(f"    Mean collapse score: {df['collapse_score'].mean():.3f}")
    print(f"    Mean trajectory correlation: {df['mean_trajectory_corr'].mean():.3f}")

    # Key correlations
    print("\n" + "=" * 60)
    print("COLLAPSE VS HAND STRENGTH")
    print("=" * 60)

    corr_collapse_v = df['collapse_score'].corr(df['mean_v'])
    corr_partip_v = df['participation_ratio'].corr(df['mean_v'])
    corr_trajcorr_v = df['mean_trajectory_corr'].corr(df['mean_v'])
    corr_configvar_v = df['config_dim_ratio'].corr(df['mean_v'])

    print(f"\n  Correlations with E[V]:")
    print(f"    collapse_score vs E[V]: {corr_collapse_v:+.3f}")
    print(f"    participation_ratio vs E[V]: {corr_partip_v:+.3f}")
    print(f"    trajectory_correlation vs E[V]: {corr_trajcorr_v:+.3f}")
    print(f"    config_dim_ratio vs E[V]: {corr_configvar_v:+.3f}")

    # Strong vs weak hands
    print("\n" + "=" * 60)
    print("STRONG VS WEAK HANDS")
    print("=" * 60)

    high_ev = df[df['mean_v'] > df['mean_v'].quantile(0.75)]
    low_ev = df[df['mean_v'] < df['mean_v'].quantile(0.25)]

    print(f"\n  High E[V] hands (top 25%, n={len(high_ev)}):")
    print(f"    Mean E[V]: {high_ev['mean_v'].mean():+.1f}")
    print(f"    Mean collapse score: {high_ev['collapse_score'].mean():.3f}")
    print(f"    Mean trajectory corr: {high_ev['mean_trajectory_corr'].mean():.3f}")
    print(f"    Mean config dim ratio: {high_ev['config_dim_ratio'].mean():.3f}")

    print(f"\n  Low E[V] hands (bottom 25%, n={len(low_ev)}):")
    print(f"    Mean E[V]: {low_ev['mean_v'].mean():+.1f}")
    print(f"    Mean collapse score: {low_ev['collapse_score'].mean():.3f}")
    print(f"    Mean trajectory corr: {low_ev['mean_trajectory_corr'].mean():.3f}")
    print(f"    Mean config dim ratio: {low_ev['config_dim_ratio'].mean():.3f}")

    # The key finding
    collapse_diff = high_ev['collapse_score'].mean() - low_ev['collapse_score'].mean()
    config_diff = low_ev['config_dim_ratio'].mean() - high_ev['config_dim_ratio'].mean()

    print(f"\n  DIFFERENCE:")
    print(f"    Collapse score: strong hands have {collapse_diff:+.3f} higher collapse")
    print(f"    Config variance: weak hands have {config_diff:.1%} more config-dependent variance")

    # Collapse categories
    print("\n" + "=" * 60)
    print("COLLAPSE CATEGORIES")
    print("=" * 60)

    highly_collapsed = df[df['collapse_score'] > 0.8]
    moderately_collapsed = df[(df['collapse_score'] > 0.5) & (df['collapse_score'] <= 0.8)]
    not_collapsed = df[df['collapse_score'] <= 0.5]

    print(f"\n  Highly collapsed (score > 0.8): {len(highly_collapsed)} ({len(highly_collapsed)/len(df)*100:.1f}%)")
    print(f"    Mean E[V]: {highly_collapsed['mean_v'].mean():+.1f}")

    print(f"\n  Moderately collapsed (0.5-0.8): {len(moderately_collapsed)} ({len(moderately_collapsed)/len(df)*100:.1f}%)")
    print(f"    Mean E[V]: {moderately_collapsed['mean_v'].mean():+.1f}")

    print(f"\n  Not collapsed (score ≤ 0.5): {len(not_collapsed)} ({len(not_collapsed)/len(df)*100:.1f}%)")
    print(f"    Mean E[V]: {not_collapsed['mean_v'].mean():+.1f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11r_manifold_collapse_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Summary
    summary = pd.DataFrame([{
        'n_hands': len(df),
        'mean_total_var': df['total_var'].mean(),
        'mean_config_dim_ratio': df['config_dim_ratio'].mean(),
        'mean_depth_dim_ratio': df['depth_dim_ratio'].mean(),
        'mean_collapse_score': df['collapse_score'].mean(),
        'mean_participation_ratio': df['participation_ratio'].mean(),
        'mean_trajectory_corr': df['mean_trajectory_corr'].mean(),
        'corr_collapse_vs_ev': corr_collapse_v,
        'corr_configdim_vs_ev': corr_configvar_v,
        'pct_highly_collapsed': len(highly_collapsed) / len(df) * 100,
        'pct_not_collapsed': len(not_collapsed) / len(df) * 100,
    }])
    summary.to_csv(tables_dir / "11r_manifold_collapse_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Collapse score vs E[V]
    ax1 = axes[0, 0]
    ax1.scatter(df['mean_v'], df['collapse_score'], alpha=0.6, s=40)
    ax1.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='High collapse')
    ax1.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Low collapse')
    ax1.set_xlabel('E[V] (Expected Value)')
    ax1.set_ylabel('Collapse Score')
    ax1.set_title(f'Manifold Collapse vs Hand Strength (r={corr_collapse_v:.2f})')
    ax1.legend()

    # Top right: Config dim ratio histogram
    ax2 = axes[0, 1]
    ax2.hist(df['config_dim_ratio'], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(df['config_dim_ratio'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["config_dim_ratio"].mean():.2f}')
    ax2.set_xlabel('Config Dimension Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Variance Explained by Opponent Configuration')
    ax2.legend()

    # Bottom left: Trajectory correlation vs E[V]
    ax3 = axes[1, 0]
    colors = ['green' if s > 0.8 else 'orange' if s > 0.5 else 'red' for s in df['collapse_score']]
    ax3.scatter(df['mean_v'], df['mean_trajectory_corr'], c=colors, alpha=0.6, s=40)
    ax3.set_xlabel('E[V]')
    ax3.set_ylabel('Mean Trajectory Correlation')
    ax3.set_title(f'Trajectory Coherence vs Hand Strength (r={corr_trajcorr_v:.2f})')

    # Bottom right: Variance decomposition bar chart
    ax4 = axes[1, 1]
    variance_components = ['Config\nVariance', 'Depth\nVariance', 'Residual']
    variance_values = [
        df['config_dim_ratio'].mean() * 100,
        df['depth_dim_ratio'].mean() * 100,
        (1 - df['config_dim_ratio'].mean() - df['depth_dim_ratio'].mean()) * 100
    ]
    ax4.bar(variance_components, variance_values, color=['red', 'blue', 'gray'], alpha=0.7)
    ax4.set_ylabel('Variance Explained (%)')
    ax4.set_title('Average Variance Decomposition')

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11r_manifold_collapse.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. VARIANCE DECOMPOSITION:")
    print(f"   Config variance explains {df['config_dim_ratio'].mean()*100:.1f}% of total V variance")
    print(f"   Depth variance explains {df['depth_dim_ratio'].mean()*100:.1f}%")
    print(f"   → Most variance is due to depth progression, not opponent config")

    print(f"\n2. COLLAPSE HYPOTHESIS:")
    if corr_collapse_v > 0.1:
        print(f"   ✓ CONFIRMED: Strong hands collapse MORE (r = {corr_collapse_v:+.3f})")
        print(f"   Top 25% hands have {collapse_diff:+.3f} higher collapse score")
    elif corr_collapse_v < -0.1:
        print(f"   ✗ REVERSED: Strong hands collapse LESS (r = {corr_collapse_v:+.3f})")
    else:
        print(f"   ~ WEAK: Collapse and strength weakly related (r = {corr_collapse_v:+.3f})")

    print(f"\n3. TRAJECTORY COHERENCE:")
    print(f"   Mean trajectory correlation: {df['mean_trajectory_corr'].mean():.3f}")
    if df['mean_trajectory_corr'].mean() > 0.5:
        print(f"   → V trajectories are SIMILAR across opponent configs")
    else:
        print(f"   → V trajectories DIVERGE based on opponent config")

    print(f"\n4. PRACTICAL IMPLICATION:")
    print(f"   {len(highly_collapsed)/len(df)*100:.0f}% of hands are 'highly collapsed'")
    print(f"   These hands have predictable outcomes regardless of opponents")


if __name__ == "__main__":
    main()
