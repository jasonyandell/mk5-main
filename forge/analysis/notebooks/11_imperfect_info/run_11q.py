#!/usr/bin/env python3
"""
11q: Per-Hand PCA Analysis

Question: Is 5D structure preserved within fixed hand?
Method: PCA on V-trajectories to find intrinsic dimensionality
What It Reveals: Does fixing P0's hand constrain the outcome manifold?

Approach:
- For each hand, extract V stats at multiple depth levels across 3 opponent configs
- Build feature matrix with depth-level V means and spreads
- PCA to find how many dimensions explain variance
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from forge.analysis.utils import features

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 50  # Preliminary
np.random.seed(42)

# Depth levels to analyze
DEPTH_LEVELS = [28, 24, 20, 16, 12, 8, 4, 1]


def load_depth_v_profile(path: Path) -> dict | None:
    """Load V distribution stats at each depth level."""
    try:
        pf = pq.ParquetFile(path)
        depth_stats = {d: [] for d in DEPTH_LEVELS}

        for batch in pf.iter_batches(batch_size=100000, columns=['state', 'V']):
            states = batch['state'].to_numpy()
            V = batch['V'].to_numpy()
            depths = features.depth(states)

            for d in DEPTH_LEVELS:
                mask = depths == d
                if mask.any():
                    depth_stats[d].extend(V[mask].tolist())

        profile = {}
        for d in DEPTH_LEVELS:
            vs = depth_stats[d]
            if len(vs) > 0:
                profile[d] = {
                    'mean_v': float(np.mean(vs)),
                    'std_v': float(np.std(vs)),
                    'min_v': float(np.min(vs)),
                    'max_v': float(np.max(vs)),
                    'n_states': len(vs)
                }

        return profile if len(profile) > 2 else None
    except Exception:
        return None


def extract_hand_features(base_seed: int) -> dict | None:
    """Extract feature vector for one hand across opponent configs."""
    decl_id = base_seed % 10

    # Load profiles for all 3 configs
    profiles = []
    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None
        profile = load_depth_v_profile(path)
        if profile is None:
            return None
        profiles.append(profile)
        gc.collect()

    if len(profiles) != 3:
        return None

    # Build feature dict
    feature_dict = {'base_seed': base_seed}

    for d in DEPTH_LEVELS:
        if all(d in p for p in profiles):
            # Mean V across configs at this depth
            means = [p[d]['mean_v'] for p in profiles]
            stds = [p[d]['std_v'] for p in profiles]

            feature_dict[f'v_mean_d{d}'] = np.mean(means)
            feature_dict[f'v_std_d{d}'] = np.mean(stds)
            feature_dict[f'v_spread_d{d}'] = max(means) - min(means)

    return feature_dict


def main():
    print("=" * 60)
    print("PER-HAND PCA ANALYSIS")
    print("Intrinsic Dimensionality of V-Space")
    print("=" * 60)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds (preliminary)...")

    # Extract features
    all_features = []
    for base_seed in tqdm(sample_seeds, desc="Processing"):
        feature_dict = extract_hand_features(base_seed)
        if feature_dict:
            all_features.append(feature_dict)

    print(f"\n✓ Extracted features for {len(all_features)} hands")

    if len(all_features) == 0:
        print("No features extracted")
        return

    df = pd.DataFrame(all_features)

    # Prepare feature matrix for PCA
    feature_cols = [c for c in df.columns if c.startswith('v_')]
    X = df[feature_cols].values

    # Check for NaN
    if np.any(np.isnan(X)):
        print("Warning: NaN values found, filling with column means")
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    print("\n" + "=" * 60)
    print("PCA ANALYSIS")
    print("=" * 60)

    pca = PCA()
    pca.fit(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"\n  Feature dimensions: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")

    print("\n  Explained Variance by Component:")
    for i, (var, cum) in enumerate(zip(explained_var[:10], cumulative_var[:10])):
        print(f"    PC{i+1}: {var*100:5.1f}% (cumulative: {cum*100:5.1f}%)")

    # Find components needed for 90% and 95% variance
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_99 = np.argmax(cumulative_var >= 0.99) + 1

    print(f"\n  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")
    print(f"  Components for 99% variance: {n_99}")

    # Effective dimensionality
    eff_dim = np.exp(-np.sum(explained_var * np.log(explained_var + 1e-10)))
    print(f"\n  Effective dimensionality (participation ratio): {eff_dim:.1f}")

    # Loadings analysis
    print("\n" + "=" * 60)
    print("PC LOADINGS (Top Components)")
    print("=" * 60)

    loadings = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=feature_cols
    )

    print("\n  PC1 Top Loadings:")
    pc1_sorted = loadings['PC1'].abs().sort_values(ascending=False)
    for feat in pc1_sorted.head(5).index:
        print(f"    {feat}: {loadings.loc[feat, 'PC1']:+.3f}")

    print("\n  PC2 Top Loadings:")
    pc2_sorted = loadings['PC2'].abs().sort_values(ascending=False)
    for feat in pc2_sorted.head(5).index:
        print(f"    {feat}: {loadings.loc[feat, 'PC2']:+.3f}")

    # Depth-wise analysis
    print("\n" + "=" * 60)
    print("DEPTH-WISE VARIANCE ANALYSIS")
    print("=" * 60)

    for d in DEPTH_LEVELS:
        mean_col = f'v_mean_d{d}'
        spread_col = f'v_spread_d{d}'
        if mean_col in df.columns:
            mean_std = df[mean_col].std()
            spread_mean = df[spread_col].mean()
            print(f"  Depth {d:2d}: V_mean std={mean_std:5.1f}, V_spread mean={spread_mean:5.1f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-hand features
    df.to_csv(tables_dir / "11q_per_hand_pca_features.csv", index=False)
    print("✓ Saved per-hand features")

    # PCA summary
    pca_summary = pd.DataFrame({
        'component': range(1, len(explained_var) + 1),
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var
    })
    pca_summary.to_csv(tables_dir / "11q_pca_variance.csv", index=False)
    print("✓ Saved PCA variance")

    # Loadings
    loadings.to_csv(tables_dir / "11q_pca_loadings.csv")
    print("✓ Saved PCA loadings")

    # Summary
    summary = pd.DataFrame([{
        'n_hands': len(df),
        'n_features': len(feature_cols),
        'n_components_90': n_90,
        'n_components_95': n_95,
        'n_components_99': n_99,
        'effective_dim': eff_dim,
        'pc1_variance': explained_var[0] * 100,
        'pc2_variance': explained_var[1] * 100,
    }])
    summary.to_csv(tables_dir / "11q_pca_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Scree plot
    ax1 = axes[0, 0]
    ax1.bar(range(1, min(11, len(explained_var)+1)), explained_var[:10] * 100, alpha=0.7)
    ax1.plot(range(1, min(11, len(explained_var)+1)), cumulative_var[:10] * 100, 'r.-')
    ax1.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.axhline(95, color='gray', linestyle=':', alpha=0.5, label='95% threshold')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('PCA Scree Plot')
    ax1.legend()

    # Top right: PC1 vs PC2 scatter
    ax2 = axes[0, 1]
    X_pca = pca.transform(X_scaled)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=40)
    ax2.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    ax2.set_title('Hands in PC Space')

    # Bottom left: Loadings heatmap for top components
    ax3 = axes[1, 0]
    loadings_small = loadings.iloc[:min(12, len(loadings)), :min(4, loadings.shape[1])]
    im = ax3.imshow(loadings_small.values.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax3.set_yticks(range(len(loadings_small.columns)))
    ax3.set_yticklabels(loadings_small.columns)
    ax3.set_xticks(range(len(loadings_small)))
    ax3.set_xticklabels(loadings_small.index, rotation=45, ha='right')
    ax3.set_title('PC Loadings')
    plt.colorbar(im, ax=ax3)

    # Bottom right: Variance by depth
    ax4 = axes[1, 1]
    depths = []
    spreads = []
    for d in DEPTH_LEVELS:
        spread_col = f'v_spread_d{d}'
        if spread_col in df.columns:
            depths.append(d)
            spreads.append(df[spread_col].mean())
    ax4.bar(range(len(depths)), spreads, alpha=0.7)
    ax4.set_xticks(range(len(depths)))
    ax4.set_xticklabels([f'd{d}' for d in depths])
    ax4.set_xlabel('Depth Level')
    ax4.set_ylabel('Mean V Spread (across opponent configs)')
    ax4.set_title('V Spread by Game Phase')

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11q_per_hand_pca.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS (PRELIMINARY)")
    print("=" * 60)

    print(f"\n1. INTRINSIC DIMENSIONALITY:")
    print(f"   {n_90} components explain 90% of variance")
    print(f"   Effective dimensionality: {eff_dim:.1f}")
    print(f"   (Original features: {len(feature_cols)})")

    if n_90 <= 3:
        print(f"   → Hand outcomes are LOW-DIMENSIONAL (can be summarized in {n_90} factors)")
    elif n_90 <= 6:
        print(f"   → Hand outcomes are MODERATELY DIMENSIONAL")
    else:
        print(f"   → Hand outcomes are HIGH-DIMENSIONAL (complex structure)")

    print(f"\n2. DOMINANT FACTORS:")
    print(f"   PC1 explains {explained_var[0]*100:.1f}% of variance")
    top_pc1 = pc1_sorted.head(1).index[0]
    print(f"   Dominated by: {top_pc1}")

    print(f"\n3. MANIFOLD CONSTRAINT:")
    compression = len(feature_cols) / n_90
    print(f"   Dimensionality compression: {compression:.1f}x")
    if compression > 2:
        print(f"   → Fixing hand SIGNIFICANTLY constrains outcome space")
    else:
        print(f"   → Fixing hand provides modest constraint")


if __name__ == "__main__":
    main()
