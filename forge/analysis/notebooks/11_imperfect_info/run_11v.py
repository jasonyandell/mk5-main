#!/usr/bin/env python3
"""
11v: Hand Similarity Clustering

Question: Do similar hands have similar outcomes?
Method: Cluster hands by features, compare within-cluster V variance
What It Reveals: Hand equivalence classes - structurally similar hands with similar outcomes

This is the inverse of 11k:
- 11k clustered by (E[V], σ(V), basins) → found STRONG/VOLATILE/WEAK
- 11v clusters by hand FEATURES → checks if similar hands produce similar outcomes
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"
np.random.seed(42)


def main():
    print("=" * 60)
    print("HAND SIMILARITY CLUSTERING")
    print("=" * 60)

    # Load 11j data which has both features and outcomes
    df = pd.read_csv(RESULTS_DIR / "tables" / "11j_basin_variance_by_seed.csv")
    print(f"Loaded {len(df)} hands from 11j analysis")

    # Features for clustering (structural hand properties)
    feature_cols = ['n_doubles', 'trump_count', 'has_trump_double', 'count_points', 'n_6_high', 'total_pips']
    X_features = df[feature_cols].values

    # Outcome variables (what we want to be consistent within clusters)
    outcome_cols = ['V_mean', 'V_std', 'V_spread']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Find optimal k
    print("\n" + "=" * 60)
    print("FINDING OPTIMAL CLUSTER COUNT (by features)")
    print("=" * 60)

    silhouette_scores = []
    K_range = range(3, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"  k={k}: silhouette={score:.3f}")

    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n  Best k={best_k} (silhouette={max(silhouette_scores):.3f})")

    # Use k=5 for interpretability
    USE_K = 5
    print(f"\n  Using k={USE_K} for analysis")

    kmeans = KMeans(n_clusters=USE_K, random_state=42, n_init=10)
    df['feature_cluster'] = kmeans.fit_predict(X_scaled)

    # Analyze clusters
    print("\n" + "=" * 60)
    print(f"CLUSTER ANALYSIS (k={USE_K})")
    print("=" * 60)

    # Cluster profiles by features
    print("\n--- Feature Profiles ---")
    cluster_features = df.groupby('feature_cluster')[feature_cols].mean().round(2)
    print(cluster_features)

    # Cluster profiles by outcomes
    print("\n--- Outcome Profiles ---")
    cluster_outcomes = df.groupby('feature_cluster')[outcome_cols + ['n_unique_basins']].agg(['mean', 'std']).round(2)
    print(cluster_outcomes)

    # Within-cluster variance vs overall variance
    print("\n" + "=" * 60)
    print("WITHIN-CLUSTER V VARIANCE ANALYSIS")
    print("=" * 60)

    overall_v_std = df['V_mean'].std()
    print(f"\nOverall E[V] std: {overall_v_std:.2f}")

    cluster_variance = []
    for c in sorted(df['feature_cluster'].unique()):
        subset = df[df['feature_cluster'] == c]
        v_std = subset['V_mean'].std()
        v_range = subset['V_mean'].max() - subset['V_mean'].min()
        n = len(subset)
        cluster_variance.append({
            'cluster': c,
            'n': n,
            'v_mean': subset['V_mean'].mean(),
            'v_std': v_std,
            'v_range': v_range,
            'variance_reduction': (1 - v_std/overall_v_std) * 100 if overall_v_std > 0 else 0
        })
        print(f"\nCluster {c} (n={n}):")
        print(f"  Mean E[V]: {subset['V_mean'].mean():+.1f}")
        print(f"  Std E[V]: {v_std:.1f} (vs overall {overall_v_std:.1f})")
        print(f"  E[V] range: [{subset['V_mean'].min():+.0f}, {subset['V_mean'].max():+.0f}]")
        print(f"  Variance reduction: {(1 - v_std/overall_v_std) * 100:.0f}%")

    variance_df = pd.DataFrame(cluster_variance)
    avg_reduction = variance_df['variance_reduction'].mean()
    print(f"\n  Average within-cluster variance reduction: {avg_reduction:.0f}%")

    # Name clusters by dominant feature
    print("\n" + "=" * 60)
    print("CLUSTER INTERPRETATION")
    print("=" * 60)

    cluster_names = {}
    for c in sorted(df['feature_cluster'].unique()):
        subset = df[df['feature_cluster'] == c]
        features = cluster_features.loc[c]

        # Determine dominant characteristic
        traits = []
        if features['n_doubles'] >= 2.0:
            traits.append("Multi-Double")
        elif features['n_doubles'] <= 1.2:
            traits.append("Few-Double")

        if features['trump_count'] >= 2.0:
            traits.append("Trump-Heavy")
        elif features['trump_count'] <= 1.0:
            traits.append("Trump-Light")

        if features['count_points'] >= 12:
            traits.append("Count-Rich")
        elif features['count_points'] <= 6:
            traits.append("Count-Poor")

        if features['n_6_high'] >= 2.5:
            traits.append("Six-Heavy")

        name = "/".join(traits) if traits else f"Cluster-{c}"
        cluster_names[c] = name

        print(f"\nCluster {c} → {name}")
        print(f"  E[V]: {subset['V_mean'].mean():+.1f} ± {subset['V_mean'].std():.1f}")
        print(f"  Features: doubles={features['n_doubles']:.1f}, trump={features['trump_count']:.1f}, counts={features['count_points']:.0f}")
        print(f"  Size: {len(subset)} hands ({len(subset)/len(df)*100:.0f}%)")

    df['cluster_name'] = df['feature_cluster'].map(cluster_names)

    # Check if similar hands have similar outcomes
    print("\n" + "=" * 60)
    print("SIMILARITY → OUTCOME CONSISTENCY")
    print("=" * 60)

    # Compare within-cluster E[V] variance to random groupings
    random_stds = []
    for _ in range(100):
        random_labels = np.random.randint(0, USE_K, len(df))
        group_stds = []
        for c in range(USE_K):
            mask = random_labels == c
            if mask.sum() > 1:
                group_stds.append(df.loc[mask, 'V_mean'].std())
        random_stds.append(np.mean(group_stds))

    mean_random_std = np.mean(random_stds)
    actual_mean_std = variance_df['v_std'].mean()

    print(f"\n  Within-cluster E[V] std (feature-based): {actual_mean_std:.2f}")
    print(f"  Within-cluster E[V] std (random): {mean_random_std:.2f}")
    print(f"  Improvement over random: {(1 - actual_mean_std/mean_random_std) * 100:.0f}%")

    if actual_mean_std < mean_random_std * 0.8:
        print("\n  ✓ Similar hands DO have similar outcomes!")
    else:
        print("\n  ✗ Structurally similar hands don't guarantee similar outcomes")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-hand clustering
    df.to_csv(tables_dir / "11v_hand_similarity.csv", index=False)
    print("✓ Saved per-hand clustering")

    # Cluster summary
    summary_df = df.groupby('cluster_name').agg({
        'V_mean': ['mean', 'std', 'min', 'max', 'count'],
        'V_std': 'mean',
        'n_doubles': 'mean',
        'trump_count': 'mean',
        'count_points': 'mean',
    }).round(2)
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns]
    summary_df.to_csv(tables_dir / "11v_cluster_summary.csv")
    print("✓ Saved cluster summary")

    # Variance analysis
    variance_df.to_csv(tables_dir / "11v_variance_analysis.csv", index=False)
    print("✓ Saved variance analysis")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Cluster sizes and E[V]
    ax1 = axes[0, 0]
    cluster_stats = df.groupby('feature_cluster').agg({
        'V_mean': 'mean',
        'base_seed': 'count'
    })
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_stats)))
    bars = ax1.bar(range(len(cluster_stats)), cluster_stats['V_mean'], color=colors)
    ax1.set_xticks(range(len(cluster_stats)))
    ax1.set_xticklabels([cluster_names[i] for i in cluster_stats.index], rotation=45, ha='right')
    ax1.set_ylabel('Mean E[V]')
    ax1.set_title('E[V] by Feature Cluster')
    ax1.axhline(y=df['V_mean'].mean(), color='r', linestyle='--', label='Overall mean')
    ax1.legend()

    # Top right: Within-cluster variance
    ax2 = axes[0, 1]
    ax2.bar(range(len(variance_df)), variance_df['variance_reduction'], color='green', alpha=0.7)
    ax2.set_xticks(range(len(variance_df)))
    ax2.set_xticklabels([cluster_names[i] for i in variance_df['cluster']], rotation=45, ha='right')
    ax2.set_ylabel('Variance Reduction (%)')
    ax2.set_title('Within-Cluster Variance Reduction')
    ax2.axhline(y=avg_reduction, color='r', linestyle='--', label=f'Avg={avg_reduction:.0f}%')
    ax2.legend()

    # Bottom left: Doubles vs E[V] colored by cluster
    ax3 = axes[1, 0]
    for c in sorted(df['feature_cluster'].unique()):
        subset = df[df['feature_cluster'] == c]
        ax3.scatter(subset['n_doubles'] + np.random.uniform(-0.15, 0.15, len(subset)),
                   subset['V_mean'], alpha=0.5, label=cluster_names[c], s=40)
    ax3.set_xlabel('Number of Doubles')
    ax3.set_ylabel('E[V]')
    ax3.set_title('Doubles vs E[V] by Cluster')
    ax3.legend(fontsize=8)

    # Bottom right: E[V] distribution by cluster
    ax4 = axes[1, 1]
    cluster_labels = [cluster_names[c] for c in sorted(df['feature_cluster'].unique())]
    data = [df[df['feature_cluster'] == c]['V_mean'].values for c in sorted(df['feature_cluster'].unique())]
    ax4.boxplot(data, labels=cluster_labels)
    ax4.set_ylabel('E[V]')
    ax4.set_title('E[V] Distribution by Cluster')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11v_hand_similarity.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. FEATURE-BASED CLUSTERING:")
    for c, name in cluster_names.items():
        n = len(df[df['feature_cluster'] == c])
        print(f"   {name}: {n} hands ({n/len(df)*100:.0f}%)")

    print(f"\n2. OUTCOME CONSISTENCY:")
    print(f"   Average within-cluster variance reduction: {avg_reduction:.0f}%")
    print(f"   (Feature clustering explains {avg_reduction:.0f}% of E[V] variance)")

    print(f"\n3. HAND EQUIVALENCE:")
    if avg_reduction > 30:
        print(f"   ✓ Hands with similar features DO produce similar outcomes")
        print(f"   → Feature-based heuristics are valid for bidding")
    else:
        print(f"   ✗ Feature similarity alone doesn't guarantee outcome similarity")
        print(f"   → Need to account for opponent distributions")


if __name__ == "__main__":
    main()
