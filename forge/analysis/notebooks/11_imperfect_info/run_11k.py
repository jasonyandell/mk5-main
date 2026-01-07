#!/usr/bin/env python3
"""
11k: Hand Classification Clustering

Question: Can we cluster hands by outcome profile?
Method: K-means on (E[V], σ(V), basin_count) vectors
What It Reveals: "Strong", "weak", "volatile" hand types

Uses data from 11j basin variance analysis.
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"
np.random.seed(42)


def main():
    print("=" * 60)
    print("HAND CLASSIFICATION CLUSTERING")
    print("=" * 60)

    # Load 11j data
    df = pd.read_csv(RESULTS_DIR / "tables" / "11j_basin_variance_by_seed.csv")
    print(f"Loaded {len(df)} hands from 11j analysis")

    # Feature vectors for clustering
    features = ['V_mean', 'V_std', 'n_unique_basins']
    X = df[features].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k using silhouette score
    print("\n" + "=" * 60)
    print("FINDING OPTIMAL CLUSTER COUNT")
    print("=" * 60)

    silhouette_scores = []
    K_range = range(2, 8)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"  k={k}: silhouette={score:.3f}")

    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n  Best k={best_k} (silhouette={max(silhouette_scores):.3f})")

    # Run final clustering with best k
    print("\n" + "=" * 60)
    print(f"CLUSTERING WITH K={best_k}")
    print("=" * 60)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Also try fixed k=3 for interpretability
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_3'] = kmeans_3.fit_predict(X_scaled)

    # Analyze clusters
    print("\n" + "=" * 60)
    print("CLUSTER PROFILES (k=3 for interpretability)")
    print("=" * 60)

    cluster_stats = df.groupby('cluster_3').agg({
        'V_mean': ['mean', 'std', 'min', 'max'],
        'V_std': ['mean', 'std'],
        'n_unique_basins': ['mean', 'median'],
        'V_spread': ['mean', 'std'],
        'n_doubles': 'mean',
        'trump_count': 'mean',
        'has_trump_double': 'mean',
        'count_points': 'mean',
        'base_seed': 'count'
    }).round(2)

    # Flatten column names
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
    print(cluster_stats)

    # Name the clusters based on characteristics
    print("\n" + "=" * 60)
    print("CLUSTER INTERPRETATION")
    print("=" * 60)

    cluster_summary = df.groupby('cluster_3').agg({
        'V_mean': 'mean',
        'V_std': 'mean',
        'n_unique_basins': 'mean',
        'V_spread': 'mean',
        'base_seed': 'count'
    }).round(1)

    # Sort by V_mean to get consistent naming
    cluster_order = cluster_summary.sort_values('V_mean', ascending=False).index.tolist()

    cluster_names = {}
    for i, cluster_id in enumerate(cluster_order):
        stats = cluster_summary.loc[cluster_id]
        if i == 0:  # Highest E[V]
            name = "STRONG"
            desc = f"High E[V] (+{stats['V_mean']:.0f}), low variance"
        elif i == len(cluster_order) - 1:  # Lowest E[V]
            name = "WEAK"
            desc = f"Low E[V] ({stats['V_mean']:+.0f}), high variance"
        else:
            name = "VOLATILE"
            desc = f"Medium E[V] ({stats['V_mean']:+.0f}), high spread"
        cluster_names[cluster_id] = name
        print(f"\nCluster {cluster_id} → {name}")
        print(f"  E[V]: {stats['V_mean']:+.1f}")
        print(f"  σ(V): {stats['V_std']:.1f}")
        print(f"  Basins: {stats['n_unique_basins']:.1f}")
        print(f"  Spread: {stats['V_spread']:.0f}")
        print(f"  Count: {int(stats['base_seed'])} hands ({int(stats['base_seed'])/len(df)*100:.0f}%)")

    df['cluster_name'] = df['cluster_3'].map(cluster_names)

    # Feature analysis by cluster
    print("\n" + "=" * 60)
    print("HAND FEATURES BY CLUSTER")
    print("=" * 60)

    feature_cols = ['n_doubles', 'trump_count', 'has_trump_double', 'count_points', 'n_6_high', 'total_pips']
    for cluster_id in cluster_order:
        name = cluster_names[cluster_id]
        subset = df[df['cluster_3'] == cluster_id]
        print(f"\n{name} hands (n={len(subset)}):")
        for col in feature_cols:
            print(f"  {col}: {subset[col].mean():.2f}")

    # Sample hands from each cluster
    print("\n" + "=" * 60)
    print("SAMPLE HANDS BY CLUSTER")
    print("=" * 60)

    for cluster_id in cluster_order:
        name = cluster_names[cluster_id]
        subset = df[df['cluster_3'] == cluster_id].sample(min(3, len(df[df['cluster_3'] == cluster_id])), random_state=42)
        print(f"\n{name} ({len(df[df['cluster_3'] == cluster_id])} total):")
        for _, row in subset.iterrows():
            print(f"  E[V]={row['V_mean']:+5.1f}, σ={row['V_std']:4.1f}, basins={int(row['n_unique_basins'])}: {row['hand_str']}")

    # Bidding recommendations by cluster
    print("\n" + "=" * 60)
    print("BIDDING RECOMMENDATIONS")
    print("=" * 60)

    for cluster_id in cluster_order:
        name = cluster_names[cluster_id]
        subset = df[df['cluster_3'] == cluster_id]
        mean_ev = subset['V_mean'].mean()
        mean_std = subset['V_std'].mean()

        if name == "STRONG":
            rec = "BID CONFIDENTLY: High EV, low variance. Bid 30-42."
        elif name == "WEAK":
            rec = "PASS: Low EV with high variance. Not worth the risk."
        else:
            rec = "CAUTIOUS BID OR PASS: Medium EV but outcomes vary widely."

        print(f"\n{name}: {rec}")
        print(f"  Expected outcome: {mean_ev:+.1f} ± {mean_std:.1f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-hand clustering
    df.to_csv(tables_dir / "11k_hand_classification.csv", index=False)
    print("✓ Saved per-hand clustering")

    # Cluster summary
    summary_df = df.groupby('cluster_name').agg({
        'V_mean': ['mean', 'std', 'min', 'max'],
        'V_std': ['mean', 'std'],
        'n_unique_basins': ['mean'],
        'V_spread': ['mean'],
        'n_doubles': 'mean',
        'trump_count': 'mean',
        'has_trump_double': 'mean',
        'count_points': 'mean',
        'base_seed': 'count'
    }).round(2)
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns]
    summary_df.to_csv(tables_dir / "11k_cluster_summary.csv")
    print("✓ Saved cluster summary")

    # Silhouette scores
    silhouette_df = pd.DataFrame({
        'k': list(K_range),
        'silhouette': silhouette_scores
    })
    silhouette_df.to_csv(tables_dir / "11k_silhouette_scores.csv", index=False)
    print("✓ Saved silhouette scores")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color map for clusters
    colors = {'STRONG': 'green', 'VOLATILE': 'orange', 'WEAK': 'red'}
    color_list = [colors[name] for name in df['cluster_name']]

    # Top left: E[V] vs σ(V) colored by cluster
    ax1 = axes[0, 0]
    for name in ['STRONG', 'VOLATILE', 'WEAK']:
        subset = df[df['cluster_name'] == name]
        ax1.scatter(subset['V_mean'], subset['V_std'], c=colors[name],
                   label=f'{name} (n={len(subset)})', alpha=0.7, s=50)
    ax1.set_xlabel('E[V]')
    ax1.set_ylabel('σ(V)')
    ax1.set_title('Hands by E[V] vs σ(V)')
    ax1.legend()

    # Top right: E[V] vs Basins colored by cluster
    ax2 = axes[0, 1]
    for name in ['STRONG', 'VOLATILE', 'WEAK']:
        subset = df[df['cluster_name'] == name]
        # Add jitter to basin count for visibility
        jitter = np.random.uniform(-0.15, 0.15, len(subset))
        ax2.scatter(subset['V_mean'], subset['n_unique_basins'] + jitter,
                   c=colors[name], label=name, alpha=0.7, s=50)
    ax2.set_xlabel('E[V]')
    ax2.set_ylabel('Unique Basins')
    ax2.set_title('Hands by E[V] vs Basin Count')
    ax2.set_yticks([1, 2, 3])

    # Bottom left: Cluster distributions
    ax3 = axes[1, 0]
    cluster_counts = df['cluster_name'].value_counts()[['STRONG', 'VOLATILE', 'WEAK']]
    ax3.bar(cluster_counts.index, cluster_counts.values,
           color=[colors[n] for n in cluster_counts.index])
    ax3.set_ylabel('Number of Hands')
    ax3.set_title('Cluster Distribution')
    for i, (name, count) in enumerate(cluster_counts.items()):
        ax3.text(i, count + 2, f'{count/len(df)*100:.0f}%', ha='center')

    # Bottom right: Silhouette scores
    ax4 = axes[1, 1]
    ax4.plot(K_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax4.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    ax4.axvline(x=3, color='g', linestyle='--', label='k=3 (interpretable)')
    ax4.set_xlabel('Number of Clusters (k)')
    ax4.set_ylabel('Silhouette Score')
    ax4.set_title('Cluster Quality by k')
    ax4.legend()

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11k_hand_classification.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    strong = df[df['cluster_name'] == 'STRONG']
    weak = df[df['cluster_name'] == 'WEAK']
    volatile = df[df['cluster_name'] == 'VOLATILE']

    print(f"\n1. THREE NATURAL HAND TYPES:")
    print(f"   STRONG: {len(strong)} hands ({len(strong)/len(df)*100:.0f}%) - E[V] {strong['V_mean'].mean():+.1f}")
    print(f"   VOLATILE: {len(volatile)} hands ({len(volatile)/len(df)*100:.0f}%) - E[V] {volatile['V_mean'].mean():+.1f}")
    print(f"   WEAK: {len(weak)} hands ({len(weak)/len(df)*100:.0f}%) - E[V] {weak['V_mean'].mean():+.1f}")

    print(f"\n2. STRONG HANDS PROFILE:")
    print(f"   Avg doubles: {strong['n_doubles'].mean():.1f}")
    print(f"   Avg trumps: {strong['trump_count'].mean():.1f}")
    print(f"   Trump double %: {strong['has_trump_double'].mean()*100:.0f}%")

    print(f"\n3. BIDDING RULE OF THUMB:")
    print(f"   {len(strong)/len(df)*100:.0f}% of hands are STRONG (bid confidently)")
    print(f"   {len(volatile)/len(df)*100:.0f}% of hands are VOLATILE (bid cautiously)")
    print(f"   {len(weak)/len(df)*100:.0f}% of hands are WEAK (pass)")


if __name__ == "__main__":
    main()
