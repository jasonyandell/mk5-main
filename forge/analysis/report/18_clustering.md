# 18: Clustering & Archetypes

K-means clustering to define empirical hand archetypes.

## 18a: K-Means Hand Archetypes

### Key Question
Can we discover natural "hand types" from the feature space?

### Method
- Standardized 10 regression features
- K-means clustering with silhouette analysis to find optimal k
- Profile clusters by feature means and E[V]/σ(V)

### Key Findings

#### Optimal K

Silhouette analysis suggests **k=2** clusters:

| k | Silhouette | Inertia |
|---|------------|---------|
| 2 | **0.191** | 1749.4 |
| 3 | 0.135 | 1542.5 |
| 4 | 0.138 | 1397.9 |
| 5 | 0.144 | 1285.4 |

The relatively low silhouette scores (max ~0.19) indicate that hand space is **continuous** rather than having sharp clusters.

#### Cluster Profiles (k=2)

| Cluster | Archetype | n | Mean E[V] | Mean σ(V) | n_doubles | trump_count |
|---------|-----------|---|-----------|-----------|-----------|-------------|
| 0 | Strong Balanced | 34 | 22.7 | 13.2 | 2.21 | 2.38 |
| 1 | Average | 166 | 12.1 | 15.5 | 1.63 | 1.10 |

### Interpretation

**Two-cluster structure**:
1. **Strong Balanced (17%)**: High doubles, high trumps, high E[V], lower variance
2. **Average (83%)**: Modal hand type with moderate features

The clusters separate along the **n_doubles** and **trump_count** axes, consistent with the napkin formula findings.

### Files Generated

- `results/tables/18a_cluster_assignments.csv` - Per-hand cluster
- `results/tables/18a_cluster_profiles.csv` - Cluster statistics
- `results/figures/18a_kmeans_selection.png` - Silhouette/elbow analysis
- `results/figures/18a_cluster_profiles.png` - Profile visualization

---

## 18b: Marker Dominoes per Archetype

### Key Question
Which dominoes are characteristic "markers" for each cluster?

### Method
- Compute domino frequency per cluster
- Fisher's exact test for enrichment
- log₂(enrichment) = log₂(freq_cluster / freq_other)

### Key Findings

#### Cluster 0 (Strong Balanced) Markers

**Enriched** (more common):
| Domino | Cluster Freq | Other Freq | log₂ |
|--------|--------------|------------|------|
| 5-1 | 41% | 19% | +1.09 |
| 6-6 | 38% | 20% | +0.94 |
| 5-5 | 47% | 30% | +0.67 |

**Depleted** (less common):
| Domino | Cluster Freq | Other Freq | log₂ |
|--------|--------------|------------|------|
| 4-2 | 12% | 32% | -1.44 |
| 6-3 | 12% | 27% | -1.17 |
| 1-0 | 12% | 26% | -1.14 |

### Interpretation

Strong hands are enriched in **doubles** (5-5, 6-6) and depleted in weak middle cards (4-2, 6-3). This confirms that cluster separation is driven by the same features identified in regression analysis.

### Files Generated

- `results/tables/18b_marker_dominoes.csv` - Enrichment by cluster
- `results/tables/18b_domino_freq_by_cluster.csv` - Frequency matrix
- `results/figures/18b_marker_heatmap.png` - Heatmap visualization

---

## 18c: Hierarchical Clustering Dendrogram

### Key Question
Does hierarchical clustering reveal nested structure?

### Method
- Ward linkage on standardized features
- Sampled 50 hands across E[V] distribution for visualization
- Compared with K-means assignments

### Key Findings

1. **Dendrogram structure**: Major splits correspond to E[V] levels
2. **K-means agreement**: Cross-tabulation shows reasonable concordance between methods
3. **Nested hierarchy**: Feature importance (doubles > trumps) reflected in branch structure

### Interpretation

Hierarchical clustering confirms that hand space has continuous gradients rather than discrete types. The dendrogram is useful for understanding which hands are most similar, but "archetypes" are convenient labels on a continuum.

### Files Generated

- `results/figures/18c_dendrogram.png` - Vertical dendrogram
- `results/figures/18c_dendrogram_horizontal.png` - Horizontal dendrogram

---

## Summary

Clustering analysis reveals:

1. **Continuous hand space**: Low silhouette scores indicate gradual transitions, not sharp boundaries
2. **Two broad categories**: Strong (high doubles/trumps) vs Average hands
3. **Marker dominoes**: Doubles (5-5, 6-6) characterize strong hands
4. **Consistent with regression**: Clusters separate along the same axes as significant predictors
