# 18: Clustering & Archetypes

K-means clustering to define empirical hand archetypes.

> **Epistemic Status**: This report clusters hands based on features derived from oracle (minimax) E[V] and σ(V). The "archetypes" are groups in oracle feature space, not necessarily categories that human players would recognize or find useful. Whether these clusters predict human gameplay outcomes is untested.

## 18a: K-Means Hand Archetypes

### Key Question
Can we discover natural "hand types" from the oracle feature space?

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

### Interpretation (Oracle Feature Space)

**Two-cluster structure in oracle data**:
1. **High Oracle E[V] (17%)**: High doubles, high trumps, high oracle E[V], lower oracle variance
2. **Modal (83%)**: Typical hand with moderate features

The clusters separate along the **n_doubles** and **trump_count** axes, consistent with the napkin formula findings from oracle regression. **Note**: These labels describe oracle-predicted outcomes. Whether human players experience these as distinct "hand types" is untested.

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

### Interpretation (Oracle Markers)

High oracle E[V] hands are enriched in **doubles** (5-5, 6-6) and depleted in middle cards (4-2, 6-3). This confirms that cluster separation is driven by the same features identified in oracle regression analysis.

**Note**: The enrichment is defined relative to oracle E[V]. Whether these dominoes are "markers" for human-perceived hand quality is untested.

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

### Interpretation (Oracle Similarity)

Hierarchical clustering confirms that oracle feature space has continuous gradients rather than discrete types. The dendrogram shows which hands are most similar in oracle E[V] terms, but "archetypes" are convenient labels on a continuum.

**Note**: Hands that are "similar" in oracle feature space may not feel similar to human players with hidden information.

### Files Generated

- `results/figures/18c_dendrogram.png` - Vertical dendrogram
- `results/figures/18c_dendrogram_horizontal.png` - Horizontal dendrogram

---

## Summary (Oracle Feature Space)

Clustering analysis of oracle data reveals:

1. **Continuous oracle feature space**: Low silhouette scores indicate gradual transitions, not sharp boundaries
2. **Two broad categories**: High oracle E[V] (high doubles/trumps) vs Modal hands
3. **Marker dominoes**: Doubles (5-5, 6-6) characterize high oracle E[V] hands
4. **Consistent with oracle regression**: Clusters separate along the same axes as significant predictors

**Scope limitation**: These clusters describe structure in oracle data. Whether they correspond to human-perceived "hand types" or predict human gameplay outcomes remains untested.

---

## Further Investigation

### Validation Needed

1. **Human perception study**: Do experienced players recognize the two clusters as distinct "hand types"? Survey or interview data could test whether oracle-derived archetypes match human intuition.

2. **Human gameplay validation**: Do the clusters predict human game outcomes, or only oracle outcomes? This requires human gameplay data.

3. **More clusters**: The k=2 solution has low silhouette (0.19). Would k=3 or k=4 be more interpretable despite lower silhouette?

### Methodological Questions

1. **Feature selection**: The clustering uses 10 regression features. Would a different feature set reveal different structure?

2. **Standardization effects**: K-means is sensitive to scaling. Would robust scaling or different normalization change the clusters?

3. **Sample size**: With only 200 hands, cluster stability is uncertain. Bootstrap analysis could quantify cluster robustness.

### Open Questions

1. **Archetype utility**: Are these archetypes *useful* for gameplay decisions, or merely descriptive? A practical test would be whether knowing your archetype helps you bid better.

2. **Declaration-specific archetypes**: Do the clusters differ by trump declaration? A hand's "type" may depend on what trumps are.

3. **Continuous vs categorical**: The low silhouette suggests a continuum. Would a regression-based approach (predict E[V] directly) be more useful than categorical archetypes?
