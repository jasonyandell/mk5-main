# 16: Embeddings & Networks

Word2Vec domino embeddings, interaction matrices, and network visualizations.

> **Epistemic Status**: This report uses machine learning techniques (Word2Vec, UMAP) to find structure in domino co-occurrence and oracle E[V] data. The co-occurrence analysis uses deal distributions (not oracle values). The interaction matrix (16c) uses oracle E[V] from marginalized data. Findings describe patterns in the data; interpretations about "strategic value" are hypotheses.

## 16a: Word2Vec Domino Embeddings

### Key Question
Which dominoes are strategically similar based on hand co-occurrence?

### Method
- Treat each hand as a "sentence" of 7 domino tokens
- Train Word2Vec (skip-gram) on 40,000 hands (10,000 seeds × 4 players)
- Parameters: vector_size=32, window=7, epochs=50, min_count=1
- Analyze cosine similarity between learned embeddings

### Key Findings

#### Doubles Cluster Together (Weakly)

| Comparison | Mean Similarity |
|------------|-----------------|
| Double-to-double | **0.079** |
| Double-to-non-double | 0.071 |
| Random baseline | 0.069 |

Doubles have ~11% higher similarity to each other than to non-doubles, but the effect is subtle.

#### Suit Structure is Weak

| Suit | Intra-suit Similarity | vs Random |
|------|----------------------|-----------|
| 0 (blanks) | 0.077 | +0.008 |
| 1 (aces) | 0.071 | +0.002 |
| 2 (deuces) | 0.073 | +0.004 |
| 3 (treys) | 0.071 | +0.002 |
| 4 (fours) | 0.064 | -0.005 |
| 5 (fives) | 0.055 | -0.014 |
| 6 (sixes) | 0.029 | **-0.040** |

The six-suit shows the weakest intra-suit similarity, possibly because 6-dominoes are distributed across many strong hands.

#### Most Similar Dominoes

| Domino | Most Similar |
|--------|--------------|
| 5-5 | 2-1, 2-0, 4-0, 3-3, 3-2 |
| 6-6 | 0-0, 1-1, 1-0, 3-3, 4-2 |
| 0-0 | 6-6, 4-0, 5-0, 5-3, 6-3 |

The big doubles (5-5, 6-6) show similarity to each other and to the double-blank (0-0).

### Interpretation

**Why is structure weak?**

The random deal mechanism means hands don't have strong "themes":
- You rarely get a hand full of one suit
- Domino co-occurrence is largely random
- The only structure comes from the 7-of-28 sampling constraint

**What the embeddings capture:**
1. Doubles are slightly more likely to co-occur (all act as trick-winners)
2. High-pip dominoes (6-x) don't cluster strongly with each other
3. No clear "archetypes" emerge from co-occurrence alone

### Implications (Co-occurrence Data)

Word2Vec on hand composition reveals that **Texas 42 dominoes are undifferentiated in co-occurrence space**—which dominoes appear together is largely random. This is a property of the dealing mechanism, not a strategic claim.

**Hypothesis**: Strategic value comes from game context (trump selection, who leads), not from co-occurrence patterns. This hypothesis is consistent with the weak embeddings but not directly tested.

### Files Generated

- `results/tables/16a_word2vec_embeddings.csv` - 32D embeddings for all 28 dominoes
- `results/tables/16a_word2vec_similarity.csv` - 28×28 cosine similarity matrix
- `results/models/16a_word2vec.model` - Trained gensim model
- `results/figures/16a_word2vec_tsne.png` - t-SNE visualization
- `results/figures/16a_word2vec_similarity.png` - Similarity heatmap

---

## 16b: UMAP of Domino Embeddings

### Key Question
Do strategic clusters emerge when projecting Word2Vec embeddings to 2D?

### Method
- UMAP projection of 32D Word2Vec embeddings
- Parameters: n_neighbors=5, min_dist=0.3, metric='cosine'
- Colored by: doubles, total pips, blank-suit, six-suit

### Key Findings

#### Weak Clustering

UMAP projection confirms the Word2Vec finding - **no strong clusters emerge**:

1. **Doubles partially cluster**: Red points tend to group, but not tightly
2. **No suit clustering**: Blank-suit and six-suit dominoes are dispersed
3. **No pip gradient**: High/low pip dominoes are scattered

#### Category Separation

Intra-category vs inter-category distances in UMAP space show ratios close to 1.0, indicating categories are not well-separated.

### Interpretation (UMAP Structure)

The random dealing mechanism doesn't create "themed" hands. Dominoes don't develop co-occurrence similarities based on which other dominoes they appear with.

**Hypothesis**: Strategic value comes from game context (trump selection, position), not hand composition. The UMAP projection is consistent with this view but doesn't test it directly—it only shows that co-occurrence structure is weak.

### Files Generated

- `results/figures/16b_umap_dominoes.png` - 2×2 grid visualization
- `results/figures/16b_umap_annotated.png` - Annotated single view
- `results/tables/16b_umap_coordinates.csv` - UMAP coordinates with metadata

---

## 16c: Domino Interaction Matrix

### Key Question
Which domino pairs have synergistic effects on E[V]?

### Method
- **Single effects**: Mean E[V] when domino is present vs absent
- **Pair synergy**: Observed E[V] - Expected (additive model)
- Expected = global_mean + effect(d1) + effect(d2)

### Key Findings

#### Single-Domino Effects (Top 5)

| Domino | Effect on E[V] |
|--------|---------------|
| 4-4 | **+8.21** |
| 5-5 | **+7.67** |
| 5-0 | +6.12 |
| 3-3 | +5.56 |
| 6-6 | +5.24 |

Doubles dominate the top effects - consistent with earlier regression findings.

#### Worst Single Effects

| Domino | Effect on E[V] |
|--------|---------------|
| 6-0 | **-9.55** |
| 4-2 | -5.61 |
| 6-5 | -5.57 |

The 6-0 has a strongly negative effect - it's a weak domino that doesn't win tricks.

#### Pair Synergies

Synergy range: **-11.86 to +14.61**

**Top positive synergies** (better together than expected):
- 4-0 + 5-3: +14.6
- 2-2 + 6-0: +12.0
- 5-0 + 5-1: +10.4

**Top negative synergies** (worse together):
- 2-2 + 3-3: -11.9 (two doubles can conflict)
- 4-0 + 4-2: -11.4
- 2-0 + 5-0: -11.0

### Interpretation (Oracle E[V] Data)

1. **Additive model works mostly**: Most synergies near zero in oracle E[V]
2. **Some non-additive pairs exist**: Range of ±15 points under oracle play
3. **Doubles can conflict**: Having two doubles doesn't always add up (oracle finding)
4. **Sample size limits precision**: With 200 hands, many pairs have few observations

**Note**: These synergies are measured in oracle (minimax) E[V]. Whether the same synergies apply to human play is untested.

### Files Generated

- `results/tables/16c_interaction_matrix.csv` - 28×28 synergy matrix
- `results/tables/16c_pair_synergies.csv` - All pairs ranked by synergy
- `results/tables/16c_single_effects.csv` - Single-domino effects
- `results/figures/16c_interaction_matrix.png` - Heatmap visualization
- `results/figures/16c_synergy_distribution.png` - Synergy histogram

---

## Further Investigation

### Validation Needed

1. **Co-occurrence vs oracle synergy**: The co-occurrence embeddings (16a/b) and interaction matrix (16c) measure different things—one uses deal distributions, the other uses oracle E[V]. A comparison could reveal whether co-occurrence predicts oracle synergy.

2. **Larger sample sizes**: The interaction matrix uses only 200 hands. Many domino pairs have few observations. Larger samples could sharpen the synergy estimates.

3. **Human play validation**: Do the oracle-derived synergies predict actual human gameplay outcomes? This requires human game data.

### Methodological Questions

1. **Word2Vec hyperparameters**: Would different vector_size, window, or epochs reveal more structure? The 32D embedding may be too high for the weak signal present.

2. **Alternative embedding methods**: Would matrix factorization (SVD) or node2vec reveal different structure than Word2Vec skip-gram?

3. **Synergy statistical significance**: The synergy values range ±15, but no confidence intervals are reported. Bootstrap CIs could distinguish real synergies from sampling noise.

### Open Questions

1. **Why do doubles cluster weakly?**: The 11% similarity premium for doubles is small. Is this from true strategic similarity or just the 7-of-28 sampling constraint?

2. **What drives pair synergies?**: The top synergies (+14.6 for 4-0 + 5-3) lack clear strategic explanation. What game mechanism creates these interactions?

3. **Embedding utility**: Can the Word2Vec embeddings improve oracle prediction, or is the structure too weak to be useful?

---

## Remaining Tasks

- 16d: Interaction network visualization
- 16e: Find domino cliques
