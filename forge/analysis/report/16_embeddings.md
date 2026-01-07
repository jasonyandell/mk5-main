# 16: Embeddings & Networks

Word2Vec domino embeddings, interaction matrices, and network visualizations.

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

### Implications

Word2Vec on hand composition reveals that **Texas 42 dominoes are strategically undifferentiated** in terms of which hands they appear in. The strategic value of a domino comes from the game context (trump selection, who leads), not from co-occurrence patterns.

### Files Generated

- `results/tables/16a_word2vec_embeddings.csv` - 32D embeddings for all 28 dominoes
- `results/tables/16a_word2vec_similarity.csv` - 28×28 cosine similarity matrix
- `results/models/16a_word2vec.model` - Trained gensim model
- `results/figures/16a_word2vec_tsne.png` - t-SNE visualization
- `results/figures/16a_word2vec_similarity.png` - Similarity heatmap

---

## Remaining Tasks

- 16b: UMAP of domino embeddings
- 16c: Domino interaction matrix
- 16d: Interaction network visualization
- 16e: Find domino cliques
