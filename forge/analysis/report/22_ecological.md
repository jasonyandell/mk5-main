# 22: Ecological Analysis

Applying ecological diversity metrics to hand composition.

## 22a: Alpha Diversity per Hand

### Key Question
Does "strategic flexibility" (suit coverage diversity) predict hand value?

### Method
- Compute suit distribution for each hand (counts of 0-6 pips)
- Calculate Shannon entropy (alpha diversity) in bits
- Correlate with E[V] and σ(V)

### Diversity Metrics

**Shannon Entropy** (alpha diversity):
```
H = -Σ p_i × log₂(p_i)
```
Where p_i is the proportion of dominoes covering suit i.

**Evenness**:
```
E = H / H_max
```
Where H_max = log₂(7) ≈ 2.81 bits.

### Key Findings

#### Diversity Statistics

| Metric | Mean | Std | Range |
|--------|------|-----|-------|
| Alpha diversity | 2.50 bits | 0.16 | [1.96, 2.75] |
| Evenness | 0.89 | 0.06 | [0.70, 0.98] |
| Suits present | 6.2 | 0.6 | [4, 7] |

Most hands cover 6 suits (out of 7) with fairly even distribution.

#### Correlations with E[V]

| Feature | r | p-value | Interpretation |
|---------|---|---------|----------------|
| Alpha diversity | **-0.205** | 0.004 | Small negative |
| Evenness | -0.205 | 0.004 | Same as entropy |
| n_suits | -0.200 | 0.004 | Small negative |
| max_suit_count | +0.137 | 0.053 | Marginal positive |

**Key insight**: Higher diversity (more even suit spread) is **negatively** correlated with E[V].

#### Correlations with σ(V)

| Feature | r | p-value | Interpretation |
|---------|---|---------|----------------|
| Alpha diversity | +0.035 | 0.62 | Not significant |
| Evenness | +0.035 | 0.62 | Not significant |

Diversity does not predict risk.

### Interpretation

**Why is diversity negatively correlated with E[V]?**

1. **Doubles reduce diversity**: Having 2+ doubles means fewer unique suits represented
2. **Doubles predict E[V]**: The napkin formula shows doubles are the strongest predictor
3. **Therefore**: High diversity → fewer doubles → lower E[V]

**Ecological analogy**: In ecological terms, "specialist" hands (concentrated in doubles/trumps) outperform "generalist" hands (even suit spread).

### Implications for Bidding

- **Don't value "balanced" hands**: Suit coverage diversity is not advantageous
- **Doubles are better than coverage**: A hand with 3 doubles beats a hand covering all 7 suits
- **Flexibility ≠ strength**: Being able to follow any suit doesn't translate to winning

### Files Generated

- `results/tables/22a_alpha_diversity.csv` - Per-hand diversity metrics
- `results/figures/22a_alpha_diversity.png` - 4-panel visualization

---

## 22b: Co-occurrence Matrix

### Key Question
Which dominoes tend to appear together in winning vs losing hands?

### Method
- Build 28×28 co-occurrence matrix from hand compositions
- Compare winner hands (E[V] > 0) vs loser hands (E[V] < 0)
- Compute enrichment ratio: (winner count) / (loser count)

### Key Findings

**Top Positive Co-occurrences (Winners)**:

| Domino 1 | Domino 2 | Winner | Loser | Enrichment |
|----------|----------|--------|-------|------------|
| 4-4 + 5-5 | | 8 | 0 | 10.0 |
| 5-5 + 6-1 | | 7 | 0 | 10.0 |
| 3-3 + 5-4 | | 6 | 0 | 10.0 |
| 5-0 + 6-6 | | 6 | 0 | 10.0 |
| 4-0 + 4-4 | | 9 | 1 | 9.18 |
| 3-3 + 5-5 | | 9 | 1 | 9.18 |

**Key pattern:** Double-double pairs dominate winner hands. The 4-4 + 5-5 combination appears in 8 winners and 0 losers.

**Top Negative Co-occurrences (Losers)**:

| Domino 1 | Domino 2 | Winner | Loser | Enrichment |
|----------|----------|--------|-------|------------|
| 4-2 + 6-0 | | 0 | 9 | 0.0 |
| 5-4 + 6-2 | | 0 | 4 | 0.0 |
| 3-0 + 6-0 | | 0 | 9 | 0.0 |

**Key pattern:** 6-0 paired with non-doubles appears in losers.

### Enrichment Distribution

| Enrichment | n_pairs | Interpretation |
|------------|---------|----------------|
| > 5.0 | 41 | Strong winner signal |
| 2.0-5.0 | 89 | Moderate winner |
| 0.5-2.0 | 127 | Near-random |
| < 0.5 | 48 | Loser signal |
| = 0.0 | 69 | Only in losers |

### Interpretation

1. **Doubles cluster together**: Double-double pairs are heavily enriched in winners
2. **6-0 is toxic**: Pairing 6-0 with other dominoes predicts losing
3. **Strategic pairs exist**: Despite random dealing, certain combinations win more often
4. **Count dominoes together**: 5-5 + 5-0 (15 count points together) enrichment = 8.16

### Files Generated

- `results/tables/22b_cooccurrence_pairs.csv` - All pair enrichments
- `results/tables/22b_cooccurrence_matrices.npz` - Full 28×28 matrices
- `results/figures/22b_cooccurrence.png` - Heatmap visualization

---

## Summary

Ecological analysis reveals:

1. **Diversity hurts E[V]**: More evenly spread hands have lower expected value (r = -0.21)
2. **Specialists win**: Concentrated holdings (doubles) outperform balanced coverage
3. **No diversity-risk link**: Suit diversity doesn't predict outcome variance
4. **Random co-occurrence**: Domino pairings follow sampling statistics, not strategic structure
