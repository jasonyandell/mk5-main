# 22: Ecological Analysis

Applying ecological diversity metrics to hand composition.

> **Epistemic Status**: This report applies ecological diversity metrics to hand composition and correlates with oracle (minimax) E[V] and σ(V). All findings describe correlations with oracle outcomes. The terms "winners" and "losers" refer to oracle-predicted outcomes (E[V] > 0 vs E[V] < 0), not human gameplay results. Bidding implications are hypotheses extrapolated from oracle correlations.

## 22a: Alpha Diversity per Hand

### Key Question
Does suit coverage diversity predict oracle hand value?

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

#### Correlations with Oracle E[V]

| Feature | r | p-value | Interpretation |
|---------|---|---------|----------------|
| Alpha diversity | **-0.205** | 0.004 | Small negative |
| Evenness | -0.205 | 0.004 | Same as entropy |
| n_suits | -0.200 | 0.004 | Small negative |
| max_suit_count | +0.137 | 0.053 | Marginal positive |

**Key insight**: Higher diversity (more even suit spread) is **negatively** correlated with oracle E[V].

#### Correlations with σ(V)

| Feature | r | p-value | Interpretation |
|---------|---|---------|----------------|
| Alpha diversity | +0.035 | 0.62 | Not significant |
| Evenness | +0.035 | 0.62 | Not significant |

Diversity does not predict risk.

### Interpretation (Oracle Correlation)

**Why is diversity negatively correlated with oracle E[V]?**

1. **Doubles reduce diversity**: Having 2+ doubles means fewer unique suits represented
2. **Doubles predict oracle E[V]**: The napkin formula shows doubles are the strongest predictor
3. **Therefore**: High diversity → fewer doubles → lower oracle E[V]

**Ecological analogy**: In oracle outcome terms, "specialist" hands (concentrated in doubles/trumps) outperform "generalist" hands (even suit spread).

### Hypothetical Implications for Bidding

The following are hypotheses extrapolated from oracle correlations. **None have been validated against human gameplay.**

- **Hypothesis**: "Balanced" hands may not be advantageous despite intuition
- **Hypothesis**: Doubles may be better than suit coverage for winning
- **Hypothesis**: Ability to follow any suit may not translate to winning in practice

**Note**: Human strategic considerations (flexibility to respond, information hiding) are not captured by oracle E[V].

### Files Generated

- `results/tables/22a_alpha_diversity.csv` - Per-hand diversity metrics
- `results/figures/22a_alpha_diversity.png` - 4-panel visualization

---

## 22b: Co-occurrence Matrix

### Key Question
Which dominoes tend to appear together in oracle-winning vs oracle-losing hands?

### Method
- Build 28×28 co-occurrence matrix from hand compositions
- Compare oracle winners (E[V] > 0) vs oracle losers (E[V] < 0)
- Compute enrichment ratio: (oracle winner count) / (oracle loser count)

### Key Findings

**Top Positive Co-occurrences (Oracle Winners)**:

| Domino 1 | Domino 2 | Winner | Loser | Enrichment |
|----------|----------|--------|-------|------------|
| 4-4 + 5-5 | | 8 | 0 | 10.0 |
| 5-5 + 6-1 | | 7 | 0 | 10.0 |
| 3-3 + 5-4 | | 6 | 0 | 10.0 |
| 5-0 + 6-6 | | 6 | 0 | 10.0 |
| 4-0 + 4-4 | | 9 | 1 | 9.18 |
| 3-3 + 5-5 | | 9 | 1 | 9.18 |

**Key pattern:** Double-double pairs dominate oracle winner hands. The 4-4 + 5-5 combination appears in 8 oracle winners and 0 oracle losers.

**Top Negative Co-occurrences (Oracle Losers)**:

| Domino 1 | Domino 2 | Winner | Loser | Enrichment |
|----------|----------|--------|-------|------------|
| 4-2 + 6-0 | | 0 | 9 | 0.0 |
| 5-4 + 6-2 | | 0 | 4 | 0.0 |
| 3-0 + 6-0 | | 0 | 9 | 0.0 |

**Key pattern:** 6-0 paired with non-doubles appears in oracle losers.

### Enrichment Distribution

| Enrichment | n_pairs | Interpretation |
|------------|---------|----------------|
| > 5.0 | 41 | Strong oracle winner signal |
| 2.0-5.0 | 89 | Moderate oracle winner |
| 0.5-2.0 | 127 | Near-random |
| < 0.5 | 48 | Oracle loser signal |
| = 0.0 | 69 | Only in oracle losers |

### Interpretation (Oracle Outcomes)

1. **Doubles cluster in oracle winners**: Double-double pairs are heavily enriched in oracle winners
2. **6-0 predicts oracle loss**: Pairing 6-0 with other dominoes predicts oracle losing
3. **Certain combinations win more in oracle**: Despite random dealing, certain combinations have higher oracle E[V]
4. **Count dominoes together**: 5-5 + 5-0 (15 count points together) enrichment = 8.16 in oracle data

**Note**: Whether these co-occurrence patterns predict human gameplay outcomes is untested.

### Files Generated

- `results/tables/22b_cooccurrence_pairs.csv` - All pair enrichments
- `results/tables/22b_cooccurrence_matrices.npz` - Full 28×28 matrices
- `results/figures/22b_cooccurrence.png` - Heatmap visualization

---

## Summary (Oracle Correlations)

Ecological analysis of oracle data reveals:

1. **Diversity hurts oracle E[V]**: More evenly spread hands have lower oracle expected value (r = -0.21)
2. **"Specialists" win in oracle**: Concentrated holdings (doubles) outperform balanced coverage in oracle outcomes
3. **No diversity-risk link**: Suit diversity doesn't predict oracle outcome variance
4. **Co-occurrence effects in oracle**: Certain domino pairings (especially double-double pairs) have higher oracle E[V]

**Scope limitation**: These patterns describe oracle (perfect-information) outcomes. Whether suit diversity affects human gameplay outcomes—where flexibility to respond may have value—is untested.

---

## Further Investigation

### Validation Needed

1. **Human gameplay validation**: Does suit diversity correlate with human game outcomes? Human strategic considerations (flexibility, information hiding) may differ from oracle dynamics.

2. **Co-occurrence in human play**: Do double-double pairs predict human wins as strongly as oracle wins?

3. **Expert player interviews**: Do experienced 42 players value "balanced" hands? If so, there's a discrepancy between oracle analysis and human intuition worth exploring.

### Methodological Questions

1. **Diversity metric choice**: Shannon entropy is one diversity measure. Would Simpson's diversity or other metrics reveal different patterns?

2. **Sample size for co-occurrence**: With 200 hands, many domino pairs have few observations. The "10.0 enrichment" (8-0 split) may be sampling noise. Larger samples needed.

3. **Multiple testing**: With 378 possible pairs, many enrichments may be significant by chance. FDR correction needed.

### Open Questions

1. **Why would diversity hurt oracle E[V]?**: The explanation (doubles reduce diversity, doubles predict E[V]) is correlational. Is there a causal mechanism in the oracle game tree?

2. **Human flexibility value**: In human play, ability to follow any suit may help gather information or avoid revealing voids. Does this value not show up in oracle analysis?

3. **Ecological validity**: How meaningful is applying ecological diversity metrics to a 7-domino hand? Is there a better framework for measuring "hand concentration"?
