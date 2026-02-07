# 17: Differential Analysis

Comparing winners vs losers to identify distinguishing features.

> **Epistemic Status**: This report compares domino frequencies between high and low oracle E[V] hands, and between high and low oracle σ(V) hands. All findings describe oracle (minimax) outcomes. The terms "winners" and "losers" refer to oracle-predicted outcomes, not human gameplay results. Interpretations about why certain dominoes are enriched are hypotheses.

## 17a: Winner vs Loser Enrichment

### Key Question
Which dominoes are over/under-represented in top 25% oracle E[V] hands?

### Method
- Split hands: Top 25% E[V] (winners) vs Bottom 25% (losers)
- Fisher's exact test for each domino's 2×2 contingency table
- BH FDR correction for multiple testing
- log₂(enrichment) = log₂(freq_winners / freq_losers)

### Key Findings

#### Significant Dominoes (FDR < 0.05)

Only 2 dominoes survive multiple testing correction:

| Domino | Winner Freq | Loser Freq | log₂(Enrichment) | p_adj |
|--------|-------------|------------|------------------|-------|
| **5-5** | 50% | 17.6% | **+1.50** | 0.017 |
| **6-0** | 16% | 47.1% | **-1.56** | 0.017 |

#### Interpretation (Oracle Data)

**5-5 (double-five)**:
- 2.8× more common in oracle winners than oracle losers
- High trump, 10 count points
- **Hypothesis**: Wins tricks reliably under oracle play, leading to positive E[V]

**6-0 (six-blank)**:
- 3× more common in oracle losers than winners
- **Hypothesis**: High suit rank but often loses to doubles/trumps under oracle play

#### Other Notable Trends (not significant after correction)

Enriched in winners (p < 0.05 uncorrected):
- 4-4, 3-3, 6-6 - other doubles

Depleted in winners (p < 0.05 uncorrected):
- 4-2, 6-2 - weak middle cards

### Consistency with Other Analyses

The enrichment pattern aligns with:
- **16c single effects**: 5-5 at +7.67, 6-0 at -9.55
- **Regression**: n_doubles predicts E[V]
- **SHAP**: Doubles have highest feature importance

### Why Few Significant Results?

With only 200 hands:
- Winners: ~50 hands
- Losers: ~51 hands
- Low power to detect moderate effects
- Only extreme effects (5-5, 6-0) survive correction

### Files Generated

- `results/tables/17a_enrichment.csv` - Full enrichment results
- `results/figures/17a_volcano_plot.png` - Volcano plot
- `results/figures/17a_enrichment_bars.png` - Bar plot

---

## 17b: High-Risk vs Low-Risk Enrichment

### Key Question
Which dominoes are over/under-represented in high oracle σ(V) hands?

### Method
- Split hands: Top 25% σ(V) (high-risk) vs Bottom 25% (low-risk)
- Fisher's exact test with BH FDR correction
- log₂(enrichment) = log₂(freq_high_risk / freq_low_risk)

### Key Findings

#### Significant Dominoes (FDR < 0.05)

3 dominoes survive multiple testing correction:

| Domino | High-Risk Freq | Low-Risk Freq | log₂(Enrichment) | p_adj |
|--------|---------------|---------------|------------------|-------|
| **6-5** | 34% | 8% | **+2.09** | 0.028 |
| **5-5** | 20% | 50% | **-1.32** | 0.028 |
| **2-0** | 14% | 44% | **-1.65** | 0.028 |

#### Interpretation (Oracle Variance)

**6-5 (six-five)**:
- 4× more common in high oracle σ(V) hands
- Mixed domino with no special power
- **Hypothesis**: Leads to opponent-dependent outcomes under oracle play

**5-5 (double-five)**:
- 2.5× more common in low oracle σ(V) hands
- High trump double
- **Hypothesis**: Wins tricks reliably regardless of opponent hands, reducing oracle variance

**2-0 (deuce-blank)**:
- 3× more common in low oracle σ(V) hands
- This is a weak domino, yet associated with low variance
- **Hypothesis**: Its weakness may be consistent across opponent configurations

### Comparison with Oracle E[V] Enrichment

The oracle E[V] vs oracle σ(V) enrichment patterns are consistent with the inverse relationship found in Section 12a:
- Dominoes enriched in high oracle E[V] hands tend to be depleted in high oracle σ(V) hands
- 5-5 is enriched in oracle winners AND depleted in high oracle variance hands

**Note**: This describes patterns in oracle data. Whether the same pattern holds for human gameplay outcomes is untested.

### Files Generated

- `results/tables/17b_risk_enrichment.csv` - Risk enrichment results
- `results/tables/17b_ev_risk_comparison.csv` - E[V] vs risk comparison
- `results/figures/17b_risk_volcano.png` - Volcano plot
- `results/figures/17b_ev_vs_risk.png` - E[V] vs risk scatter

---

## Further Investigation

### Validation Needed

1. **Larger sample sizes**: With only ~50 hands per group, power is limited. Only extreme effects (5-5, 6-0) survive FDR correction. A larger dataset could detect more moderate enrichments.

2. **Human gameplay validation**: Do the oracle-derived enrichments predict human game outcomes? This requires human gameplay data.

3. **Mechanism testing**: The hypotheses about *why* certain dominoes are enriched (e.g., "wins tricks reliably") are plausible but untested. Simulation or detailed game tree analysis could test these mechanisms.

### Methodological Questions

1. **Threshold sensitivity**: The 25% cutoffs for "winners" vs "losers" and "high-risk" vs "low-risk" are arbitrary. Would different thresholds reveal different patterns?

2. **Confounding**: Domino presence may correlate with other features (e.g., n_doubles). Multivariate analysis could disentangle individual domino effects.

3. **Multiple testing power**: With 28 dominoes and strict FDR correction, only the largest effects are detected. Is there a principled way to increase power while controlling FDR?

### Open Questions

1. **Why is 2-0 associated with low variance?**: The deuce-blank is weak but consistent. What game mechanism makes its weakness predictable?

2. **Interaction effects**: Are there domino *pairs* enriched in winners/losers beyond additive effects? This would complement Section 16c's synergy analysis.

3. **Declaration-specific enrichment**: Do the enrichment patterns differ by declaration (trump suit)? 5-5's value likely depends on whether fives are trump.

---

## Remaining Tasks

- 17c: Volcano plot variations
