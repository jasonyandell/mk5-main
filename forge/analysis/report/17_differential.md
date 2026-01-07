# 17: Differential Analysis

Comparing winners vs losers to identify distinguishing features.

## 17a: Winner vs Loser Enrichment

### Key Question
Which dominoes are over/under-represented in top 25% E[V] hands?

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

#### Interpretation

**5-5 (double-five)**:
- 2.8× more common in winners than losers
- High trump, wins tricks, 10 count points
- Strongest positive signal

**6-0 (six-blank)**:
- 3× more common in losers than winners
- Weak domino: high suit rank but no trick-winning power
- Strongest negative signal

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
Which dominoes are over/under-represented in high σ(V) hands?

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

#### Interpretation

**6-5 (six-five)**:
- 4× more common in high-risk hands
- Mixed domino with no special power
- Leads to unpredictable outcomes

**5-5 (double-five)**:
- 2.5× more common in low-risk hands
- High trump double = guaranteed trick winner
- Leads to predictable outcomes

**2-0 (deuce-blank)**:
- 3× more common in low-risk hands
- Interesting: this is a weak domino
- Perhaps its weakness is predictable?

### Comparison with E[V] Enrichment

The E[V] vs Risk enrichment correlation confirms the inverse relationship:
- Dominoes good for E[V] tend to be bad for risk (lower σ[V])
- 5-5 is enriched in winners AND depleted in high-risk

### Files Generated

- `results/tables/17b_risk_enrichment.csv` - Risk enrichment results
- `results/tables/17b_ev_risk_comparison.csv` - E[V] vs risk comparison
- `results/figures/17b_risk_volcano.png` - Volcano plot
- `results/figures/17b_ev_vs_risk.png` - E[V] vs risk scatter

---

## Remaining Tasks

- 17c: Volcano plot variations
