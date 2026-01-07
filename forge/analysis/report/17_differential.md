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

## Remaining Tasks

- 17b: Distinguishing dominoes (top 10 vs bottom 10)
- 17c: Volcano plot variations
