# 13: Statistical Rigor

Adding confidence intervals, effect sizes, and rigorous statistical testing to key findings.

## 13a: Bootstrap CIs for Regression Coefficients

### Key Question
How certain are the 11f regression coefficients? Are they statistically significant?

### Method
- Bootstrap resampling: 1000 iterations
- Percentile confidence intervals (95%)
- Linear regression: hand features → E[V]

### Key Findings

| Feature | Coefficient | 95% CI | Width | Significant? |
|---------|-------------|--------|-------|--------------|
| n_doubles | **+5.7** | [+2.3, +9.2] | 6.9 | **Yes** |
| trump_count | **+3.2** | [+1.3, +4.7] | 3.5 | **Yes** |
| has_trump_double | +2.8 | [-2.6, +8.4] | 11.0 | No |
| n_voids | +2.8 | [-3.5, +8.9] | 12.4 | No |
| n_6_high | -1.6 | [-5.0, +1.8] | 6.8 | No |
| max_suit_length | -0.7 | [-6.2, +4.4] | 10.6 | No |
| n_5_high | -0.5 | [-3.3, +2.1] | 5.4 | No |
| count_points | +0.2 | [-0.1, +0.5] | 0.6 | No |
| total_pips | +0.1 | [-0.4, +0.6] | 1.0 | No |
| n_singletons | -0.3 | [-3.7, +3.5] | 7.3 | No |

### Model Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| R² | 0.259 | [0.197, 0.400] |
| Intercept | -2.9 | [-22.3, +17.4] |

### Critical Insight: Only Two Significant Predictors

Of 10 hand features, **only n_doubles and trump_count** have confidence intervals that exclude zero.

This is a major refinement from 11f/12b:
- **Original claim**: "n_doubles (+5.7), trump_count (+3.2), has_trump_double (+2.8), n_voids (+2.8) all predict E[V]"
- **Refined claim**: "Only n_doubles and trump_count are statistically significant; other features have too much uncertainty"

### The Robust Napkin Formula

```
E[V] ≈ -3 + 5.7×(doubles) + 3.2×(trump_count)
```

Everything else is noise. The simpler 2-feature model may generalize better than the 10-feature model.

### Implications for Bidding

1. **Count your doubles**: The most reliable signal (+5.7 points per double)
2. **Count your trumps**: Second most reliable (+3.2 points per trump)
3. **Everything else is uncertain**: has_trump_double, n_voids, and n_6_high all have CIs that include zero

### Files Generated

- `results/tables/13a_bootstrap_coefficients.csv` - Full coefficient table with CIs
- `results/figures/13a_bootstrap_regression_ci.png` - Forest plot visualization

---

## Remaining Tasks

- Additional rigor tasks TBD based on epic t42-6xhh
