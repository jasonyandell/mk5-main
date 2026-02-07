# 12: Validate & Scale

Scaling existing analyses to n=201 seeds and recomputing with consistent methodology.

> **Epistemic Status**: This report validates oracle (marginalized data) findings from Section 11 at larger scale. All correlations and feature relationships describe oracle E[V] and oracle σ(V). The "inverse risk-return" finding is a property of oracle outcomes. Whether these relationships hold for human gameplay outcomes is untested.

## 12a: E[V] vs σ(V) Correlation at Scale

### Key Question
Is the negative correlation between expected value and risk real, or a small-sample artifact?

### Method
Using scipy.stats.pearsonr with Fisher transformation confidence intervals on the full n=200 seed marginalized dataset (from 11s).

### Key Findings

| Metric | Correlation | 95% CI | p-value | Effect Size |
|--------|-------------|--------|---------|-------------|
| r(E[V], σ[V]) | **-0.381** | [-0.494, -0.256] | 2.6×10⁻⁸ | medium |
| r(E[V], V_spread) | **-0.398** | [-0.509, -0.274] | 5.4×10⁻⁹ | medium |

### Statistical Summary

- **n = 200** base seeds
- **R² = 0.145** for σ(V), meaning ~14.5% of variance explained
- **R² = 0.158** for V_spread
- **Effect size**: medium by Cohen's conventions (0.3 < |r| < 0.5)

### Interpretation (Oracle Data)

The negative correlation in oracle data is **confirmed** at scale with high statistical confidence:

1. **Inverse oracle risk-return relationship**: Higher oracle E[V] hands also have lower oracle σ(V) - the opposite of typical financial markets
2. **Not spurious in oracle**: p < 10⁻⁸ rules out sampling artifact
3. **Moderate effect**: ~15% of variance explained - meaningful but not dominant

**Note**: This is an oracle finding. Whether human players experience a similar inverse risk-return relationship in actual gameplay is untested.

### Original Hypothesis Correction

The task hypothesized r ≈ -0.55. Actual finding: **r ≈ -0.38 to -0.40**. The effect is real but smaller than initially estimated.

### Files Generated

- `results/tables/12a_ev_sigma_correlation.csv` - Summary statistics
- `results/figures/12a_ev_sigma_correlation.png` - Scatter plot with regression

---

## 12b: Unified Feature Extraction

### Key Question
Can we consolidate the duplicated feature extraction code across 7+ run_11*.py scripts?

### Method
Created `forge/analysis/utils/hand_features.py` with:
- `extract_hand_features(hand, trump_suit)` - single source of truth
- `HAND_FEATURE_NAMES` - consistent column ordering
- `REGRESSION_FEATURES` - subset for ML models

### Output

**Master feature file**: `results/tables/12b_unified_features.csv`
- 200 base seeds × 20 columns
- V statistics: E[V], σ(V), V_spread, V_min, V_max
- 12 hand features

### Feature Summary

| Feature | Mean | Range | r with E[V] |
|---------|------|-------|-------------|
| n_doubles | 1.73 | [0, 4] | **+0.395** |
| trump_count | 1.32 | [0, 5] | +0.229 |
| has_trump_double | 0.17 | [0, 1] | +0.242 |
| count_points | 9.20 | [0, 25] | +0.197 |
| n_voids | 0.67 | [0, 3] | +0.200 |
| n_6_high | 1.74 | [0, 4] | **-0.161** |

### Key Findings (Oracle Predictors)

1. **n_doubles is king for oracle E[V]**: Strongest predictor of oracle E[V] (r = +0.40, p < 10⁻⁸)
2. **Trump features matter for oracle**: trump_count and has_trump_double both positive oracle predictors
3. **6-highs correlate with lower oracle E[V]**: Negative correlation (-0.16)
4. **Total pips irrelevant for oracle**: Near-zero correlation (+0.04) - raw hand strength doesn't predict oracle outcomes

**Note**: These are oracle-predictor relationships. Whether n_doubles and trump_count similarly predict human game outcomes is untested.

### Files Generated

- `utils/hand_features.py` - Unified feature extraction module
- `results/tables/12b_unified_features.csv` - Master feature dataset

---

## Further Investigation

### Validation Needed

1. **Human gameplay validation**: Do the oracle correlations (E[V]-σ[V], n_doubles-E[V]) hold in human game outcomes? This requires human gameplay data.

2. **Cross-declaration stability**: The unified features pool across declarations. Do the correlations hold within each trump suit separately?

3. **Sample size sensitivity**: n=200 provides adequate power for the observed effects. Would n=1000 reveal additional significant predictors?

### Methodological Questions

1. **Marginalized vs full oracle**: These findings use marginalized data (fixed P0 hand). Would full oracle data show similar patterns?

2. **Feature collinearity**: n_doubles and has_trump_double may be correlated. Is there multicollinearity affecting the regression?

3. **Heterogeneity by declaration**: The feature correlations may vary by trump suit. A stratified analysis could reveal this.

### Open Questions

1. **Why inverse risk-return?**: The negative E[V]-σ[V] correlation is unexpected. What game mechanism causes high-value hands to also be low-variance?

2. **Why are 6-highs negative?**: Intuitively, 6-highs seem strong. Why do they correlate negatively with oracle E[V]?

3. **Human decision relevance**: If n_doubles predicts oracle E[V], should human bidders count their doubles? This assumes oracle patterns transfer to human play.

---

## Remaining Tasks

- Additional validation tasks TBD based on epic t42-1wp2
