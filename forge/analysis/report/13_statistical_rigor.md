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

## 13b: Bootstrap CIs for Risk Formula

### Key Question
Are the risk formula coefficients (predicting σ(V)) statistically significant?

### Method
Same bootstrap approach as 13a: 1000 iterations, percentile CIs.

### Key Findings

| Feature | Coefficient | 95% CI | Significant? |
|---------|-------------|--------|--------------|
| total_pips | **+0.30** | [+0.01, +0.57] | **Marginal** |
| n_doubles | -1.40 | [-3.32, +0.77] | No |
| n_5_high | -1.09 | [-2.84, +0.63] | No |
| trump_count | -0.61 | [-1.56, +0.42] | No |
| has_trump_double | -0.55 | [-4.51, +3.08] | No |
| n_voids | +0.53 | [-3.49, +4.38] | No |

### Model Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| R² | 0.081 | [0.058, 0.203] |

### Critical Insight: Risk is Unpredictable

Only **total_pips** barely reaches significance (CI just excludes zero at [+0.01, +0.57]).

All other features have CIs that include zero. Combined with R² of only 6-20%, this confirms:

**Risk (outcome variance) is fundamentally unpredictable from hand features alone.**

The uncertainty in Texas 42 comes from opponent hand distribution, not from your own hand quality.

### Implications

1. **Don't try to assess hand "riskiness"**: No reliable signal exists
2. **Focus on E[V] predictors**: n_doubles and trump_count are meaningful; risk is not
3. **Embrace uncertainty**: Half of game outcomes are determined by luck (opponent hands)

### Files Generated

- `results/tables/13b_bootstrap_risk_coefficients.csv` - Full coefficient table with CIs
- `results/figures/13b_bootstrap_risk_ci.png` - Forest plot visualization

---

## 13c: Effect Sizes for Key Comparisons

### Key Question
Are our findings practically meaningful, or just statistically significant?

### Method
Computed standardized effect sizes:
- **r**: Pearson correlation (|r| < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, >0.5 large)
- **Cohen's d**: Standardized mean difference (|d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large)
- **R²**: Proportion of variance explained (<0.01 negligible, 0.01-0.09 small, 0.09-0.25 medium, >0.25 large)

### Key Findings

#### Correlation Effect Sizes

| Comparison | r | r² | Magnitude |
|------------|---|----|-----------|
| E[V] vs σ(V) | -0.38 | 0.15 | **Medium** |
| E[V] vs V_spread | -0.40 | 0.16 | **Medium** |
| n_doubles vs E[V] | +0.40 | 0.16 | **Medium** |
| trump_count vs E[V] | +0.23 | 0.05 | Small |
| count_points vs E[V] | +0.20 | 0.04 | Small |
| n_6_high vs E[V] | -0.16 | 0.03 | Small |
| total_pips vs E[V] | +0.04 | 0.00 | Negligible |

#### Group Comparison Effect Sizes (Cohen's d)

| Comparison | d | Magnitude |
|------------|---|-----------|
| ≥2 doubles vs <2 on E[V] | +0.76 | **Medium** |
| High risk vs Low risk on E[V] | -0.79 | **Medium** |
| ≥2 trumps vs <2 on E[V] | +0.48 | Small |
| ≥15 count pts vs <15 on E[V] | +0.27 | Small |

#### Regression R² (Variance Explained)

| Model | R² | % | Magnitude |
|-------|----|----|-----------|
| Hand features → E[V] | 0.26 | 26% | **Large** |
| Hand features → σ(V) | 0.08 | 8% | Small |

### Practical Significance Summary

**Medium/Large Effects (Practically Meaningful)**:
1. **n_doubles → E[V]**: r = +0.40, d = +0.76 — Real impact on outcomes
2. **E[V] ↔ σ(V)**: r = -0.38 — Genuine inverse risk-return relationship
3. **Hand features → E[V]**: R² = 0.26 — Useful prediction

**Small Effects (Modest Impact)**:
- trump_count, count_points have small correlations with E[V]
- Risk prediction (R² = 0.08) is weak

**Negligible Effects**:
- total_pips → E[V]: r = +0.04 — Raw hand "strength" doesn't matter

### Files Generated

- `results/tables/13c_effect_sizes.csv` - Full effect size summary
- `results/figures/13c_effect_sizes.png` - Visualization

---

## 13d: Fisher z-Transform Correlation CIs

### Key Question
Which correlations are statistically significant when properly accounting for sampling uncertainty?

### Method
- scipy.stats.pearsonr with confidence_interval() method
- Fisher z-transformation: z = arctanh(r), SE = 1/sqrt(n-3)
- 95% confidence intervals via inverse transform

### Key Findings

#### Significant Correlations (10 of 16)

| Comparison | r | 95% CI | Magnitude |
|------------|---|--------|-----------|
| E[V] vs V_spread | -0.40 | [-0.51, -0.27] | **Medium** |
| n_doubles vs E[V] | +0.40 | [+0.27, +0.51] | **Medium** |
| E[V] vs σ(V) | -0.38 | [-0.49, -0.26] | **Medium** |
| has_trump_double vs E[V] | +0.24 | [+0.11, +0.37] | Small |
| trump_count vs E[V] | +0.23 | [+0.09, +0.36] | Small |
| n_voids vs E[V] | +0.20 | [+0.06, +0.33] | Small |
| count_points vs E[V] | +0.20 | [+0.06, +0.33] | Small |
| n_6_high vs σ(V) | +0.19 | [+0.05, +0.32] | Small |
| n_6_high vs E[V] | -0.16 | [-0.29, -0.02] | Small |
| total_pips vs σ(V) | +0.15 | [+0.01, +0.28] | Small |

#### Non-Significant Correlations (6 of 16)

| Comparison | r | 95% CI | Note |
|------------|---|--------|------|
| n_doubles vs σ(V) | -0.14 | [-0.27, +0.00] | Marginal |
| trump_count vs σ(V) | -0.09 | [-0.23, +0.05] | Negligible |
| max_suit_length vs E[V] | -0.08 | [-0.22, +0.06] | Negligible |
| n_5_high vs E[V] | +0.08 | [-0.06, +0.21] | Negligible |
| total_pips vs E[V] | +0.04 | [-0.10, +0.17] | Negligible |
| n_singletons vs E[V] | +0.00 | [-0.14, +0.14] | Negligible |

### Critical Insight: Bivariate vs Multivariate

The Fisher z-transform CIs reveal an important distinction:

**Bivariately significant but multivariately not**:
- has_trump_double vs E[V]: r = +0.24 (significant)
- n_voids vs E[V]: r = +0.20 (significant)
- count_points vs E[V]: r = +0.20 (significant)

Yet in the multivariate regression (13a), these features have CIs that include zero. This means their bivariate correlations are largely explained by their association with n_doubles and trump_count.

### Implications

1. **Bivariate screening is encouraging**: Many features correlate with E[V]
2. **Multivariate tells the real story**: Only n_doubles and trump_count survive
3. **Risk remains unpredictable**: Even with CIs, σ(V) predictors are weak

### Files Generated

- `results/tables/13d_correlation_cis.csv` - Full correlation table with Fisher CIs
- `results/figures/13d_correlation_cis.png` - Forest plot visualization

---

## 13e: Power Analysis

### Key Question
Is n=200 sufficient to detect our observed effects with adequate statistical power?

### Method
- Power functions for correlation tests (t-distribution with non-centrality parameter)
- statsmodels.stats.power for Cohen's d (TTestIndPower)
- F-test power for regression R²
- Target: 80% power at α=0.05

### Key Findings

#### Power for Current Sample Size (n=200)

| Analysis | Effect Size | Type | Power | n for 80% | Sufficient? |
|----------|-------------|------|-------|-----------|-------------|
| r(E[V], σ[V]) | -0.38 | r | 1.000 | 51 | ✓ |
| r(n_doubles, E[V]) | +0.40 | r | 1.000 | 46 | ✓ |
| r(trump_count, E[V]) | +0.23 | r | 0.911 | 145 | ✓ |
| d(≥2 doubles vs <2) | 0.76 | d | 1.000 | 29×2=58 | ✓ |
| d(high vs low risk) | 0.79 | d | 1.000 | 27×2=54 | ✓ |
| d(≥2 trumps vs <2) | 0.48 | d | 0.922 | 70×2=140 | ✓ |
| R²(hand→E[V]) | 0.26 | R² | 1.000 | 57 | ✓ |
| R²(hand→σ[V]) | 0.08 | R² | 0.810 | 197 | ✓ |

#### Sample Size Requirements for 80% Power

**Correlations:**
| Target r | n needed |
|----------|----------|
| 0.1 (small) | 782 |
| 0.2 (small-medium) | 193 |
| 0.3 (medium) | 84 |
| 0.4 (medium) | 46 |
| 0.5 (large) | 29 |

**Group Comparisons (Cohen's d):**
| Target d | n per group |
|----------|-------------|
| 0.2 (small) | 394 |
| 0.5 (medium) | 64 |
| 0.8 (large) | 26 |

### Critical Insight: n=200 is Sufficient

**All key findings are well-powered:**
1. **r(E[V], σ[V]) = -0.38**: Power ≈ 1.00 — would only need n=51
2. **r(n_doubles, E[V]) = +0.40**: Power ≈ 1.00 — would only need n=46
3. **d(≥2 doubles vs <2) = 0.76**: Power ≈ 1.00 — would only need n=58 total

**Borderline but adequate:**
- R²(hand→σ[V]) = 0.08: Power = 0.81, would need n=197 for 80%
- We have n=200, so this is just at the threshold

### Implications

1. **No immediate scale-up needed**: n=200 provides >80% power for all key findings
2. **Main effects are robust**: Core relationships (E[V]-σ[V], n_doubles) have power ≈ 1.00
3. **If detecting smaller effects**: To find r=0.1 effects, would need n≈782
4. **Risk model is at limit**: The weak R²=0.08 for σ(V) just barely achieves 80% power

### Files Generated

- `results/tables/13e_power_analysis.csv` - Summary table
- `results/figures/13e_power_curves.png` - Power curves for correlations, d, and R²

---

## 13f: Multiple Comparison Correction (BH FDR)

### Key Question
Do our significant findings survive correction for multiple testing?

### Method
- Benjamini-Hochberg (BH) false discovery rate correction
- Controls FDR (expected proportion of false discoveries)
- Also compared with Bonferroni, Holm, Sidak for reference
- statsmodels.stats.multitest.multipletests

### Key Findings

#### Tests by Correction Method

| Method | Significant | Type |
|--------|-------------|------|
| Uncorrected | 10 | None |
| **BH FDR** | **9** | **FDR** |
| BY FDR | 8 | FDR |
| Holm | 6 | FWER |
| Bonferroni | 5 | FWER |
| Sidak | 5 | FWER |

#### Tests Surviving BH FDR Correction

| Comparison | r | p_raw | p_adj_BH |
|------------|---|-------|----------|
| E[V] vs V_spread | -0.40 | 5.4×10⁻⁹ | <0.0001 |
| n_doubles vs E[V] | +0.40 | 6.9×10⁻⁹ | <0.0001 |
| E[V] vs σ(V) | -0.38 | 2.6×10⁻⁸ | <0.0001 |
| has_trump_double vs E[V] | +0.24 | 5.6×10⁻⁴ | 0.0022 |
| trump_count vs E[V] | +0.23 | 1.1×10⁻³ | 0.0036 |
| n_voids vs E[V] | +0.20 | 4.5×10⁻³ | 0.0119 |
| count_points vs E[V] | +0.20 | 5.2×10⁻³ | 0.0119 |
| n_6_high vs σ(V) | +0.19 | 6.8×10⁻³ | 0.0135 |
| n_6_high vs E[V] | -0.16 | 2.3×10⁻² | 0.0412 |

#### Test Lost to BH FDR

| Comparison | r | p_raw | p_adj_BH |
|------------|---|-------|----------|
| total_pips vs σ(V) | +0.15 | 0.035 | 0.056 |

### Critical Insight: Core Findings Are Robust

**9 of 10 significant correlations survive BH FDR correction** at α = 0.05.

The only test lost (total_pips vs σ(V)) was marginal anyway:
- Effect size: small (r = 0.15)
- p_adj = 0.056 (just above threshold)

**All key findings survive**:
1. E[V] vs σ(V) (r = -0.38) — headline inverse risk-return
2. n_doubles vs E[V] (r = +0.40) — strongest predictor
3. trump_count vs E[V] (r = +0.23) — second key predictor

### FWER vs FDR

- **FWER** (Bonferroni, Holm): Controls family-wise error rate (any false positive)
- **FDR** (BH, BY): Controls expected proportion of false discoveries

For exploratory analysis with 16 tests, **FDR is appropriate**. We accept that ~1 in 20 discoveries may be false positive.

### Files Generated

- `results/tables/13f_multiple_comparison.csv` - Full table with adjusted p-values
- `results/figures/13f_multiple_comparison.png` - Visualization of BH procedure

---

## Summary

Statistical rigor analyses confirm:

1. **Only two predictors survive multivariate analysis**: n_doubles and trump_count
2. **Risk is unpredictable**: σ(V) model has weak R² (0.08) and borderline power
3. **Effect sizes are medium**: Practically meaningful, not just statistically significant
4. **n=200 is sufficient**: All key findings have adequate power (>80%)
5. **Multiple testing robust**: 9 of 10 correlations survive BH FDR correction
