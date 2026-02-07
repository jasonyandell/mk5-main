# 13: Statistical Rigor

Adding confidence intervals, effect sizes, and rigorous statistical testing to key findings.

> **Epistemic Status**: This report applies statistical rigor to oracle (perfect-information minimax) findings from Sections 11-12. All regressions, confidence intervals, effect sizes, and cross-validation results describe oracle E[V] and oracle σ(V). The "napkin formula" and its validation characterize oracle predictions. Whether these statistical relationships hold for human gameplay outcomes is untested.

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

### The Robust Oracle Napkin Formula

```
Oracle E[V] ≈ -3 + 5.7×(doubles) + 3.2×(trump_count)
```

Everything else is noise in the oracle model. The simpler 2-feature model may generalize better than the 10-feature model for oracle prediction.

### Hypothetical Implications for Bidding (Oracle-Derived)

The following are hypotheses extrapolated from oracle regression analysis. **None have been validated against human gameplay.**

1. **Hypothesis**: Counting doubles may be the most reliable bidding signal (+5.7 oracle points per double)
2. **Hypothesis**: Counting trumps may be second most reliable (+3.2 oracle points per trump)
3. **Hypothesis**: Other features (has_trump_double, n_voids, n_6_high) may not add bidding value

**Note**: Human bidding involves strategic considerations (signaling, partner communication) not captured by oracle regression.

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

### Critical Insight: Oracle Risk is Unpredictable

Only **total_pips** barely reaches significance (CI just excludes zero at [+0.01, +0.57]).

All other features have CIs that include zero. Combined with R² of only 6-20%, this confirms:

**Oracle risk (σ(V)) is fundamentally unpredictable from hand features alone in marginalized data.**

The oracle variance comes from opponent hand distribution (which is marginalized), not from your own hand quality.

### Hypothetical Implications (Oracle-Derived)

The following are hypotheses extrapolated from oracle risk analysis. **None have been validated against human gameplay.**

1. **Hypothesis**: Human players may not be able to assess hand "riskiness" reliably
2. **Hypothesis**: Bidders might focus on E[V] predictors (doubles, trumps) rather than risk assessment
3. **Hypothesis**: A significant portion of game outcomes may be determined by opponent hands

**Note**: In human play with hidden information, players may use inference and signaling to reduce uncertainty. This oracle analysis doesn't capture such dynamics.

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

### Practical Significance Summary (Oracle Data)

**Medium/Large Effects in Oracle Data (Practically Meaningful)**:
1. **n_doubles → oracle E[V]**: r = +0.40, d = +0.76 — Real impact on oracle outcomes
2. **Oracle E[V] ↔ oracle σ(V)**: r = -0.38 — Genuine inverse oracle risk-return relationship
3. **Hand features → oracle E[V]**: R² = 0.26 — Useful oracle prediction

**Small Effects in Oracle Data (Modest Impact)**:
- trump_count, count_points have small correlations with oracle E[V]
- Oracle risk prediction (R² = 0.08) is weak

**Negligible Effects in Oracle Data**:
- total_pips → oracle E[V]: r = +0.04 — Raw hand "strength" doesn't predict oracle outcomes

**Note**: These effect sizes characterize oracle outcomes. Whether similar effect sizes hold for human game outcomes is untested.

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

### Implications (Oracle Analysis)

1. **Bivariate screening is encouraging in oracle data**: Many features correlate with oracle E[V]
2. **Multivariate tells the real story for oracle**: Only n_doubles and trump_count survive
3. **Oracle risk remains unpredictable**: Even with CIs, oracle σ(V) predictors are weak

**Note**: These correlations describe oracle outcomes. The distinction between bivariate and multivariate significance may or may not transfer to human game outcomes.

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

### Implications (Oracle Sample Size)

1. **No immediate scale-up needed for oracle analysis**: n=200 provides >80% power for all key oracle findings
2. **Main oracle effects are robust**: Core oracle relationships (E[V]-σ[V], n_doubles) have power ≈ 1.00
3. **If detecting smaller oracle effects**: To find r=0.1 effects in oracle data, would need n≈782
4. **Oracle risk model is at limit**: The weak R²=0.08 for oracle σ(V) just barely achieves 80% power

**Note**: This power analysis applies to oracle data. Human gameplay data would require separate power analysis with potentially different effect sizes.

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

**All key oracle findings survive**:
1. Oracle E[V] vs oracle σ(V) (r = -0.38) — headline inverse oracle risk-return
2. n_doubles vs oracle E[V] (r = +0.40) — strongest oracle predictor
3. trump_count vs oracle E[V] (r = +0.23) — second key oracle predictor

### FWER vs FDR

- **FWER** (Bonferroni, Holm): Controls family-wise error rate (any false positive)
- **FDR** (BH, BY): Controls expected proportion of false discoveries

For exploratory analysis with 16 tests, **FDR is appropriate**. We accept that ~1 in 20 discoveries may be false positive.

### Files Generated

- `results/tables/13f_multiple_comparison.csv` - Full table with adjusted p-values
- `results/figures/13f_multiple_comparison.png` - Visualization of BH procedure

---

## 13g: Cross-Validation for Regressions

### Key Question
How well do our regression models generalize to unseen data?

### Method
- 10-fold CV with 10 repeats (100 total fits)
- Compared napkin formula (2 features) vs full model (10 features)
- Also tested Ridge regularization
- sklearn.model_selection.RepeatedKFold

### Key Findings

#### E[V] Prediction

| Model | Train R² | CV R² | Overfit Ratio |
|-------|----------|-------|---------------|
| **Napkin (2 features)** | 0.233 | **0.152 ± 0.21** | **1.5x** |
| Full (10 features) | 0.259 | 0.109 ± 0.25 | 2.4x |
| Ridge Full | 0.259 | 0.111 ± 0.24 | 2.3x |

**Key insight**: The napkin formula **generalizes better** than the full model:
- Lower overfit ratio (1.5x vs 2.4x)
- Higher CV R² (0.152 vs 0.109)
- Simpler models win for this task

#### σ(V) Prediction

| Model | Train R² | CV R² | Status |
|-------|----------|-------|--------|
| Napkin | 0.030 | -0.065 ± 0.17 | **FAILS** |
| Full | 0.081 | -0.126 ± 0.20 | **FAILS** |
| Ridge Full | 0.081 | -0.123 ± 0.20 | **FAILS** |

**Key insight**: σ(V) prediction **completely fails cross-validation**:
- All models have negative CV R² (worse than predicting the mean)
- Any training R² is pure overfitting
- Confirms 13b, 14b: risk is fundamentally unpredictable

### Critical Insight: Oracle Napkin Formula Wins

The 2-feature napkin formula (n_doubles, trump_count) is the **best model for oracle prediction**:
1. Lowest overfitting
2. Best generalization to unseen oracle data
3. Simplest interpretation

**Recommended oracle formula**: Oracle E[V] ≈ -3 + 5.7×(doubles) + 3.2×(trumps)

### Implications (Oracle Prediction)

1. **Simpler is better for oracle**: Adding features hurts oracle generalization
2. **Don't predict oracle risk**: No model works for oracle σ(V)
3. **Use the napkin formula for oracle**: Validated by cross-validation on oracle data

**Note**: Whether this formula generalizes to human gameplay outcomes is untested. Human games may have different predictors or relationships.

### Files Generated

- `results/tables/13g_cross_validation.csv` - Summary statistics
- `results/figures/13g_cross_validation.png` - Train vs CV comparison
- `results/figures/13g_learning_curve.png` - Learning curve

---

## Summary (Oracle Statistical Rigor)

Statistical rigor analyses of oracle data confirm:

1. **Only two predictors survive multivariate analysis for oracle E[V]**: n_doubles and trump_count
2. **Oracle risk is unpredictable**: Oracle σ(V) model has weak R² (0.08) and fails cross-validation
3. **Effect sizes are medium in oracle data**: Practically meaningful, not just statistically significant
4. **n=200 is sufficient for oracle analysis**: All key oracle findings have adequate power (>80%)
5. **Multiple testing robust for oracle**: 9 of 10 oracle correlations survive BH FDR correction
6. **Oracle napkin formula validated**: CV R² = 0.15, lowest overfitting of all oracle models

**Scope limitation**: These statistical rigor findings describe oracle (perfect-information) outcomes. Whether the same statistical relationships hold for human gameplay—with hidden information, suboptimal play, and strategic considerations—is untested.

---

## Further Investigation

### Validation Needed

1. **Human gameplay regression**: Does the napkin formula (n_doubles + trump_count) predict human game outcomes? This requires human gameplay data with hand-level outcomes.

2. **Human risk prediction**: Is human outcome variance also unpredictable from hand features? Human inference and signaling might change this relationship.

3. **Effect size transfer**: Do the medium effect sizes (r ≈ 0.4 for n_doubles) hold for human data? Effect sizes might be larger (human games more correlated with hand features) or smaller (more variance from skill/luck).

### Methodological Questions

1. **Bootstrap assumptions**: Bootstrap CIs assume exchangeability. With 200 hands from random deals, this is likely satisfied, but domain-specific violations could exist.

2. **Cross-validation leakage**: Do hands share any structure (e.g., similar trump suits) that could create dependence across folds?

3. **Regression linearity**: The napkin formula assumes linear relationships. Are there threshold effects (e.g., "2+ doubles" vs "1 or fewer") that would justify different functional forms?

### Open Questions

1. **Why do only two features survive?**: The multivariate reduction from 10 to 2 significant features is striking. Is this a sample size issue, or is there fundamental redundancy in hand features?

2. **Oracle vs human predictor structure**: If human gameplay involves more strategic complexity, might different features (e.g., flexibility, signaling potential) become predictive?

3. **Generalization to bidding strategy**: The napkin formula describes oracle outcomes. Should bidders use it directly, or does optimal human bidding require different considerations?
