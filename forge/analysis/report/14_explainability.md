# 14: Explainability (SHAP)

SHAP (SHapley Additive exPlanations) analysis for understanding per-hand feature contributions.

## 14a: SHAP on E[V] Model

### Key Question
What drives individual hand predictions? Can we explain why specific hands have high or low E[V]?

### Method
- Model: GradientBoostingRegressor (n_estimators=100, max_depth=3)
- Explainer: shap.TreeExplainer (fast, exact for tree-based models)
- Output: Per-sample SHAP values + global importance

### Model Performance

| Metric | Value |
|--------|-------|
| CV R² | 0.20 ± 0.30 |
| Train R² | 0.81 |

Note: High train R² vs low CV R² indicates overfitting, but SHAP analysis is still valid for understanding model behavior.

### Global Feature Importance (Mean |SHAP|)

| Feature | Mean |SHAP| | Mean SHAP | Std SHAP |
|---------|-------------|-----------|----------|
| **n_doubles** | **4.84** | +0.25 | 5.82 |
| **trump_count** | **4.39** | -0.06 | 6.03 |
| n_singletons | 2.17 | -0.12 | 2.54 |
| count_points | 2.17 | +0.04 | 2.44 |
| total_pips | 2.00 | -0.12 | 2.95 |
| n_6_high | 1.44 | +0.06 | 2.16 |
| has_trump_double | 1.09 | -0.07 | 1.45 |
| max_suit_length | 0.81 | -0.09 | 1.53 |
| n_voids | 0.79 | -0.07 | 0.99 |
| n_5_high | 0.67 | -0.04 | 0.98 |

### Key Findings

#### 1. n_doubles and trump_count Dominate

SHAP confirms our regression findings:
- **n_doubles** (mean |SHAP| = 4.84) is the most important feature
- **trump_count** (mean |SHAP| = 4.39) is second most important
- Together they account for ~45% of total feature importance

#### 2. Other Features Have Non-Zero but Smaller Impact

Unlike linear regression where many coefficients were non-significant, SHAP shows:
- n_singletons, count_points, total_pips each contribute ~2 points of |SHAP|
- GradientBoosting captures nonlinear relationships that linear models miss

#### 3. Per-Hand Explanations

Waterfall plots reveal why specific hands have extreme E[V]:

**Best Hand (E[V] = 42.0)**:
- High n_doubles pushes prediction strongly positive
- Favorable trump_count adds additional positive contribution

**Worst Hand (E[V] = -29.3)**:
- Low n_doubles provides no positive contribution
- Unfavorable combination of other features

#### 4. SHAP Additivity Verified

SHAP values sum exactly to (prediction - base_value), confirming correct implementation:
- Base value = 14.03 (expected E[V])
- Sum of SHAP values = prediction - 14.03
- Max error: 0.000000

### Implications for Bidding

1. **n_doubles matters most**: Each double can swing E[V] by several points
2. **trump_count is second**: Strong trump holding improves outcomes
3. **Other features matter at margins**: n_singletons, count_points provide additional signal
4. **Per-hand analysis possible**: SHAP waterfall explains any specific hand's prediction

### Files Generated

- `results/tables/14a_shap_importance.csv` - Feature importance summary
- `results/tables/14a_shap_values.csv` - Per-sample SHAP values
- `results/figures/14a_shap_beeswarm.png` - Global importance plot
- `results/figures/14a_shap_bar.png` - Bar chart of mean |SHAP|
- `results/figures/14a_shap_scatter.png` - Feature relationship plots
- `results/figures/14a_shap_waterfall_best.png` - Best hand breakdown
- `results/figures/14a_shap_waterfall_worst.png` - Worst hand breakdown
- `results/figures/14a_shap_interaction.png` - n_doubles × trump_count interaction

---

## 14b: SHAP on σ(V) (Risk) Model

### Key Question
Can we identify which hand features contribute to outcome variance (risk)?

### Method
Same as 14a: GradientBoostingRegressor + TreeExplainer, but target = σ(V).

### Model Performance

| Metric | Value |
|--------|-------|
| CV R² | **-0.34 ± 0.71** |
| Train R² | 0.67 |

**Critical Finding**: Negative CV R² means the model is worse than predicting the mean. This confirms σ(V) is unpredictable from hand features.

### Global Feature Importance (Mean |SHAP|)

| Feature | Mean |SHAP| | vs E[V] Ratio |
|---------|-------------|---------------|
| total_pips | 2.08 | 0.96 (similar) |
| trump_count | 1.26 | 3.49 (E[V] much higher) |
| count_points | 1.01 | 2.15 |
| n_5_high | 0.99 | 0.68 (σ higher) |
| n_singletons | 0.85 | 2.55 |
| n_voids | 0.77 | 1.03 (similar) |
| n_doubles | 0.70 | **6.92** (E[V] dominates) |

### Key Findings

#### 1. Risk Model Fits Noise

The negative CV R² (-0.34) proves the model is fitting random patterns:
- High train R² (0.67) is pure overfitting
- No feature reliably predicts σ(V)

#### 2. Different Feature Profiles

Comparing E[V] vs σ(V) models:
- **n_doubles**: E[V] importance 6.9× higher than σ(V) importance
- **trump_count**: E[V] importance 3.5× higher
- **total_pips**: Similar importance in both models (uninformative)

#### 3. Confirms 13b Bootstrap Analysis

The SHAP analysis corroborates bootstrap CIs:
- Linear regression on σ(V): R² = 0.08 (low)
- GradientBoosting on σ(V): CV R² = -0.34 (worse than baseline)
- Nonlinear models don't help predict risk

### Implications

1. **Don't try to predict risk**: No hand features reliably indicate outcome variance
2. **Opponent hands dominate**: σ(V) comes from unknown opponent distributions
3. **Focus on E[V]**: n_doubles and trump_count predict expected value; risk is unknowable

### Files Generated

- `results/tables/14b_shap_sigma_importance.csv` - Feature importance
- `results/tables/14b_shap_sigma_values.csv` - Per-sample SHAP values
- `results/figures/14b_shap_beeswarm.png` - Global importance plot
- `results/figures/14b_shap_bar.png` - Bar chart
- `results/figures/14b_shap_scatter.png` - Feature relationships
- `results/figures/14b_shap_waterfall_highrisk.png` - High risk hand breakdown
- `results/figures/14b_shap_waterfall_lowrisk.png` - Low risk hand breakdown

---

## Remaining Tasks

- SHAP interaction values analysis (optional)
