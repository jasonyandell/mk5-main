# 14: Explainability (SHAP)

SHAP (SHapley Additive exPlanations) analysis for understanding per-hand feature contributions.

> **Epistemic Status**: This report uses SHAP to explain model predictions of oracle E[V] and oracle σ(V). All findings describe what drives oracle outcome predictions—not human gameplay outcomes. The "implications for bidding" are hypotheses extrapolated from oracle model behavior; none have been validated against human gameplay.

## 14a: SHAP on E[V] Model

### Key Question
What drives individual oracle E[V] predictions? Can we explain why specific hands have high or low oracle E[V]?

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

### Hypothetical Implications for Bidding (Oracle-Derived)

The following are extrapolations from oracle model behavior. **None have been validated against human gameplay.**

1. **n_doubles matters most for oracle E[V]**: Each double can swing oracle E[V] by several points
2. **trump_count is second for oracle**: Strong trump holding improves oracle outcomes
3. **Other features matter at margins**: n_singletons, count_points provide additional oracle signal
4. **Per-hand analysis possible**: SHAP waterfall explains any specific hand's oracle prediction

**Note**: Whether these oracle-derived insights translate to human bidding decisions is untested.

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

### Implications (Oracle Risk)

1. **Oracle risk is unpredictable from hand features**: No hand features reliably indicate oracle σ(V)
2. **Opponent hands dominate oracle variance**: σ(V) comes from opponent distributions in marginalized data
3. **Focus on oracle E[V]**: n_doubles and trump_count predict oracle expected value; oracle risk is unknowable from your hand

**Note**: These findings describe oracle outcome variance. Whether human players should similarly "ignore risk" in bidding is untested.

### Files Generated

- `results/tables/14b_shap_sigma_importance.csv` - Feature importance
- `results/tables/14b_shap_sigma_values.csv` - Per-sample SHAP values
- `results/figures/14b_shap_beeswarm.png` - Global importance plot
- `results/figures/14b_shap_bar.png` - Bar chart
- `results/figures/14b_shap_scatter.png` - Feature relationships
- `results/figures/14b_shap_waterfall_highrisk.png` - High risk hand breakdown
- `results/figures/14b_shap_waterfall_lowrisk.png` - Low risk hand breakdown

---

## 14c: SHAP Interaction Values

### Key Question
Do any feature pairs have synergistic effects on E[V]?

### Method
- shap.TreeExplainer.shap_interaction_values()
- Returns (n_samples, n_features, n_features) matrix
- Diagonal = main effects, off-diagonal = interactions

### Key Findings

#### Top Feature Interactions

| Feature 1 | Feature 2 | Mean |Interaction| |
|-----------|-----------|---------------------|
| n_doubles | n_singletons | 0.73 |
| trump_count | total_pips | 0.54 |
| max_suit_length | n_singletons | 0.47 |
| trump_count | has_trump_double | 0.40 |
| **n_doubles** | **trump_count** | **0.37** |

#### Main Effects vs Interactions

| Feature | Main Effect | Total Interactions | Main/Total |
|---------|-------------|-------------------|------------|
| n_doubles | 4.92 | 2.27 | 68% |
| trump_count | 3.94 | 2.18 | 64% |
| count_points | 1.92 | 1.41 | 58% |
| n_singletons | 1.75 | 2.30 | 43% |

### Critical Insight: Main Effects Dominate

**Main effects account for 60-70% of SHAP for the key predictors**:
- n_doubles: 68% main effect, 32% interactions
- trump_count: 64% main effect, 36% interactions

**Feature effects are largely additive**:
- No strong synergies discovered
- n_doubles × trump_count interaction (0.37) is smaller than main effects (4.9, 3.9)
- Supports simple additive napkin formula

**Surprise finding**: n_doubles × n_singletons (0.73) is the top interaction, likely because hands with many doubles tend to have fewer singletons (structural correlation).

### Implications (Oracle Additivity)

1. **Napkin formula is justified for oracle**: Additive model captures most oracle signal
2. **No multiplicative terms needed for oracle**: Oracle E[V] = a×doubles + b×trumps works
3. **Interactions are second-order in oracle**: Main effects dominate oracle predictions

**Note**: This validates additive structure in oracle data. Whether human game outcomes are similarly additive in doubles/trumps is untested.

### Files Generated

- `results/tables/14c_shap_interactions.csv` - Top feature pair interactions
- `results/tables/14c_main_vs_interactions.csv` - Main vs interaction breakdown
- `results/figures/14c_shap_interaction_heatmap.png` - Full interaction matrix
- `results/figures/14c_doubles_trump_interaction.png` - Key interaction scatter

---

## Further Investigation

### Validation Needed

1. **Human SHAP analysis**: Apply SHAP to models trained on human game outcomes. Do n_doubles and trump_count have similar importance for human wins?

2. **Cross-validation of bidding implications**: Test whether bidders who count doubles (as SHAP suggests matters most) actually win more human games.

3. **Risk model on human data**: The oracle σ(V) is unpredictable. Is human outcome variance similarly unpredictable from hand features?

### Methodological Questions

1. **Model overfitting**: CV R² = 0.20 vs Train R² = 0.81 suggests overfitting. Does SHAP still provide valid feature importance when the model overfits?

2. **SHAP on small samples**: With n=200 hands, how stable are the SHAP values? Bootstrap analysis of SHAP importance could quantify this.

3. **Alternative models**: Would SHAP on a different model (Random Forest, XGBoost) give consistent feature importance rankings?

### Open Questions

1. **Why are main effects so dominant?**: The 60-70% main effect proportion suggests near-additive structure. What game mechanism causes this?

2. **Surprising interaction**: n_doubles × n_singletons (0.73) is the top interaction. Is this a spurious structural correlation or a real game effect?

3. **Transfer to human play**: If SHAP explains oracle predictions, can these explanations help human players understand their hands better?
