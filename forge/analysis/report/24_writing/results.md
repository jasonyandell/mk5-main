# Results

## 3.1 Oracle Evaluation and Marginalization

The oracle solved 600 game configurations (200 base seeds × 3 opponent configurations) under perfect play. Expected value E[V] ranged from -29.3 to +42.0 (mean = 13.9, SD = 16.6). Outcome variance σ(V) ranged from 0.0 to 37.7 points across opponent configurations.

## 3.2 Inverse Risk-Return Relationship

**Finding**: Strong hands have lower outcome variance (Figure 1).

We observed a significant negative correlation between expected value and risk:

| Measure | r | 95% CI | p-value |
|---------|---|--------|---------|
| E[V] vs σ(V) | **-0.381** | [-0.494, -0.256] | 2.6×10⁻⁸ |
| E[V] vs V_spread | **-0.398** | [-0.509, -0.274] | 5.4×10⁻⁹ |

This is the opposite of typical financial markets, where higher expected returns typically require accepting higher risk. In Texas 42, good hands are also safer hands.

**Effect size**: |r| = 0.38-0.40 represents a medium effect by Cohen's conventions.

## 3.3 Feature Importance for E[V] Prediction

### Bivariate Correlations

Ten features were correlated with E[V] (Table 1). After BH FDR correction, 9 correlations remained significant:

| Feature | r | 95% CI | p_adj |
|---------|---|--------|-------|
| n_doubles | **+0.395** | [+0.27, +0.51] | <0.0001 |
| has_trump_double | +0.242 | [+0.11, +0.37] | 0.0006 |
| trump_count | **+0.229** | [+0.09, +0.36] | 0.0036 |
| n_voids | +0.197 | [+0.06, +0.33] | 0.0073 |
| count_points | +0.197 | [+0.06, +0.33] | 0.0073 |

### Multivariate Regression

Multiple regression with bootstrap confidence intervals (B=1000) identified two significant predictors:

| Feature | β | 95% CI | Significant? |
|---------|---|--------|--------------|
| n_doubles | **+5.7** | [+2.3, +9.2] | **Yes** |
| trump_count | **+3.2** | [+1.3, +4.7] | **Yes** |
| has_trump_double | +2.8 | [-2.6, +8.4] | No |
| n_voids | +2.8 | [-3.5, +8.9] | No |
| n_6_high | -1.6 | [-5.0, +1.8] | No |

Model fit: **R² = 0.26** (95% CI: [0.20, 0.40]).

**Napkin formula**: E[V] ≈ 14 + 6×(n_doubles) + 3×(trump_count)

### Cross-Validation

10-fold cross-validation with 10 repeats confirmed generalization:

| Model | Train R² | CV R² | Overfit Ratio |
|-------|----------|-------|---------------|
| Napkin (2 features) | 0.23 | **0.15** | 1.5× |
| Full (10 features) | 0.26 | 0.11 | 2.4× |

The parsimonious napkin formula generalizes better than the full model.

### SHAP Analysis

GradientBoostingRegressor with SHAP values confirmed feature importance:

| Feature | Mean |SHAP| | Rank |
|---------|-------------|------|
| n_doubles | **4.84** | 1 |
| trump_count | **4.39** | 2 |
| n_singletons | 2.17 | 3 |
| count_points | 2.17 | 4 |
| total_pips | 2.00 | 5 |

SHAP interaction analysis showed main effects dominate: n_doubles main effect accounts for 68% of total |SHAP|, trump_count for 64%.

## 3.4 Risk Prediction (σ[V])

**Finding**: Outcome variance is nearly unpredictable from hand features.

Regression on σ(V) yielded **R² = 0.08** (95% CI: [0.06, 0.20])—the model explains only 6-20% of variance. Cross-validated R² was **negative** (-0.34), indicating predictions worse than the mean.

**Interpretation**: The uncertainty in Texas 42 outcomes derives from unknown opponent hands, not from one's own hand features.

## 3.5 Domino Enrichment Analysis

### Winner vs Loser Enrichment (E[V])

Fisher's exact tests identified dominoes over-represented in high E[V] hands (top 25%):

| Domino | log₂(Enrichment) | p_adj | Interpretation |
|--------|-----------------|-------|----------------|
| **5-5** | +1.50 | 0.017 | 2.8× in winners |
| **6-0** | -1.56 | 0.017 | 3× in losers |

Only 2 of 28 dominoes survived FDR correction. 5-5 (double-five, 10 count points) is the strongest positive signal; 6-0 (weak domino) is the strongest negative.

### High-Risk vs Low-Risk Enrichment (σ[V])

| Domino | log₂(Enrichment) | p_adj | Interpretation |
|--------|-----------------|-------|----------------|
| **6-5** | +2.09 | 0.028 | 4× in high-risk |
| **5-5** | -1.32 | 0.028 | 2.5× in low-risk |
| **2-0** | -1.65 | 0.028 | 3× in low-risk |

**Key insight**: 5-5 is enriched in winners AND depleted in high-risk hands, confirming that good dominoes lead to predictable outcomes.

## 3.6 Game Phase Analysis

Best-move consistency varied dramatically by game depth (Figure 2):

| Phase | Depth | Consistency | n States |
|-------|-------|-------------|----------|
| Early game | 24-28 | 40% | 18 |
| Mid-game | 5-23 | 22% | 147,529 |
| End-game | 0-4 | **100%** | 19,472 |

**Interpretation**: Games transition from **order → chaos → resolution**. Opening hands offer clear best moves, mid-game expands options creating strategic uncertainty, and end-game collapses to deterministic play.

## 3.7 Power Analysis

Post-hoc power analysis confirmed adequate sample size:

| Effect | Observed | Power | n for 80% |
|--------|----------|-------|-----------|
| r(E[V], σ[V]) = -0.38 | 2.6×10⁻⁸ | 1.000 | 51 |
| r(n_doubles, E[V]) = +0.40 | 6.9×10⁻⁹ | 1.000 | 46 |
| R²(hand→E[V]) = 0.26 | <0.001 | 1.000 | 57 |

All primary findings have power ≈ 1.00 at n=200. To detect smaller effects (|r| = 0.1), n ≈ 782 would be required.

## 3.8 Embedding Analysis

Word2Vec embeddings of domino co-occurrence (n=40,000 hands) showed weak clustering:

| Comparison | Mean Similarity |
|------------|-----------------|
| Double-to-double | 0.079 |
| Double-to-non-double | 0.071 |
| Random baseline | 0.069 |

UMAP projection confirmed no strong clusters. Dominoes appear strategically undifferentiated in co-occurrence space; value comes from game context (trump, position) rather than hand composition.

## 3.9 Summary of Effect Sizes

| Finding | Effect | Magnitude |
|---------|--------|-----------|
| E[V] ↔ σ(V) correlation | r = -0.38 | Medium |
| n_doubles → E[V] | r = +0.40 | Medium |
| ≥2 doubles vs <2 on E[V] | d = +0.76 | Medium |
| Hand features → E[V] | R² = 0.26 | Large |
| Hand features → σ(V) | R² = 0.08 | Small |
