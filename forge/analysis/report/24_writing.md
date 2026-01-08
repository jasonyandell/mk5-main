# 24: Writing

Publication figures and visual summaries for Texas 42 oracle analysis.

## Overview

Module 24 creates publication-quality visualizations that synthesize findings from earlier analysis modules. The focus is on clear, actionable figures suitable for practitioners.

---

## 24a: Methodology Schematic (fig1)

### Purpose
Visual explanation of the marginalization approach used throughout the analysis.

### Key Elements

The figure shows the data pipeline:

1. **Input**: A seed (e.g., 42) determines P0's fixed hand
2. **Fixed P0 Hand**: `deal_from_seed(seed)` gives 7 dominoes
3. **Marginalization**: Multiple opponent configurations sampled (3 per seed)
4. **Oracle**: Perfect minimax play computed for each configuration
5. **Aggregation**: Mean and variance across configurations yield E[V] and σ(V)

### Key Insight

The same hand can have different outcomes depending on opponent holdings. Marginalization quantifies this uncertainty.

### Sample Size

- n = 200 seeds (unique declarer hands)
- 3 configurations per seed
- 600 total games analyzed

### Files Generated

- `results/figures/fig1_methodology.png` (300 DPI)
- `results/figures/fig1_methodology.pdf` (vector)

---

## 24b: Napkin Formula (fig4)

### Purpose
Distill the regression findings into a simple, memorable rule-of-thumb for practitioners.

### The Formula

**E[V] ≈ 14 + 6×(n_doubles) + 3×(trump_count)**

Where:
- `n_doubles` = number of doubles in hand (0-7)
- `trump_count` = number of trumps (0-7)

### Examples

| Hand Type | n_doubles | trump_count | E[V] |
|-----------|-----------|-------------|------|
| **Weak** | 0 | 1 | 14 + 0 + 3 = **17** |
| **Average** | 2 | 1 | 14 + 12 + 3 = **29** |
| **Strong** | 3 | 2 | 14 + 18 + 6 = **38** |

### Key Insights

1. **Each double adds ~6 points** to expected value
2. **Each trump adds ~3 points**
3. **Doubles matter twice as much as trumps** (ratio 6:3 = 2:1)
4. Model explains ~26% of variance (R² = 0.26)

### Derivation

From the bootstrap regression analysis (13a):
- Intercept ≈ 14 (baseline expected value)
- n_doubles coefficient ≈ 5.7 (rounded to 6)
- trump_count coefficient ≈ 3.2 (rounded to 3)

### Why This Works

1. **Doubles are trick winners**: Each double is the highest card in its suit
2. **Trumps control**: Having trumps lets you win when you can't follow suit
3. **Additive effects**: SHAP analysis (14c) confirmed minimal interaction between features

### Limitation

This is a simplified model. Actual outcomes depend on opponent hands and play.

### Files Generated

- `results/figures/fig4_napkin_formula.png` (300 DPI)
- `results/figures/fig4_napkin_formula.pdf` (vector)

---

## 24c: SHAP Summary (fig6)

### Purpose
Visualize feature importance using SHAP (SHapley Additive exPlanations) for machine learning interpretability.

### Method

- Model: GradientBoostingRegressor (n_estimators=100, max_depth=3)
- SHAP: TreeExplainer for exact, fast computation
- Train R² = 0.81 (note: overfitting, but SHAP still informative)

### Feature Importance Ranking

| Rank | Feature | Mean |SHAP| | Direction |
|------|---------|-------------|-----------|
| 1 | Number of Doubles | **4.84** | ↑ |
| 2 | Trump Count | **4.39** | ~ |
| 3 | Number of Singletons | 2.17 | ↓ |
| 4 | Count Points | 2.17 | ~ |
| 5 | Total Pips | 2.00 | ↓ |
| 6 | Six-High Cards | 1.44 | ~ |
| 7 | Has Trump Double | 1.09 | ~ |
| 8 | Max Suit Length | 0.80 | ~ |
| 9 | Number of Voids | 0.79 | ~ |
| 10 | Five-High Cards | 0.67 | ~ |

### Visualizations

1. **Beeswarm plot**: Shows per-sample SHAP values for all hands
   - x-axis: SHAP value (impact on prediction)
   - Color: Feature value (high = red, low = blue)
   - Position: Each dot is one hand

2. **Multi-panel figure**:
   - Panel A: Horizontal bar chart of mean |SHAP| by feature
   - Panel B: Scatter plot of n_doubles effect (feature value vs SHAP)
   - Panel C: Scatter plot of trump_count effect

### Key Findings

1. **n_doubles and trump_count dominate**: Together account for ~45% of total feature importance
2. **Monotonic relationships**: More doubles/trumps → higher SHAP values
3. **Other features have smaller, nonlinear effects**: Captured by GradientBoosting
4. **Confirms napkin formula**: The two significant features match regression findings

### Interpretation

SHAP provides per-sample explanations:
- Why does hand X have high E[V]? → "n_doubles pushed it +8 points above average"
- Why does hand Y have low E[V]? → "Zero doubles contributed -6 points"

### Files Generated

- `results/figures/fig6_shap_summary.png` (300 DPI) - Beeswarm plot
- `results/figures/fig6_shap_summary.pdf` (vector)
- `results/figures/fig6_shap_panels.png` (300 DPI) - Multi-panel figure
- `results/figures/fig6_shap_panels.pdf` (vector)

---

## Summary

Module 24 synthesizes key findings into publication-ready figures:

1. **Methodology (fig1)**: Shows how marginalization yields E[V] and σ(V) from seeds
2. **Napkin Formula (fig4)**: Practitioner-friendly bidding heuristic
3. **SHAP Summary (fig6)**: ML-interpretable feature importance visualization

### Connections to Other Modules

- **12b**: Unified feature extraction used in SHAP
- **13a**: Bootstrap regression coefficients → napkin formula
- **14a**: Original SHAP analysis → fig6 recreates with publication styling
- **19c**: Bayesian model comparison confirms napkin formula optimality
