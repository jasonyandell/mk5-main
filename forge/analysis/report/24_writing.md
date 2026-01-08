# 24: Writing

Publication figures and visual summaries for Texas 42 oracle analysis.

> **Epistemic Status**: This module creates publication figures for oracle (perfect-information minimax) analysis. All visualized findings describe oracle outcomes. The "napkin formula" and SHAP explanations characterize oracle predictions. Figures labeled "for practitioners" reflect oracle-derived heuristics whose applicability to human gameplay is untested.

## Overview

Module 24 creates publication-quality visualizations that synthesize findings from earlier oracle analysis modules. The focus is on clear figures that accurately represent oracle findings.

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

### Key Insight (Oracle Marginalization)

The same hand can have different oracle outcomes depending on opponent holdings. Marginalization quantifies this oracle uncertainty by computing statistics across opponent configurations.

**Note**: This methodology produces oracle E[V] and σ(V), not human game outcome distributions.

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
Distill the oracle regression findings into a simple, memorable formula for predicting oracle E[V].

### The Oracle Formula

**Oracle E[V] ≈ 14 + 6×(n_doubles) + 3×(trump_count)**

Where:
- `n_doubles` = number of doubles in hand (0-7)
- `trump_count` = number of trumps (0-7)

**Note**: This formula predicts oracle (perfect-information minimax) expected outcomes, not human game outcomes.

### Examples (Oracle Predictions)

| Hand Type | n_doubles | trump_count | Oracle E[V] |
|-----------|-----------|-------------|-------------|
| **Weak** | 0 | 1 | 14 + 0 + 3 = **17** |
| **Average** | 2 | 1 | 14 + 12 + 3 = **29** |
| **Strong** | 3 | 2 | 14 + 18 + 6 = **38** |

### Key Insights (Oracle Data)

1. **Each double adds ~6 oracle points** to oracle expected value
2. **Each trump adds ~3 oracle points**
3. **Doubles matter twice as much as trumps** for oracle predictions (ratio 6:3 = 2:1)
4. Model explains ~26% of oracle variance (R² = 0.26)

### Derivation

From the bootstrap regression analysis (13a):
- Intercept ≈ 14 (baseline expected value)
- n_doubles coefficient ≈ 5.7 (rounded to 6)
- trump_count coefficient ≈ 3.2 (rounded to 3)

### Why This Works (Oracle Mechanics)

1. **Doubles are trick winners in oracle**: Each double is the highest card in its suit under perfect play
2. **Trumps control in oracle**: Having trumps lets you win when you can't follow suit
3. **Additive effects in oracle**: SHAP analysis (14c) confirmed minimal interaction between features for oracle prediction

### Limitation (Oracle Scope)

This is a simplified oracle model. Oracle outcomes depend on opponent hands (captured via marginalization). **Whether this formula applies to human gameplay is untested.** Human games involve hidden information, inference, and suboptimal play not captured by the oracle.

### Files Generated

- `results/figures/fig4_napkin_formula.png` (300 DPI)
- `results/figures/fig4_napkin_formula.pdf` (vector)

---

## 24c: SHAP Summary (fig6)

### Purpose
Visualize oracle feature importance using SHAP (SHapley Additive exPlanations) for oracle model interpretability.

### Method

- Model: GradientBoostingRegressor predicting oracle E[V] (n_estimators=100, max_depth=3)
- SHAP: TreeExplainer for exact, fast computation of oracle model explanations
- Train R² = 0.81 (note: overfitting, but SHAP still informative for oracle feature importance)

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

### Key Findings (Oracle Model)

1. **n_doubles and trump_count dominate oracle prediction**: Together account for ~45% of total oracle feature importance
2. **Monotonic relationships in oracle**: More doubles/trumps → higher oracle SHAP values
3. **Other features have smaller, nonlinear effects in oracle**: Captured by GradientBoosting
4. **Confirms oracle napkin formula**: The two significant features match oracle regression findings

### Interpretation (Oracle Explanations)

SHAP provides per-sample explanations for oracle predictions:
- Why does hand X have high oracle E[V]? → "n_doubles pushed it +8 oracle points above average"
- Why does hand Y have low oracle E[V]? → "Zero doubles contributed -6 oracle points"

**Note**: These SHAP explanations describe what drives oracle model predictions. Whether similar feature importance holds for human game outcomes is untested.

### Files Generated

- `results/figures/fig6_shap_summary.png` (300 DPI) - Beeswarm plot
- `results/figures/fig6_shap_summary.pdf` (vector)
- `results/figures/fig6_shap_panels.png` (300 DPI) - Multi-panel figure
- `results/figures/fig6_shap_panels.pdf` (vector)

---

## Summary (Oracle Publication Figures)

Module 24 synthesizes key oracle findings into publication-ready figures:

1. **Methodology (fig1)**: Shows how marginalization yields oracle E[V] and σ(V) from seeds
2. **Napkin Formula (fig4)**: Formula for predicting oracle E[V] from hand features
3. **SHAP Summary (fig6)**: Oracle model feature importance visualization

### Connections to Other Modules

- **12b**: Unified feature extraction used in SHAP
- **13a**: Bootstrap regression coefficients → oracle napkin formula
- **14a**: Original SHAP analysis → fig6 recreates with publication styling
- **19c**: Bayesian model comparison confirms oracle napkin formula optimality

**Scope limitation**: All figures visualize oracle (perfect-information) findings. Captions and legends should clearly indicate oracle scope to avoid implying human gameplay applicability.

---

## Further Investigation

### Validation Needed

1. **Human-applicable figures**: If human gameplay data becomes available, corresponding figures showing human outcome distributions would allow comparison with oracle figures.

2. **Figure captions**: Publication figures should include explicit captions noting oracle scope. Current visualizations might be misinterpreted as applying to human gameplay.

3. **Practitioner testing**: The napkin formula is labeled "for practitioners" but untested on human games. A validation study comparing formula predictions to human outcomes would ground (or refute) this claim.

### Methodological Questions

1. **Visualization choices**: The figures emphasize oracle findings. Should alternative visualizations highlight the gap between oracle and human play?

2. **Sample representation**: The 200-seed sample may not cover all hand configurations. Do the visualizations adequately represent edge cases?

3. **Audience assumptions**: Publication figures assume readers understand oracle vs human distinction. Should figures include explicit annotations for general audiences?

### Open Questions

1. **Transfer of visual insights**: Do the visual patterns (e.g., SHAP beeswarm showing doubles/trumps dominance) hold for human outcome models?

2. **Publication venue requirements**: Different venues may have different requirements for oracle vs human distinction clarity. How should figures be adapted?

3. **Interactive visualizations**: Would interactive figures (allowing comparison across oracle configurations) be more informative than static publication figures?
