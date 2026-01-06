# 03: Count Domino Analysis

## Overview

This section presents our most significant finding: **count domino capture explains 76% of V variance overall, rising to >99% in late-game positions**. This suggests the game's complexity concentrates in a small number of key dominoes.

---

## 3.1 The Count Dominoes

Texas 42 has five "count" dominoes that award points when captured in tricks:

| Domino | Pips | Points | % of Total |
|--------|------|--------|------------|
| 5-5 | 10 | 10 | 23.8% |
| 6-4 | 10 | 10 | 23.8% |
| 5-0 | 5 | 5 | 11.9% |
| 4-1 | 5 | 5 | 11.9% |
| 3-2 | 5 | 5 | 11.9% |

**Total count points**: 35 of 42 possible (83.3%)
**Remaining 7 points**: 1 per trick won (7 tricks × 1 point)

**Hypothesis**: If count capture determines most points, it should predict V strongly.

---

## 3.2 Count Capture Statistics

From a sample of 50,000 states:

| Domino | Played % | Team0 Capture % |
|--------|----------|-----------------|
| 3-2 | 68.1% | 40.7% |
| 4-1 | 65.6% | 50.1% |
| 5-0 | 66.4% | 29.3% |
| 5-5 | 65.7% | 68.0% |
| 6-4 | 72.6% | 36.5% |

**Observations**:
1. Counts are played ~65-73% of the time by late game
2. 5-5 strongly favors Team0 (68.0%) — likely correlation with declaration
3. 5-0 strongly favors Team1 (70.7%) — possibly a defensive domino

**Question**: Is the Team0 bias due to declaration advantage, or does the domino distribution vary by seed?

---

## 3.3 Regression Model: Count Capture → V

We model V as a linear function of count capture indicators:

```
V = Σᵢ βᵢ · capture(countᵢ, Team0) + ε
```

Where `capture(d, t)` = 1 if team t captured domino d, 0 otherwise.

### Learned Coefficients

| Count | True Points | Learned β | Ratio |
|-------|-------------|-----------|-------|
| 3-2 | 5 | 5.84 | 1.17 |
| 4-1 | 5 | 5.66 | 1.13 |
| 5-0 | 5 | 4.92 | 0.98 |
| 6-4 | 10 | 10.42 | 1.04 |
| 5-5 | 10 | 9.14 | 0.91 |
| depth | - | 0.088 | - |

**Observations**:
1. Learned coefficients closely match true point values (ratio 0.91-1.17)
2. 3-2 and 4-1 are slightly overweighted (capturing them also implies trick wins)
3. 5-5 is slightly underweighted (perhaps easier to lose)
4. Depth coefficient is small but positive (later game → more determined)

### Model Performance

| Model | R² | RMSE |
|-------|-----|------|
| Simple (fixed β = point values) | 0.552 | 6.96 |
| Learned coefficients | 0.759 | 5.11 |
| Learned + depth | 0.759 | 5.11 |

![Model Comparison](../results/figures/03c_model_comparison.png)

**Key finding**: Learned coefficients achieve R² = 0.759, explaining three-quarters of V variance with only 5 binary features.

---

## 3.4 Variance Decomposition by Depth

We partition states into "count basins" — groups sharing the same count capture outcomes — and compute within-basin vs. total variance:

| Depth | Total σ² | Within-Basin σ² | R² (explained) | n States | n Basins |
|-------|----------|-----------------|----------------|----------|----------|
| 5 | 96.7 | 33.5 | 0.653 | 7,149 | 16 |
| 6 | 82.6 | 20.0 | 0.758 | 4,494 | 18 |
| 7 | 81.6 | 10.2 | 0.875 | 2,798 | 16 |
| 8 | 73.2 | 0.31 | **0.996** | 1,412 | 15 |
| 9 | 101.9 | 32.8 | 0.678 | 14,594 | 23 |
| 10 | 80.8 | 18.3 | 0.773 | 7,799 | 25 |
| 11 | 78.3 | 8.9 | 0.887 | 3,830 | 20 |
| 12 | 66.4 | 0.31 | **0.995** | 1,345 | 14 |
| 13 | 104.2 | 29.6 | 0.716 | 3,325 | 22 |
| 14 | 76.3 | 16.5 | 0.784 | 1,654 | 18 |
| 15 | 70.8 | 6.7 | 0.905 | 821 | 11 |
| 16 | 59.1 | 0.38 | **0.994** | 197 | 7 |
| 17 | 89.7 | 27.9 | 0.689 | 205 | 7 |
| 18 | 108.6 | 11.7 | 0.892 | 115 | 6 |

![Variance by Depth](../results/figures/03b_var_explained_by_depth.png)

**Pattern**: R² follows a 4-depth cycle:
- Depths 5, 9, 13, 17 (first play of trick): R² ≈ 0.65-0.72
- Depths 8, 12, 16 (trick boundary): R² > 0.99

**Interpretation**: At trick boundaries, all uncertainty resolves to count capture. Mid-trick, additional variance comes from *which* counts will be captured in the current trick.

---

## 3.5 Basin Structure Analysis

Within each count basin, what explains the residual variance?

![Basin V Distributions](../results/figures/03b_basin_v_distributions.png)

**Late-game basins** (depth 8, 12, 16): Within-basin σ² < 0.4, nearly deterministic
**Early-game basins** (depth 5, 9): Within-basin σ² ≈ 20-35, substantial residual variation

The residual variance at depth 5 implies other factors matter early:
- Which team will win future tricks (not yet determined)
- Positional advantages that don't show in current count state

---

## 3.6 Residual Analysis

![Residual Analysis](../results/figures/03c_residual_analysis.png)

**Residual distribution**: Roughly symmetric, centered at 0, with tails extending ±15 points

![Residual by Depth](../results/figures/03c_residual_by_depth.png)

**Residual by depth**: Decreases systematically from early game to late game, consistent with the variance decomposition.

---

## 3.7 Implications

### The Game's Core Structure
Texas 42 is fundamentally a count-capture game. The trick-taking mechanics determine *which team captures which counts*, and counts determine ~76% of the outcome. This is analogous to how chess is "about" king safety despite having many other pieces.

### For Neural Networks
A model that accurately predicts count capture outcomes should achieve high V prediction accuracy. Our 97.8% accurate Transformer explicitly encodes count information (0/5/10 point value per domino), which this analysis validates.

### For Simplified Oracles
A "count-only" oracle that tracks only count capture states would be ~250× smaller (16 basins vs ~4000 states per depth) while retaining >99% accuracy in late game and ~75% overall.

### The Remaining 24%
The unexplained variance comes from:
1. Mid-trick uncertainty (which counts will be captured)
2. Non-count trick points (7 points total)
3. Subtle positional advantages (tempo, trump control)

---

## 3.8 Questions for Statistical Review

1. **Model specification**: Should we include interaction terms (e.g., capturing both 5-5 and 6-4)? The current model assumes additivity.

2. **Heteroscedasticity**: Residual variance depends strongly on depth. Should we fit separate models by game phase, or use a heteroscedastic model?

3. **Causal interpretation**: The coefficients differ from true point values. Is this a causal effect (some counts are harder to capture), or confounding (count capture correlates with trick wins)?

4. **Basin definition**: We define basins by exact count outcomes. Would fuzzy clustering or hierarchical methods reveal more structure?

5. **Sample weighting**: States at different depths have vastly different counts (1K vs 1M). How should we weight the regression?

---

*Next: [04 Symmetry Analysis](04_symmetry.md)*
