# 08: Deep Count Capture Analysis

## Overview

This section extends the count analysis from Section 03, investigating three key questions:
1. **Lock-in depth**: When does each count's fate become determined?
2. **Residual decomposition**: What explains the ~0.3-0.4 residual variance within basins?
3. **Capture predictors**: What features predict who captures each count?

**Key finding**: Count capture remains uncertain until the last 2-3 dominoes, and the primary predictor of capture is simply who holds the count domino.

---

## 8.1 Count Lock-In Depth Analysis

### Question
At what depth does each count domino become "locked in" (capture probability → 0 or 1)?

### Method
For states at each depth, track P(Team 0 captures count_i) along the principal variation. Define "locked" as P < 0.05 or P > 0.95.

### Results

![Lock-In Curves](../results/figures/08a_lock_in_curves.png)

| Domino | Points | Lock-in Depth | Max Uncertain Depth | Uncertain Depths |
|--------|--------|---------------|---------------------|------------------|
| 3-2    | 5      | 2             | 27                  | 26               |
| 4-1    | 5      | 2             | 27                  | 26               |
| 5-0    | 5      | 3             | 27                  | 25               |
| 5-5    | 10     | 2             | 27                  | 26               |
| 6-4    | 10     | 3             | 27                  | 25               |

![Lock-In Combined](../results/figures/08a_lock_in_combined.png)

### Interpretation

**Counts remain uncertain until depth 2-3** (the last few dominoes). This means:
1. The oracle is NOT simply confirming foregone conclusions
2. Strategic play matters throughout most of the game
3. Counts can swing until the very end

This contradicts a hypothesis that counts "lock in early." Instead, the game maintains genuine uncertainty almost to the finish.

---

## 8.2 Residual Variance Decomposition

### Question
What explains the ~0.31-0.38 within-basin variance observed in Section 03?

### Theoretical Bounds
- Total game points: 42
- Count points: 35 (from 5 count dominoes)
- Trick points: 7 (1 per trick)

If V = count_diff + trick_diff, then residual variance should be bounded by Var(trick_diff) ≤ 7.

### Results

![Residual Distribution](../results/figures/08b_residual_distribution.png)

**Overall residual statistics**:
- Mean: ~0 (unbiased)
- Range: approximately [-7, +7] (matches theoretical trick point range)

### Variance by Basin

![Residual by Basin](../results/figures/08b_residual_by_basin.png)

Residual variance is roughly uniform across basins, suggesting trick outcomes don't systematically favor particular count configurations.

### Variance by Depth

![Residual by Depth](../results/figures/08b_residual_by_depth.png)

Residual variance decreases with depth as remaining trick outcomes become determined.

### Variance Decomposition

![Variance Decomposition](../results/figures/08b_variance_decomposition.png)

| Component | Variance | % of Total |
|-----------|----------|------------|
| Total V   | ~600     | 100%       |
| Count capture (explained) | ~550 | ~92% |
| Residual (trick points) | ~50 | ~8% |

**Key finding**: Count capture explains ~92% of V variance. The residual (~8%) corresponds to the 7 non-count trick points.

---

## 8.3 Count Capture Predictors

### Question
What features predict count capture outcomes from the initial deal?

### Method
Extract features at game start:
- Trump advantage (team0 trumps - team1 trumps)
- Per-count: which team holds it, whether it's a trump

Fit logistic regression and random forest per count domino.

### Baseline: Who Holds It

| Domino | Points | Holder's Team Captures % |
|--------|--------|--------------------------|
| 3-2    | 5      | 42.9%                    |
| 4-1    | 5      | 64.3%                    |
| 5-0    | 5      | 64.3%                    |
| 5-5    | 10     | 71.4%                    |
| 6-4    | 10     | 57.1%                    |

Holding the count gives roughly 50-70% capture probability — better than random but far from deterministic.

### Model Performance

![Feature Importance](../results/figures/08c_feature_importance.png)

| Domino | Logistic CV Acc | RF CV Acc |
|--------|-----------------|-----------|
| 3-2    | 0.867           | 0.900     |
| 4-1    | 0.820           | 0.827     |
| 5-0    | 0.587           | 0.433     |
| 5-5    | 0.647           | 0.680     |
| 6-4    | 0.680           | 0.713     |
| **Mean** | **0.720**     | **0.711** |

### Feature Importance

![Aggregate Importance](../results/figures/08c_aggregate_importance.png)

**Most important features**:
1. Trump advantage
2. Which team holds each count
3. Whether count is a trump

### Interpretation

Models achieve ~72% accuracy predicting capture from initial features. This is better than the 50-57% baseline (holder wins) but still leaves substantial uncertainty. The game is NOT "decided at declaration time" — play matters.

---

## 8.4 Synthesis

### What We Learned

1. **Counts don't lock in early**: Uncertainty persists until depth 2-3. The game has genuine strategic depth throughout.

2. **Residual = trick points**: The ~8% unexplained variance matches the theoretical 7 trick points. Count capture fully explains the count-point component.

3. **Prediction is imperfect**: Initial features predict capture with ~72% accuracy. Holding the count helps (50-70%), but trump control and play quality matter.

### Implications

- **For players**: Don't give up early — counts can flip until the last tricks
- **For models**: A model that perfectly predicts count capture would achieve R² ≈ 0.92
- **For complexity**: The game's irreducible randomness comes from play decisions, not initial deal

---

## 8.5 Manifold Analysis

### Question
Do game paths lie on a low-dimensional manifold? Is the intrinsic dimension ≈ 5 (one per count)?

### Method
1. Sample paths from many seeds/declarations
2. Compute basin (5-bit count capture pattern) for each path
3. PCA on basin features to estimate intrinsic dimension
4. Basin entropy to measure outcome diversity

### Results

![Manifold Overview](../results/figures/08d_manifold_overview.png)

| Metric | Value |
|--------|-------|
| Seeds analyzed | 28 |
| Unique basins observed | 13 of 32 |
| PCA components for 95% variance | **5** |
| Basin entropy | 3.06 bits (61% of max 5.0) |
| Effective outcomes | ~21 |

### PCA Variance Explained

| Component | Variance | Cumulative |
|-----------|----------|------------|
| PC1 | 47.3% | 47.3% |
| PC2 | 29.4% | 76.7% |
| PC3 | 10.3% | 87.1% |
| PC4 | 7.6% | 94.7% |
| PC5 | 5.3% | 100.0% |

### Interpretation

**5 components for 95% variance**: This matches the hypothesis that the game has ~5 effective degrees of freedom (one per count domino). The count capture outcomes form the natural coordinates of the game's "manifold."

**13 of 32 basins observed**: Not all count combinations are equally reachable. Some basins (where one team sweeps all counts) are rare.

**Entropy = 61% of max**: Outcomes are neither fully uniform nor highly concentrated. There's genuine diversity in how games play out.

---

## 8.6 Synthesis

### Key Findings

| Analysis | Finding | Implication |
|----------|---------|-------------|
| 08a Lock-in | Counts uncertain until depth 2-3 | Game has strategic depth throughout |
| 08b Residual | ~92% variance from counts, ~8% from tricks | Count capture is the game |
| 08c Predictors | ~72% accuracy from initial features | Play matters, not just the deal |
| 08d Manifold | 5 dimensions, 61% entropy | The game explores its possibility space |

### The Game's True Structure

Texas 42 is a **5-dimensional game** in the space of count capture outcomes:
- Each count domino represents one degree of freedom
- Count capture explains ~92% of V variance
- The remaining 8% is from trick points (7 points distributed among 7 tricks)
- Play decisions matter: outcomes aren't determined by the initial deal

### Implications for AI

A perfect count-capture predictor would achieve R² ≈ 0.92 on V. The remaining 8% requires modeling trick-by-trick dynamics. This suggests a two-level architecture:
1. **Count module**: Predict which team captures each count
2. **Trick module**: Given counts, predict final trick point distribution

---

*End of Section 08*
