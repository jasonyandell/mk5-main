# 09: Path Analysis Battery

## Overview

This section investigates the fundamental structure of game paths through geometric and information-theoretic analysis.

**Core Question:** What is the effective dimensionality of Texas 42?

**Sub-analyses:**
- 09a: Convergence (basin funnel, depth, divergence points)
- 09b: Geometry (intrinsic dimension, clustering, manifold)
- 09c: Information theory (entropy, conditional entropy, mutual info)
- 09d: Temporal (autocorrelation, change points, periodicity)
- 09e: Topology (homology, Reeb graphs, DAG structure)
- 09f: Compression (suffix/prefix sharing, LZ complexity)
- 09g: Prediction (basin from k moves, counterfactuals)
- 09h: Fractal/Scaling (roughness, DFA, branching dimension)
- 09i: Decision quality (Q-gap, mistake impact, decision sparsity)

---

## 9.1 Convergence Analysis (Basin Funnel)

### Question
Do all paths from a (seed, declaration) converge to 1-2 basins? Is the game "decided at declaration"?

### Method
A **basin** is defined by the count capture signature — a 5-bit value indicating which team captured each of the 5 count dominoes. There are 2^5 = 32 possible basins.

For each (seed, decl), sample multiple starting positions at various depths and trace to terminal basin. Count unique basins reachable.

### Results

![Basin Funnel](../results/figures/09a_basin_funnel.png)

| Metric | Value |
|--------|-------|
| Total deals analyzed | 4 |
| Mean unique basins per deal | **16.25** |
| Median unique basins per deal | 16.0 |
| Max unique basins | 21 |
| % single-outcome deals | 0.0% |
| % ≤2 outcome deals | 0.0% |
| % ≤4 outcome deals | 0.0% |

### Entropy Decay by Depth

![Entropy Decay](../results/figures/09a_entropy_decay.png)

Basin entropy remains high (~2-3 bits) throughout the game, only dropping near the terminal states.

### Divergence Points

![Divergence Points](../results/figures/09a_divergence_points.png)

| Metric | Value |
|--------|-------|
| Mean divergence depth | 26.0 |
| Median divergence depth | 27.0 |

Paths diverge early (depth 26-27, after trick 1-2), not late.

### Interpretation

**The "decided at declaration" hypothesis is REJECTED.**

With mean unique basins ≈ 16 per deal, there's genuine strategic depth. All analyzed deals have multiple reachable basins — none have a single deterministic outcome.

**Key findings:**
1. **Many outcomes possible**: ~16 of 32 basins are reachable from a typical deal
2. **Early divergence**: Paths split early (trick 1-2), not converging until the very end
3. **High entropy throughout**: Basin uncertainty remains ~2-3 bits until terminal

**Implication for ML:** A transformer cannot simply "classify the deal type" — it must genuinely reason about game dynamics. The effective dimensionality is NOT ~5 (count capture outcomes), but much higher.

---

## 9.2 Geometry Analysis

### Question
What is the intrinsic dimension of path space? Is it ≈ 5 (one per count domino)?

### Method
For each sampled starting state, trace the principal variation (PV) and record:
- **V-trajectory**: [V₀, V₁, ..., V_terminal] — the value at each step along optimal play
- **Basin ID**: 5-bit encoding of count capture outcomes

Apply dimensionality analysis:
1. **PCA**: Find components explaining 90%/95%/99% variance
2. **Levina-Bickel MLE**: k-NN based intrinsic dimension estimator
3. **K-means clustering**: Compare cluster assignments to basin IDs using ARI/NMI

### Results

![PCA Variance](../results/figures/09b_pca_variance.png)

| Metric | Value |
|--------|-------|
| Total paths analyzed | 30 |
| Unique basins observed | 14 |
| PCA dim for 90% variance | 2 |
| PCA dim for 95% variance | **5** |
| Levina-Bickel dim (k=10) | 3.04 |

### Clustering Analysis

![Clustering](../results/figures/09b_clustering.png)

| Metric | Value |
|--------|-------|
| K-means ARI (best k) | 0.394 |
| K-means NMI (best k) | 0.609 |

Clustering moderately aligns with basin IDs — paths leading to the same basin tend to cluster together, but not perfectly.

### Interpretation

**The "5-dimensional" hypothesis is SUPPORTED.**

The PCA 95% variance dimension is exactly 5, matching the number of count dominoes. The Levina-Bickel estimator suggests even lower effective dimension (~3).

**Key findings:**
1. **Low intrinsic dimension**: 95% of V-trajectory variance explained by 5 components
2. **Count capture dominates**: The 5 count domino outcomes largely explain path structure
3. **Partial clustering**: Paths to same basin are similar but not identical (ARI=0.39)

**Reconciliation with 9.1:** While 09a found ~16 distinct basins are reachable (high outcome diversity), 09b finds the *structure* of paths is low-dimensional. This means:
- **Many endpoints** (basins) are possible from a deal
- But the **path geometry** connecting them is governed by just ~5 degrees of freedom
- The game has strategic depth in *which* basin to reach, but paths are constrained

**Implication for ML:** A transformer can learn a low-dimensional latent representation (~5D) for V-trajectories. The challenge is not representing path structure but predicting *which* of the ~16 reachable basins optimal play achieves.

---

## 9.3 Information Theory Analysis

### Question
Does the basin capture all information about a path? Are paths deterministic from the deal?

### Method
Compute information-theoretic measures:
1. **H(path), H(basin)**: Raw entropy of paths and basins
2. **H(path|basin)**: Residual path entropy after knowing basin
3. **H(path|deal)**: Entropy of paths given (seed, decl)
4. **I(early; late)**: Mutual information between early and late play

### Results

![Mutual Information](../results/figures/09c_mutual_info.png)

| Metric | Value |
|--------|-------|
| Total paths | 28 |
| H(path) | 4.81 bits |
| H(basin) | 3.97 bits |
| H(path\|basin) | 0.84 bits |
| I(path; basin) | 3.97 bits |
| H(path\|deal) | **0.00 bits** |

### Interpretation

**Key findings:**

1. **Paths are deterministic from deal**: H(path|deal) = 0. Given the hands and declaration, the optimal play path is unique. There is no "choice" in minimax optimal play.

2. **Basin explains 82.5% of path entropy**: Knowing which basin a path ends in tells you most (but not all) about the path. The remaining 0.84 bits capture which of several paths to the same basin was taken.

3. **Strong early-late coupling**: I(early₈; late₈) = 100% normalized. Early play completely determines late play along optimal paths.

**Implication for ML:**
- The oracle provides unique optimal paths - no "exploration" or "alternative solutions"
- A model that learns the deal→path mapping learns deterministic behavior
- Training is learning a pure function, not a distribution

**Reconciliation with 09g (80.9% forced moves):**
These findings are consistent. Given the deal, most positions have only one legal move (forced). The few non-forced positions have a single optimal action determined by the minimax solution. The game tree may have branching, but the *optimal* path through it is unique.

---

## 9.4 Temporal Analysis

### Question
Does the path structure reflect the 4-play trick periodicity? Does path history matter for predicting V?

### Method
Trace V-trajectories along principal variations and analyze:
1. **Autocorrelation**: Correlation of V(t) with V(t-k) at various lags
2. **Change point detection**: PELT algorithm for regime changes
3. **Periodicity**: Fourier analysis of ΔV for trick-boundary signal
4. **Predictive models**: Compare R² of V ~ depth vs V ~ V_lag1

### Results

![Autocorrelation](../results/figures/09d_autocorrelation.png)

| Metric | Value |
|--------|-------|
| Total paths analyzed | 74 |
| Mean path length | 20.9 |
| Autocorr at lag 1 | **0.737** |
| Autocorr at lag 4 | 0.468 |
| Change points per path | 2.9 |

### Predictive Model Comparison

| Model | R² |
|-------|-----|
| V ~ depth | **0.0052** |
| V ~ V_lag1 | **0.8013** |
| V ~ depth + V_lag1 | 0.8015 |

### Periodicity Analysis

![Periodicity](../results/figures/09d_periodicity.png)

Fourier analysis shows some signal at the 4-period (trick boundary) frequency, with lag-4 autocorrelation (0.468) remaining substantial.

### Change Point Detection

![Change Points](../results/figures/09d_changepoints.png)

Mean 2.9 change points per path, distributed throughout the game rather than concentrated at trick boundaries.

### Interpretation

**Key findings:**

1. **Path history dominates depth**: R²(lag1) = 0.80 vs R²(depth) = 0.005. Knowing the previous V tells you almost everything; knowing depth tells you almost nothing. This is a striking result.

2. **Strong temporal memory**: Lag-1 autocorrelation of 0.74 indicates V evolves smoothly along paths. The game state "remembers" where it's been.

3. **Moderate trick-boundary signal**: Lag-4 autocorrelation (0.47) is substantial, suggesting the 4-move trick structure is visible in temporal dynamics.

4. **Change points are distributed**: ~3 regime changes per game, not concentrated at specific boundaries.

**Implication for ML:**
- A transformer MUST use positional/sequential information — depth alone is useless
- Recurrent processing or attention over the move sequence is essential
- The game cannot be modeled as i.i.d. samples at each depth

---

## 9.5 Topology Analysis

### Question
Do game paths form a simple tree, or do they have richer topological structure (reconvergence)?

### Method
1. **Branching analysis**: Count legal moves per state by depth
2. **State diversity**: Compare unique played-masks to total states at each depth
3. **Persistent homology**: Compute Betti numbers on path embedding space

### Results

![Branching](../results/figures/09e_branching.png)

| Metric | Value |
|--------|-------|
| Total states analyzed | 15,000 |
| Mean legal moves per state | **2.00** |
| % branch points (multi-move) | **61.8%** |
| Mean diversity ratio | **0.549** |
| β₀ (connected components) | 0 |
| β₁ (loops) | 0 |

### State Diversity by Depth

![Diversity](../results/figures/09e_diversity.png)

The diversity ratio (unique played-masks / total states) is ~0.55, meaning about half of the played-mask configurations correspond to multiple distinct game states.

### Interpretation

**Key findings:**

1. **Moderate branching**: 61.8% of states have multiple legal moves, with mean ~2 options. This aligns with 09i's finding of 61.1% multi-action states.

2. **Significant reconvergence**: Diversity ratio of 0.55 indicates that the same set of played dominoes can lead to different game states (different trick configurations, scores). The game DAG is NOT a tree.

3. **Simple path topology**: β₀=0, β₁=0 suggests no non-trivial loops in the path embedding space (when viewing V-trajectories as points).

**Implication for ML:**
- The game tree has DAG structure, not pure tree
- Same played dominoes → multiple possible states (order matters within tricks, but trick outcomes matter more than individual move order)
- A model could potentially learn "move-order invariance" for certain subsequences

---

## 9.6 Compression Analysis

### Question
How compressible are game paths? Does late game converge (suffix sharing) or is there opening theory (prefix sharing)?

### Method
For each sampled path (action sequence along PV):
1. **Suffix sharing**: Build trie on reversed sequences, measure common endings
2. **Prefix sharing**: Build trie on forward sequences, measure common openings
3. **LZ complexity**: Compress concatenated paths with zlib
4. **Minimum description length**: Compute H(path | basin)

### Results

![Prefix/Suffix Sharing](../results/figures/09f_prefix_suffix.png)

| Metric | Value |
|--------|-------|
| Total paths analyzed | 29 |
| Mean path length | 27.7 |
| Prefix compression ratio | 1.04x |
| Suffix compression ratio | 1.05x |
| LZ compression ratio | 2.15x |
| Mean shared prefix (same basin) | 0.29 |
| Mean shared suffix (same basin) | 0.27 |

### Information-Theoretic Metrics

![Entropy by Basin](../results/figures/09f_entropy_by_basin.png)

| Metric | Value |
|--------|-------|
| H(path) | 4.86 bits |
| H(path \| basin) | 2.72 bits |
| I(path; basin) | 2.13 bits |

### Interpretation

**Key findings:**

1. **No late game stereotype**: Paths don't share common endings (mean suffix sharing < 1 action). Even within the same basin, late game play varies.

2. **No opening theory**: Paths start diversely (mean prefix sharing < 1 action). There's no dominant opening sequence.

3. **Moderate compressibility**: LZ compression ratio of 2.15x is similar to random baseline (2.2x), indicating paths have minimal repetitive structure.

4. **Partial basin predictability**: Knowing the basin explains 44% of path entropy. Significant path variation remains after conditioning on outcome.

**Implication for data storage:** Standard compression (zlib) provides ~2x reduction. Domain-specific compression schemes (e.g., based on game structure) may not significantly outperform this. The oracle data cannot be dramatically compressed by exploiting path structure.

**Contrast with 09b:** While path *value trajectories* are low-dimensional (5 PCA components), the *action sequences* themselves are highly varied. The same basin can be reached through many different action paths.

---

## 9.7 Prediction Analysis

### Question
Can we predict the final basin from early moves? At what depth does prediction stabilize?

### Method
1. **Basin prediction by depth**: Train RandomForest classifier on action prefixes of length k, measure accuracy
2. **Path continuation entropy**: H(next action | prefix) at each prefix length
3. **Q-gap analysis**: Distribution of Q-gaps (best - second best) throughout game

### Results

![Basin Prediction](../results/figures/09g_basin_prediction.png)

| Metric | Value |
|--------|-------|
| Total paths analyzed | 28 |
| Unique basins | 14 |
| Depth for 90% prediction | N/A |
| Mean continuation entropy | 1.18 bits |

### Key Finding: Forced Moves

![Q-Gap by Position](../results/figures/09g_qgap_by_position.png)

| Metric | Value |
|--------|-------|
| Mean Q-gap | 2.9 |
| **% Forced moves** | **80.9%** |

**Most moves in Texas 42 are forced** — there's only one legal action. The 20% of positions with genuine choices are the critical decision points.

### Interpretation

**Key findings:**

1. **Basin prediction doesn't stabilize early**: With limited data, prediction accuracy remains low throughout. More data needed for definitive answer, but results suggest late-game determination.

2. **Moderate continuation entropy** (1.18 bits, 42% of max): Given a prefix, next moves are somewhat predictable but not deterministic.

3. **Most moves forced**: 80.9% of positions have only one legal action. The game's complexity emerges from the ~20% of positions where multiple moves are available.

**Implication for transformer training:**
- The model should focus learning capacity on the ~20% of non-forced positions
- Planning/search is likely required since basin prediction doesn't stabilize early
- The high fraction of forced moves may simplify training (fewer "decision" states to learn)

---

## 9.8 Fractal/Scaling Analysis

*To be completed*

---

## 9.9 Decision Quality Analysis

### Question
What fraction of moves are "real" decisions? How punishing are mistakes?

### Method
Sample states from oracle shards and analyze Q-value structure:
1. **Q-gap distribution**: (best Q - second best Q) measures decision difficulty
2. **Decision sparsity**: Fraction of positions with genuine choices at various thresholds
3. **Mistake impact**: Expected V drop for taking suboptimal action

### Results

![Q-gap Distribution](../results/figures/09i_qgap_distribution.png)

| Metric | Value |
|--------|-------|
| Total states analyzed | 10,979 |
| Mean Q-gap | 2.93 |
| Median Q-gap | **0.00** |
| % Forced (1 legal action) | 38.9% |
| % Multi-action positions | 61.1% |
| Mean n_actions (when multi) | 2.65 |

### Decision Sparsity by Threshold

![Decision Sparsity](../results/figures/09i_decision_sparsity.png)

| Q-gap Threshold | % of Positions |
|-----------------|----------------|
| > 0 | 21.8% |
| > 1 | 21.7% |
| > 2 | 16.8% |
| > 5 | 16.3% |
| > 10 | **10.7%** |
| > 20 | 4.4% |

### Mistake Impact

![Mistake Impact](../results/figures/09i_mistake_impact.png)

| Metric | Value |
|--------|-------|
| Mean V drop for 2nd-best | **4.8 points** |
| Median V drop | 0.0 |
| High-stakes decisions (Q-gap > 10) | 10.7% |
| Critical decisions per game | ~3 |

### Interpretation

**Key findings:**

1. **Moderate decision density**: 61.1% of positions have multiple legal actions (contrast with 09g's 80.9% forced along PV — the difference is that PV traverses constrained paths).

2. **Median Q-gap is 0**: Even when multiple actions are legal, the second-best is often equivalent to the best. Only 21.8% of positions have Q-gap > 0.

3. **Mistakes are moderately costly**: Mean 4.8 point drop for taking second-best. This is significant (~10% of typical hand value).

4. **~3 critical decisions per game**: Only 10.7% of positions have Q-gap > 10. In a 28-move game, expect ~3 truly high-stakes choices.

**Reconciliation with 09g:**
- 09g found 80.9% forced moves *along the principal variation*
- 09i found 38.9% forced *across all sampled states*
- The difference: optimal play traverses constrained subgraph where forced moves dominate

**Implication for ML:**
- Focus model capacity on the ~22% of positions with non-zero Q-gap
- The ~3 critical decisions per game are where games are won/lost
- A policy that gets high-stakes decisions right can tolerate errors elsewhere

---

## 9.10 Synthesis

*To be completed after all sub-analyses*

---

*End of Section 09*
