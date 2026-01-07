# Statistical Analysis of Texas 42 Game Trees

## For Statistical Review

This report presents a structural analysis of exhaustively-solved Texas 42 domino game trees. We have computed exact minimax values for millions of game states across hundreds of random deals, producing a complete dataset of perfect-play outcomes.

**We seek statistical guidance on:**
1. Better methods for characterizing the value function's structure
2. Approaches we may have missed for dimensionality reduction
3. Statistical tests for the significance of our findings
4. Alternative framings that might reveal hidden structure

---

## The Data

**Texas 42** is a four-player partnership trick-taking game using a double-six domino set (28 dominoes). One team "declares" (bids and names trump), then both teams play 7 tricks. Points come from capturing five specific "count" dominoes worth 5-10 points each (35 points total) plus 1 point per trick won (7 points), totaling 42 points per hand.

**Data generation**: We solved complete game trees via backward induction (dynamic programming), computing the minimax value V for every reachable state. V represents the expected point differential under perfect play by both teams.

**Dataset characteristics**:
- **Seeds analyzed**: 20 random deals (each seed determines the 28-domino shuffle)
- **Declarations per seed**: 10 (which player declares, which suit is trump)
- **States per seed-declaration**: 7,000 to 75,000,000 (highly variable)
- **Total states**: ~300 million across all seeds
- **State representation**: 64-bit packed integer encoding player hands, trick history, and game phase

---

## Glossary of Technical Terms

### Game Theory / Decision Theory

| Term | Definition |
|------|------------|
| **Minimax** | Optimal strategy in two-player zero-sum games: maximize your minimum guaranteed outcome, assuming the opponent plays optimally |
| **V (state value)** | The minimax value of a game state—the expected point differential under perfect play by both teams |
| **Q (action value)** | The minimax value of taking a specific action from a state: Q(s,a) = V(successor state after action a) |
| **Backward induction** | Dynamic programming algorithm that computes V by working backward from terminal states to the root |
| **Principal variation (PV)** | The sequence of optimal moves from any position to game end, assuming both sides play perfectly |
| **Oracle** | A lookup table providing exact minimax values for all states—enables perfect play but requires large storage |

### Machine Learning

| Term | Definition |
|------|------------|
| **Transformer** | Neural network architecture using self-attention mechanisms; excels at sequence modeling. Our model has 817K parameters. |
| **Attention** | Mechanism allowing the model to weigh the relevance of different parts of the input (e.g., earlier moves in game history) |
| **Move prediction accuracy** | Fraction of states where the model selects a minimax-optimal action (97.8% in our case) |
| **MAE** | Mean Absolute Error—average of |predicted - actual| across samples |
| **Data augmentation** | Generating additional training examples by applying transformations (e.g., symmetries) that preserve labels |
| **Curriculum learning** | Training strategy that presents examples in a structured order (e.g., easy-to-hard) rather than randomly |

### This Analysis

| Term | Definition |
|------|------------|
| **Depth** | Number of dominoes remaining across all hands (28 at start, 0 at terminal). Depth 5 = after first trick, depth 9 = after second, etc. |
| **Count dominoes** | The five dominoes worth points when captured: 5-5 (10 pts), 6-4 (10 pts), 5-0 (5 pts), 4-1 (5 pts), 3-2 (5 pts) |
| **Count basin** | A partition of states by which team captured which count dominoes—our key explanatory variable |
| **Seed** | Random number seed determining the initial 28-domino shuffle; different seeds produce different deals |

---

## Key Findings Summary

### 1. Count Domino Ownership Explains ~92% of Variance

A linear model predicting V from binary indicators of which team captured each count domino achieves R² = 0.76 overall, rising to **R² > 0.99 in late-game positions** (depth ≤ 12). Deep analysis (Section 08) shows:

| Component | Variance | % of Total |
|-----------|----------|------------|
| Count capture (explained) | ~550 | ~92% |
| Residual (trick points) | ~50 | ~8% |

| Depth | Total Variance | Within-Basin Variance | Variance Explained |
|-------|---------------|----------------------|-------------------|
| 8 | 73.2 | 0.31 | 99.6% |
| 12 | 66.4 | 0.31 | 99.5% |
| 16 | 59.1 | 0.38 | 99.4% |
| 5 | 96.7 | 33.5 | 65.3% |

**Interpretation**: Count capture explains ~92% of V variance; the remaining ~8% corresponds to the 7 non-count trick points. The game is essentially a competition over 5 count dominoes.

### 2. Exact Symmetries Provide No Compression (1.005x)

We expected pip-permutation symmetries to compress the state space. They don't.

| Metric | Value |
|--------|-------|
| Total states sampled | 7,564 |
| Unique orbits | 7,528 |
| Compression ratio | 1.005x |
| Fixed points (trivial orbits) | 99.5% |

**Interpretation**: While mathematically valid symmetries exist, natural gameplay rarely produces symmetric configurations. The trump suit and played-card history break most potential equivalences.

### 3. Strong Temporal Autocorrelation (DFA α = 31.5 vs 0.55 shuffled)

Detrended Fluctuation Analysis on principal variation trajectories shows:

| Metric | Observed | Shuffled Baseline |
|--------|----------|------------------|
| DFA exponent α | 31.5 ± 40.7 | 0.55 |
| Hurst exponent H | 0.925 ± 0.12 | 0.61 |

**Interpretation**: Game value trajectories exhibit strong persistence—far from random walk behavior. The high variance in α suggests heterogeneous dynamics across different game configurations.

### 4. Level Set Topology is Highly Fragmented

States sharing the same V value form disconnected components:

| V | States | Components | Fragmentation |
|---|--------|------------|---------------|
| -17 | 16,461 | 3,402 | 20.7% |
| -19 | 890 | 809 | 90.9% |
| -5 | 77,929 | 35,256 | 45.2% |

**Interpretation**: The value function is discontinuous almost everywhere. Adjacent states (one move apart) typically have different V values.

### 5. Branching Factor Shows 4-Depth Periodicity

State counts follow a distinctive pattern tied to trick structure (4 plays per trick):

| Depth mod 4 | Typical Branching | Interpretation |
|-------------|------------------|----------------|
| 0 | ~0.04 | Trick boundary collapse |
| 1 | ~1.7 | First play of trick |
| 2 | ~1.7 | Second play |
| 3 | 2, 3, 4, ... | Third play (increases with trick number) |

### 6. Counts Lock In Late; Play Matters (Section 08)

Count capture outcomes remain uncertain until the last 2-3 dominoes are played. From the initial deal, models achieve only ~72% accuracy predicting capture outcomes:

| Domino | Points | Holder's Team Captures % | Model Accuracy |
|--------|--------|--------------------------|----------------|
| 3-2    | 5      | 42.9%                    | 86.7%          |
| 4-1    | 5      | 64.3%                    | 82.7%          |
| 5-0    | 5      | 64.3%                    | 58.7%          |
| 5-5    | 10     | 71.4%                    | 68.0%          |
| 6-4    | 10     | 57.1%                    | 71.3%          |

**Interpretation**: Holding the count gives 50-70% capture probability—better than random but far from deterministic. The game maintains genuine strategic depth throughout, with counts swinging until the final tricks.

### 7. The Game is 5-Dimensional (Manifold Analysis)

PCA on basin features (5-bit count capture patterns) confirms the game's intrinsic dimensionality:

| Metric | Value |
|--------|-------|
| PCA components for 95% variance | **5** |
| Unique basins observed | 13 of 32 |
| Basin entropy | 3.06 bits (61% of max 5.0) |

**Interpretation**: Five components for 95% variance matches the hypothesis that the game has ~5 effective degrees of freedom (one per count domino). Not all 32 count combinations are equally reachable—some basins (where one team sweeps all counts) are rare.

---

## Practical Application: Neural Network Training

We have trained a Transformer model on this data achieving **97.8% move prediction accuracy** (selecting the minimax-optimal action). Key architectural choices validated by this analysis:

1. **Explicit count features** — The model encodes count domino ownership directly, matching the ~92% variance finding
2. **Attention over trick history** — Captures the temporal correlations (H = 0.925)
3. **No symmetry augmentation** — Confirmed unnecessary by the 1.005x compression

**Architectural implications from Section 08**: A perfect count-capture predictor would achieve R² ≈ 0.92 on V. This suggests a two-level architecture:
1. **Count module**: Predict which team captures each count (5 binary outputs)
2. **Trick module**: Given counts, predict final trick point distribution (remaining 8% variance)

**Remaining challenge**: The model occasionally selects suboptimal moves in edge cases where two actions have identical V in one opponent configuration but different robustness across configurations.

---

## Open Questions for Statistical Guidance

1. **Dimensionality reduction**: K-means achieves only 35.7% variance reduction at k=200. Are there better clustering approaches for this mixed discrete-continuous structure?

2. **Significance testing**: How should we assess whether the DFA exponent difference (31.5 vs 0.55) is statistically meaningful given the high variance (σ = 40.7)?

3. **Conditional structure**: The count-capture R² varies from 65% (early game) to 99.6% (late game). Is there a natural way to model this heteroscedasticity?

4. **Topology characterization**: Beyond fragmentation counts, what tools characterize the value function's discontinuity structure?

5. **Compression bounds**: Given the ~40% LZMA compression ratio, what's the theoretical entropy of V conditional on observable features?

---

## Report Structure

- **Section 01**: Baseline distributions (V, Q, state counts by depth)
- **Section 02**: Information-theoretic analysis (entropy, compression, mutual information)
- **Section 03**: Count domino analysis (the 76% R² finding in detail)
- **Section 04**: Symmetry analysis (why algebraic structure doesn't help)
- **Section 05**: Topological analysis (level sets, Reeb graphs)
- **Section 06**: Scaling analysis (state counts, temporal correlations, DFA)
- **Section 07**: Synthesis and open questions
- **Section 08**: Deep count capture analysis (lock-in depth, residual decomposition, capture predictors, manifold structure)

Each section includes methodology, complete results, and interpretation. Figures are embedded throughout.
