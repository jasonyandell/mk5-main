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

## 8. Imperfect Information Analysis (Section 11)

Using marginalized oracle data (201 base seeds × 3 opponent configurations), we quantified the impact of hidden information on game outcomes.

### Outcome Variance Decomposition

| Component | % of Total Variance | Interpretation |
|-----------|---------------------|----------------|
| Between-hand | **47%** | Which hand you were dealt |
| Within-hand | **53%** | Which cards opponents hold |
| Predictable hand effect | **12%** | Explainable by features (doubles, trumps) |
| Pure opponent effect | **49%** | Irreducible opponent distribution variance |

**Important**: This is NOT a "skill vs luck" measurement. Both components are determined by the random deal, not player decisions. The oracle plays perfectly - no human skill is measured here.

**What this tells us**: Even with perfect play, 53% of outcome variance comes from opponent card distribution. This is irreducible through better play - it's baked into the deal.

### The Napkin Bidding Formula

From hand features regression (R² = 0.25, CV R² = 0.18):

```
E[V] ≈ -4.1 + 6.4×(doubles) + 3.2×(trump_count) + 2.2×(trump_double) - 1.2×(6_highs)
```

**Key predictors (by |correlation|)**:
1. **n_doubles**: +0.40 (strongest predictor)
2. **has_trump_double**: +0.24
3. **trump_count**: +0.23
4. **count_points**: +0.20
5. **n_6_high**: -0.16 (6-highs are liabilities!)

### Count Lock Predictability

From count locks analysis (R² = 0.46, CV R² = 0.37):

| Count | Lock Rate | Holding→Lock Correlation |
|-------|-----------|-------------------------|
| 5-5 | **48%** | +0.79 |
| 3-2 | 44% | +0.51 |
| 4-1 | 34% | +0.68 |
| 6-4 | 30% | +0.81 |
| 5-0 | 25% | **+0.81** |

**Key insight**: count_points is the dominant predictor (+0.607). Total pips is irrelevant (+0.01).

### Information Value is Surprisingly Low

| Metric | Value |
|--------|-------|
| Mean info gain from perfect knowledge | **0.54 points** |
| Moves that agree with/without perfect info | **74%** |
| Positions benefiting from perfect info | 27% |

**Implication**: "Play the board, not the player." Opponent inference adds <1 point of expected value on average.

### Partner Inference Potential

| Metric | Value |
|--------|-------|
| Action consistency rate | **80.1%** |
| Actions revealing hand info | **20%** |

**Implication**: Partner actions are mostly determined by game state, not hand. Moderate inference potential - 20% of actions vary with partner's hand.

### Best Move Robustness

| Analysis | Finding |
|----------|---------|
| Overall consistency (11c) | 54.5% same best move across configs |
| Common state robustness (11o) | **97%** robust (same best move) |
| Endgame (depth 0-4) | **100%** deterministic |
| Early game (depth 17+) | **10%** consistency |

**Interpretation**: Early game is chaos (10% consistency), but most game states (97% of common positions) have clear optimal moves regardless of opponent hands.

### Risk vs Return (11s)

| Metric | Value |
|--------|-------|
| E[V] vs σ(V) correlation | **-0.381** |
| Hand features → σ(V) R² | **0.081** |

**Critical Finding**: Good hands are also safer hands (negative correlation). This is the opposite of typical financial markets. Risk is fundamentally unpredictable from hand features (R² = 0.08 means 92% unexplained).

### Hand Classification

Hands naturally cluster into three types:

| Type | % | E[V] | σ(V) | Recommendation |
|------|---|------|------|----------------|
| **STRONG** | 18% | +33.7 | 4.4 | Bid confidently |
| **VOLATILE** | 40% | +16.9 | 11.9 | Cautious |
| **WEAK** | 42% | +2.7 | 22.7 | Pass |

**The 18/40/42 rule**: Only ~18% of hands justify confident bidding.

---

## 9. Statistical Rigor (Section 12-13)

Scaled analyses to n=201 seeds with rigorous statistical testing.

### E[V] vs σ(V) Correlation Confirmed

| Metric | Value | 95% CI |
|--------|-------|--------|
| r(E[V], σ[V]) | **-0.381** | [-0.494, -0.256] |
| p-value | 2.6×10⁻⁸ | |
| Effect size | Medium | |

**The inverse risk-return relationship is real** - good hands are also safer hands.

### The Napkin Formula (Bootstrap Validated)

Only **two features survive multivariate analysis**:

| Feature | Coefficient | 95% CI | Significant? |
|---------|-------------|--------|--------------|
| n_doubles | **+5.7** | [+2.3, +9.2] | **Yes** |
| trump_count | **+3.2** | [+1.3, +4.7] | **Yes** |
| All others | varies | includes 0 | No |

**Cross-validation confirms**: Napkin model (2 features) has CV R² = 0.15, outperforming the full 10-feature model.

### Power Analysis

All key findings have >80% power at n=200:
- Main effects (r ≈ 0.4): Power ≈ 1.00
- Group comparisons (d ≈ 0.76): Power ≈ 1.00
- Risk model (R² = 0.08): Power = 0.81 (borderline)

---

## 10. Explainability (Section 14)

SHAP analysis confirms regression findings.

### Feature Importance (Mean |SHAP|)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | n_doubles | **4.84** |
| 2 | trump_count | **4.39** |
| 3-10 | Others | < 2.2 |

**Key insight**: Main effects account for 68% of prediction; interactions are small. The napkin formula is justified.

### Risk Model Fails Cross-Validation

| Model | Train R² | CV R² |
|-------|----------|-------|
| E[V] (napkin) | 0.23 | **+0.15** |
| σ(V) | 0.08 | **-0.13** |

**Risk is fundamentally unpredictable** from hand features.

---

## 11. Core Visualizations (Section 15)

### Risk-Return Scatter
- **r = -0.38**: Inverse relationship confirmed
- Good hands (high E[V]) cluster in low-σ region
- No hands have both high E[V] AND high risk

### UMAP Hand Space
- **No natural archetypes**: Hand space is continuous
- Gradual transitions between good and bad hands
- Features show gradients, not clusters

### Phase Transition
| Phase | Depth | Consistency |
|-------|-------|-------------|
| Opening | 24-28 | 40% |
| Mid-game | 5-23 | 22% (chaos) |
| End-game | 0-4 | **100%** (deterministic) |

---

## 12. Advanced Analysis (Sections 16-23)

### Embeddings (Section 16)
- Word2Vec on hand composition shows **weak structure**
- Doubles cluster slightly (sim = 0.079 vs 0.069 random)
- No strong strategic archetypes from co-occurrence

### Differential Analysis (Section 17)
Only 2 dominoes survive FDR correction:
- **5-5**: 2.8× enriched in winners (best domino)
- **6-0**: 3× enriched in losers (worst domino)

### Clustering (Section 18)
- Optimal k=2 clusters (silhouette = 0.19)
- **Strong Balanced** (17%): High doubles/trumps, E[V]=22.7
- **Average** (83%): Modal hand type, E[V]=12.1

### Bayesian Modeling (Section 19)
- PyMC confirms frequentist findings
- LOO-CV favors napkin model (67.5% weight)
- Hierarchical model shows doubles worth +8 pts in control hands

### Time Series (Section 20)
- V trajectories are predictive of outcome
- MiniRocket achieves **93% accuracy by trick 3**
- Three phases: Deterministic → Chaotic → Resolution

### Survival Analysis (Section 21)
- Control hands reach "decided" by trick 4-5
- Volatile hands stay uncertain until trick 6-7
- Decision time correlates with n_doubles

### Ecological Analysis (Section 22)
- **Diversity hurts E[V]**: r = -0.21
- Specialists (concentrated in doubles) beat generalists
- Double-double pairs (4-4 + 5-5) appear in 8 winners, 0 losers

### Phase Diagram (Section 23)
Additive structure confirmed across (doubles, trumps) grid:
- +1 double → **+6.7 E[V]**
- +1 trump → **+3.0 E[V]**
- Ratio: 2.2:1 (doubles twice as valuable)

---

## 13. Strategic Analysis (Section 25)

### Mistake Cost by Phase
| Phase | Mean Cost | Forced Plays |
|-------|-----------|--------------|
| Early (20-28) | **4.9 pts** | 69% |
| Mid (8-19) | 2.7 pts | 75% |
| Late (0-7) | 1.0 pts | 92% |

**Focus on tricks 3-4**: Peak mistake cost occurs here.

### Endgame is 100% Deterministic
At depth ≤ 4, **every position has exactly one optimal action**. Q-spread = 0 everywhere.

### Heuristic Accuracy
| Heuristic | Accuracy |
|-----------|----------|
| Lead any double | **34.2%** |
| Play random | 19.3% |
| Follow with lowest | 17.7% |

**No heuristic beats 35%**: Context is king.

### Variance Decomposition
| Component | % of Total |
|-----------|------------|
| Your hand | 23% |
| Opponent hands | **77%** |

**Opponent configuration matters more than your own hand!**

### Partner Synergy
- P0×P2 interaction: **Not significant** (p = 0.60)
- Your doubles' value is independent of partner's doubles
- Team strength is additive, not multiplicative

---

## Summary of Key Findings

### The Napkin Formula
```
E[V] ≈ 14 + 6×(n_doubles) + 3×(trump_count)
```

### Key Numbers to Remember
| Finding | Value |
|---------|-------|
| E[V]-σ(V) correlation | **-0.38** (inverse risk-return) |
| Doubles per point | **+6 pts** each |
| Trumps per point | **+3 pts** each |
| Risk R² | **0.08** (unpredictable) |
| Endgame determinism | **100%** at depth ≤ 4 |
| Best heuristic accuracy | **34%** |
| Opponent variance share | **77%** |

---

## Report Structure

- **Section 01**: Baseline distributions (V, Q, state counts by depth)
- **Section 02**: Information-theoretic analysis (entropy, compression, mutual information)
- **Section 03**: Count domino analysis (the 76% R² finding in detail)
- **Section 04**: Symmetry analysis (why algebraic structure doesn't help)
- **Section 05**: Topological analysis (level sets, Reeb graphs)
- **Section 06**: Scaling analysis (state counts, temporal correlations, DFA)
- **Section 07**: Synthesis and open questions
- **Section 08**: Deep count capture analysis (lock-in depth, residual decomposition)
- **Section 11**: Imperfect information analysis (variance decomposition, bidding formulas)
- **Section 12**: Validate & Scale (n=201 seeds, unified features)
- **Section 13**: Statistical Rigor (bootstrap CIs, effect sizes, power analysis, cross-validation)
- **Section 14**: Explainability (SHAP analysis)
- **Section 15**: Core Visualizations (risk-return, UMAP, phase transition)
- **Section 16**: Embeddings & Networks (Word2Vec, interaction matrix)
- **Section 17**: Differential Analysis (winner/loser enrichment, volcano plots)
- **Section 18**: Clustering & Archetypes (K-means, silhouette)
- **Section 19**: Bayesian Modeling (PyMC, LOO-CV, hierarchical by archetype)
- **Section 20**: Time Series (V trajectory, MiniRocket classification, phase segmentation)
- **Section 21**: Survival Analysis (decision time, archetype survival curves)
- **Section 22**: Ecological Analysis (alpha diversity, co-occurrence matrix)
- **Section 23**: Phase Diagram (doubles-trumps grid, contour plots)
- **Section 24**: Writing (publication figures)
- **Section 25**: Strategic Analysis (mistake costs, bid optimization, heuristic derivation)

Each section includes methodology, complete results, and interpretation. Figures are in `results/figures/`.
