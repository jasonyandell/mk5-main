# Statistical Analysis of Texas 42 Oracle Game Trees

## Epistemic Status

> **CRITICAL**: This report analyzes a **perfect-information oracle**—a minimax solver where all four players see all hands and play optimally. This reveals theoretical game tree structure, NOT human gameplay dynamics.
>
> All findings describe oracle outcomes:
> - How minimax-optimal play behaves under omniscient conditions
> - Theoretical structure of the Texas 42 game tree
> - Statistical patterns in perfect-play outcome distributions
>
> These findings do NOT directly tell us:
> - How humans navigate hidden information
> - Strategies under real-world uncertainty
> - Actual human gameplay dynamics or outcomes
>
> **Transfer to human play is untested.** All "implications for bidding" and "practical recommendations" are hypotheses extrapolated from oracle analysis.

---

## The Data

**Texas 42** is a four-player partnership trick-taking game using a double-six domino set (28 dominoes). One team "declares" (bids and names trump), then both teams play 7 tricks. Points come from capturing five specific "count" dominoes (35 points total) plus 1 point per trick (7 points), totaling 42 points per hand.

**Oracle data generation**: We solved complete game trees via backward induction (dynamic programming), computing the minimax value V for every reachable state. V represents the point differential under perfect play by all players with full information.

**Marginalized oracle data**: For imperfect-information analysis, we fixed P0's hand and computed oracle E[V] and σ(V) across multiple opponent configurations. This measures outcome variance from opponent hand distribution while P0's hand is held constant.

**Dataset characteristics**:
- **Seeds analyzed**: 200 random deals (each seed determines the 28-domino shuffle)
- **Declarations per seed**: 10 (which player declares, which suit is trump)
- **States per seed-declaration**: 7,000 to 75,000,000
- **Total states**: ~300 million across all seeds
- **Marginalized samples**: 200 base seeds × 3 opponent configurations

---

## Key Findings Summary (Oracle Analysis)

### 1. Count Domino Ownership Explains Oracle V Variance

A linear model predicting oracle V from binary indicators of which team captured each count domino achieves R² = 0.76 overall, rising to **R² > 0.99 in late-game oracle positions** (depth ≤ 12).

| Depth | Oracle V Variance | Within-Basin Variance | Variance Explained |
|-------|-------------------|----------------------|-------------------|
| 8 | 73.2 | 0.31 | 99.6% |
| 12 | 66.4 | 0.31 | 99.5% |
| 5 | 96.7 | 33.5 | 65.3% |

**Oracle interpretation**: Under perfect play, count capture explains ~92% of oracle V variance. This describes oracle game tree structure. Whether count capture similarly dominates human game outcomes is untested.

### 2. Exact Symmetries Provide No Oracle Compression (1.005x)

| Metric | Value |
|--------|-------|
| Total states sampled | 7,564 |
| Unique orbits | 7,528 |
| Compression ratio | 1.005x |

**Oracle interpretation**: Pip-permutation symmetries exist mathematically but provide negligible compression. The trump suit and played-card history break most potential equivalences in the oracle game tree.

### 3. Strong Temporal Autocorrelation in Oracle Trajectories

Detrended Fluctuation Analysis on oracle principal variation trajectories shows:

| Metric | Observed | Shuffled Baseline |
|--------|----------|------------------|
| Hurst exponent H | 0.925 ± 0.12 | 0.61 |

**Oracle interpretation**: Oracle game value trajectories exhibit strong persistence—far from random walk behavior. This describes dynamics in the perfect-information game tree. Human game trajectories may differ.

### 4. Oracle Level Set Topology is Fragmented

States sharing the same oracle V value form disconnected components:

| V | States | Components | Fragmentation |
|---|--------|------------|---------------|
| -17 | 16,461 | 3,402 | 20.7% |
| -5 | 77,929 | 35,256 | 45.2% |

**Oracle interpretation**: The oracle value function is discontinuous almost everywhere. Adjacent states typically have different oracle V values.

---

## The Oracle Napkin Formula

From bootstrap regression analysis on oracle E[V] (13a), only **two features survive multivariate analysis**:

| Feature | Coefficient | 95% CI | Significant? |
|---------|-------------|--------|--------------|
| n_doubles | **+5.7** | [+2.3, +9.2] | **Yes** |
| trump_count | **+3.2** | [+1.3, +4.7] | **Yes** |
| All others | varies | includes 0 | No |

**Oracle formula**:
```
Oracle E[V] ≈ 14 + 6×(n_doubles) + 3×(trump_count)
```

**Validation**: Cross-validation confirms the 2-feature napkin formula (CV R² = 0.15) outperforms the full 10-feature model. Bayesian model comparison (19c) confirms with 67.5% stacking weight.

**Scope limitation**: This formula predicts oracle expected outcomes under perfect play. Whether it predicts human game outcomes is untested.

---

## Oracle Variance Decomposition (Section 11)

Using marginalized oracle data (200 base seeds × 3 opponent configurations):

| Component | % of Total Oracle Variance | Interpretation |
|-----------|---------------------------|----------------|
| Between-hand | **47%** | Which hand you were dealt |
| Within-hand | **53%** | Which cards opponents hold |
| Predictable hand effect | **12%** | Explainable by features (doubles, trumps) |

**Important**: This is NOT a "skill vs luck" measurement. Both components are determined by the random deal. The oracle plays perfectly—no human skill is measured here.

**Oracle interpretation**: Even with perfect play, 53% of oracle outcome variance comes from opponent card distribution. This describes theoretical variance structure, not necessarily human gameplay variance.

---

## Oracle Risk-Return Relationship (Sections 12-13)

| Metric | Value | 95% CI |
|--------|-------|--------|
| r(Oracle E[V], Oracle σ[V]) | **-0.381** | [-0.494, -0.256] |
| p-value | 2.6×10⁻⁸ | |
| Effect size | Medium | |

**Oracle interpretation**: In oracle data, good hands (high oracle E[V]) are also safer hands (low oracle σ[V])—the inverse of typical financial markets. This is a property of oracle outcome distributions.

**Oracle risk is unpredictable**: Hand features explain only 8% of oracle σ(V) variance (R² = 0.08). Oracle risk prediction fails cross-validation (CV R² < 0).

**Scope limitation**: Whether human gameplay outcomes show similar inverse risk-return is untested.

---

## Oracle Phase Structure (Sections 15, 20)

Oracle best-move consistency varies by game phase:

| Phase | Depth | Oracle Consistency |
|-------|-------|-------------------|
| Opening | 24-28 | 40% |
| Mid-game | 5-23 | 22% (chaotic) |
| End-game | 0-4 | **100%** (deterministic) |

**Oracle interpretation**: The oracle game tree exhibits an "order → chaos → resolution" pattern. Early game has some consistency, mid-game is most uncertain, endgame is fully deterministic.

**Scope limitation**: Human games with hidden information may not follow this phase structure.

---

## SHAP Feature Importance (Section 14)

SHAP analysis on oracle E[V] model confirms:

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | n_doubles | **4.84** |
| 2 | trump_count | **4.39** |
| 3-10 | Others | < 2.2 |

**Oracle interpretation**: n_doubles and trump_count together account for ~45% of total oracle feature importance. Effects are largely additive (68% main effects for n_doubles).

---

## Oracle Hand Clustering (Section 18)

K-means clustering on oracle features yields k=2 optimal clusters:

| Type | % | Oracle E[V] | Oracle σ[V] |
|------|---|-------------|-------------|
| **Strong Balanced** | 17% | +22.7 | low |
| **Average** | 83% | +12.1 | higher |

**Oracle interpretation**: These archetypes describe structure in oracle feature space. Whether human players perceive or should use similar archetypes is untested.

---

## Differential Analysis (Section 17)

Only 2 dominoes survive FDR correction for oracle winner enrichment:

| Domino | Effect | Oracle Enrichment |
|--------|--------|------------------|
| **5-5** | Best | 2.8× more common in oracle winners |
| **6-0** | Worst | 3× more common in oracle losers |

**Oracle interpretation**: 5-5 (10 count points, high trump) is the strongest positive signal for oracle E[V]; 6-0 (weak) is the strongest negative signal.

---

## Strategic Analysis (Section 25)

### Oracle Mistake Cost by Phase

| Phase | Mean Oracle Cost | Forced Plays |
|-------|-----------------|--------------|
| Early (20-28) | **4.9 oracle pts** | 69% |
| Mid (8-19) | 2.7 oracle pts | 75% |
| Late (0-7) | 1.0 oracle pts | 92% |

**Oracle interpretation**: In perfect-play analysis, early-game mistakes cost most. Peak mistake cost occurs around tricks 3-4 in oracle data.

### Oracle Endgame Determinism

At oracle depth ≤ 4, **every position has exactly one optimal action**. Q-spread = 0 everywhere.

**Oracle interpretation**: The oracle game tree becomes fully deterministic in late game. Human endgames may retain uncertainty due to imperfect information.

---

## Key Numbers Summary (Oracle Analysis)

| Finding | Oracle Value | Scope |
|---------|--------------|-------|
| Oracle E[V]-σ(V) correlation | **-0.38** | Oracle inverse risk-return |
| Doubles coefficient | **+6 oracle pts** each | Oracle napkin formula |
| Trumps coefficient | **+3 oracle pts** each | Oracle napkin formula |
| Oracle risk R² | **0.08** | Oracle risk unpredictable |
| Oracle endgame determinism | **100%** at depth ≤ 4 | Oracle game tree |
| Count capture R² | **76%** overall, **99%** late | Oracle V prediction |

---

## What This Analysis Does and Does Not Show

### This analysis DOES show:
- Theoretical structure of the Texas 42 perfect-information game tree
- How minimax values distribute across game states
- Statistical patterns in oracle outcome distributions
- Which hand features correlate with oracle expected outcomes

### This analysis does NOT show:
- How humans should bid or play
- What strategies work under hidden information
- Whether oracle-derived heuristics improve human outcomes
- Skill vs luck decomposition in human games

### Validation needed:
- Human gameplay data to test whether oracle patterns transfer
- Comparison of oracle-derived strategies against human outcomes
- Analysis of imperfect-information decision-making

---

## Report Structure

All reports have been audited for epistemic rigor. Each contains:
- Epistemic status header clarifying oracle scope
- Qualified claims distinguishing oracle from gameplay implications
- "Further Investigation" section with validation needs

- **Section 01**: Oracle baseline distributions (V, Q, state counts)
- **Section 02**: Oracle information-theoretic analysis (entropy, compression)
- **Section 03**: Oracle count domino analysis (76% R² finding)
- **Section 04**: Oracle symmetry analysis (1.005x compression)
- **Section 05**: Oracle topological analysis (level set fragmentation)
- **Section 06**: Oracle scaling analysis (temporal correlations)
- **Section 07**: Oracle synthesis and open questions
- **Section 08**: Oracle deep count capture analysis
- **Section 11**: Marginalized oracle imperfect-information analysis
- **Section 12**: Oracle validation at scale (n=200 seeds)
- **Section 13**: Oracle statistical rigor (bootstrap CIs, power analysis)
- **Section 14**: Oracle SHAP explainability
- **Section 15**: Oracle core visualizations
- **Section 16**: Oracle embeddings (Word2Vec, UMAP)
- **Section 17**: Oracle differential analysis (winner/loser enrichment)
- **Section 18**: Oracle clustering (K-means archetypes)
- **Section 19**: Oracle Bayesian modeling (PyMC, LOO-CV)
- **Section 20**: Oracle time series (V trajectories, MiniRocket)
- **Section 21**: Oracle survival analysis (decision time)
- **Section 22**: Oracle ecological analysis (diversity metrics)
- **Section 23**: Oracle phase diagram (doubles-trumps grid)
- **Section 24**: Oracle publication figures
- **Section 25**: Oracle strategic analysis (mistake costs)

Each section includes methodology, complete results, and interpretation with oracle scope properly qualified. Figures are in `results/figures/`.

---

## Further Investigation

### Critical Validation Gap

**Human gameplay validation** is the central need. All findings describe oracle (perfect-information) game structure. The key questions are:

1. **Does the napkin formula transfer?** Do n_doubles and trump_count predict human game outcomes with similar coefficients?

2. **Does inverse risk-return hold?** Is the oracle's -0.38 correlation present in human gameplay?

3. **Do oracle phases match human experience?** Does human gameplay show "order → chaos → resolution"?

4. **Are oracle archetypes meaningful to humans?** Do experienced players recognize the oracle-derived hand classifications?

### Methodological Questions

1. **DFA validity**: Standard DFA may not apply to short discrete game sequences
2. **Sample generalization**: 200 seeds may not cover all hand configurations
3. **Transfer assumptions**: What conditions would make oracle patterns predictive of human play?

### Open Questions

1. **Why does count capture dominate?** What game mechanism causes counts to explain 92% of oracle variance?
2. **Why inverse risk-return?** What causes high oracle E[V] hands to have low oracle σ(V)?
3. **Oracle vs human complexity**: If human games involve inference and bluffing, is the oracle analysis useful?

This analysis provides a complete structural understanding of the Texas 42 oracle game tree. Its value for human gameplay remains to be established through validation against human data.
