# 06: Scaling and Temporal Analysis

Oracle game tree scaling and temporal structure of minimax values.

> **Epistemic Status**: This report analyzes scaling properties of the oracle game tree and temporal correlations in minimax values. State counts and autocorrelation statistics are empirical. The DFA results should be interpreted cautiously (see Section 6.5 caveats). Implications for ML architecture are hypotheses.

---

## 6.1 State Count Scaling

### State Counts by Depth

| Depth | Mean | Std | Min | Max | CV |
|-------|------|-----|-----|-----|-----|
| 5 | 1.82M | 646K | 927K | 3.67M | 0.35 |
| 9 | 8.66M | 6.75M | 1.48M | 27.1M | 0.78 |
| 13 | 3.16M | 3.20M | 261K | 11.3M | 1.01 |
| 17 | 196K | 191K | 17.6K | 702K | 0.97 |
| 21 | 3,593 | 2,907 | 456 | 12K | 0.81 |
| 25 | 35 | 17 | 10 | 69 | 0.49 |
| 27 | 7 | 0 | 7 | 7 | 0.00 |

![State Count Scaling](../results/figures/06a_state_count_scaling.png)

**Observations**:
1. Peak at depth 9 (after second trick), not uniform
2. High coefficient of variation (0.35-1.01) indicates substantial seed variation
3. Deterministic endpoints: depth 27 always has 7 states, depth 28 always has 1

### Scaling Model Comparison

We fit power law (N ∝ d^α) and exponential (N ∝ e^(βd)) models:

| Model | Parameter | R² |
|-------|-----------|-----|
| Power law | α = -2.6 | 0.24 |
| Exponential | β = -0.26 | 0.24 |

![Scaling Comparison](../results/figures/06a_scaling_comparison.png)

**Conclusion**: Neither model fits well (R² = 0.24). The state count has structure not captured by simple scaling laws.

---

## 6.2 Branching Factor Analysis

Effective branching factor B(d) = N(d) / N(d+4) measures growth per trick:

| Depth | Branching Factor |
|-------|-----------------|
| 0→4 | 0.0005 |
| 4→8 | 0.0046 |
| 5→9 | **1.69** |
| 6→10 | **1.68** |
| 7→11 | **2.00** |
| 8→12 | 0.037 |
| 9→13 | **2.14** |
| 10→14 | **2.15** |
| 11→15 | **3.00** |
| 12→16 | 0.20 |

![Branching Factor](../results/figures/06a_branching_factor.png)

**Pattern**: The branching factor shows strong 4-depth periodicity:
- **Depths 5,6,9,10,13,14,17,18...** (mid-trick): B ≈ 1.7-2.6
- **Depths 8,12,16,20...** (trick boundary): B ≈ 0.04-0.68

**Interpretation**: At trick boundaries, many game paths converge (same count outcome regardless of specific cards played). Mid-trick, paths diverge.

---

## 6.3 Principal Variation Analysis

The **principal variation (PV)** is the sequence of minimax-optimal moves from any position. We extracted V along PV paths:

### PV Metadata (sample of 90 paths)

| Metric | Value |
|--------|-------|
| Mean PV length | 22.4 moves |
| Min length | 21 |
| Max length | 24 |
| Mean |start V| | 32.1 |
| End V (all) | 0 |

![PV Overview](../results/figures/06b_pv_overview.png)

All paths end at V = 0 (game terminal state), but start values vary widely.

### V Changes Along PV

![V Changes](../results/figures/06b_v_changes.png)

**Pattern**: V typically decreases along the PV (as uncertainty resolves), with occasional jumps when count dominoes are captured.

![Mean Trajectory](../results/figures/06b_mean_trajectory.png)

---

## 6.4 Temporal Correlation Analysis

### Autocorrelation Function

![Autocorrelation](../results/figures/06b_autocorrelation.png)

| Lag | Autocorrelation |
|-----|-----------------|
| 1 | 0.94 |
| 2 | 0.89 |
| 4 | 0.78 |
| 8 | 0.51 |
| 12 | 0.28 |

**Finding**: Strong positive autocorrelation persists to lag 8+. V at move n is highly correlated with V at move n-4 (previous trick).

---

## 6.5 Detrended Fluctuation Analysis (DFA)

DFA estimates the Hurst exponent H, which characterizes long-range correlations:
- H = 0.5: Random walk (no memory)
- H > 0.5: Persistent (trending)
- H < 0.5: Anti-persistent (mean-reverting)

### DFA Results

| Metric | Observed | Shuffled |
|--------|----------|----------|
| Mean α | 31.5 | 0.55 |
| Std α | 40.7 | - |
| Mean H | 0.925 | 0.61 |
| Std H | 0.12 | - |

![DFA Analysis](../results/figures/06c_dfa_analysis.png)

**Key findings**:
1. **α = 31.5 vs 0.55**: 57× higher than shuffled baseline
2. **H = 0.925**: Strong persistence (near-perfect trending)
3. **High variance (σ = 40.7)**: Heterogeneous dynamics across games

![Hurst Analysis](../results/figures/06c_hurst_analysis.png)

### Interpretation
The DFA exponent of 31.5 is unusually high. Typical time series have α ∈ [0.5, 1.5]. Our value suggests:
- Either the game has extreme long-range memory
- Or the DFA methodology needs adjustment for this discrete, finite domain

**Caution**: Standard DFA assumes continuous, stationary processes. Game trajectories are discrete and bounded. The absolute α value should be interpreted cautiously, but the comparison to shuffled baseline is meaningful.

---

## 6.6 Potential Implications (Hypotheses)

The following are speculative implications of the scaling findings. None have been experimentally validated.

### For Sequential Models (Hypothesis)
The strong autocorrelation (ρ₁ = 0.94) *suggests* that models with history access may perform better than memoryless models. **Hypothesis**: Transformer architectures with attention over game history could exploit this temporal structure. **Untested**: No direct comparison of architectures has been conducted.

### For Training (Hypothesis)
**Hypothesis**: Trajectory-based training (full game sequences) may capture structure that IID sampling misses. Curriculum learning along game trajectories could exploit the correlation structure. **Untested**: These training strategies have not been compared.

### For Game Structure (Grounded Interpretation)
The 4-depth periodicity in branching factor aligns with the trick structure of the game (4 cards per trick). Count captures at trick boundaries appear to act as "bottlenecks" where game paths converge. **Note**: This is an interpretation of the data pattern, not a formal proof of causality.

### For Oracle Compression (Hypothesis)
**Hypothesis**: Tricks may be natural units for compression. A trick-level oracle (storing outcomes per trick rather than per move) could potentially reduce storage. **Untested**: No trick-level compression has been implemented.

---

## 6.7 Questions for Statistical Review

1. **DFA validity**: Is DFA appropriate for bounded, discrete sequences of length ~24? What alternative methods exist for short time series?

2. **α interpretation**: Why is α = 31.5 so extreme? Is this a real phenomenon or a methodological artifact?

3. **Heterogeneity**: The high variance in α (40.7) suggests different games have very different dynamics. Should we stratify by game characteristics?

4. **Stationarity**: The mean V decreases along PV (non-stationary). How does this affect the correlation analysis?

5. **Causal structure**: The 4-depth periodicity matches trick structure. Can we formally test whether trick boundaries cause the observed correlation pattern?

---

## Further Investigation

### Validation Needed

1. **DFA methodology**: Consult time series literature on appropriate methods for short, bounded, discrete sequences

2. **Architecture comparison**: Experimentally compare memoryless vs history-aware models to test the temporal structure hypothesis

3. **Trick-level compression**: Implement and benchmark a trick-level oracle to test compression hypothesis

### Methodological Questions

1. **Alternative correlation methods**: Test Hurst exponent with R/S analysis, wavelets, or other short-series methods

2. **Stratified analysis**: Analyze DFA by game characteristics (seed, declaration) to understand the high variance

3. **Detrending approaches**: Test different detrending methods given the non-stationary V trajectory

### Open Questions

1. **Branching factor causality**: Can trick structure formally explain the 4-depth periodicity, or is this coincidental?

2. **Human relevance**: Do human players perceive the temporal correlation structure, or is it only apparent with perfect information?

3. **Cross-game comparison**: Do other trick-taking games show similar branching factor periodicity?

---

*Next: [07 Synthesis](07_synthesis.md)*
