# 07: Synthesis and Open Questions

## Overview

This section synthesizes findings across all analyses and poses open questions for statistical guidance.

---

## 7.1 Summary of Key Findings

### The Structural Picture

| Analysis | Finding | Confidence |
|----------|---------|------------|
| Counts (03) | R² = 0.76 overall, >0.99 late-game | High |
| Symmetry (04) | 1.005× compression (negligible) | High |
| Topology (05) | Highly fragmented level sets | High |
| Temporal (06) | H = 0.925, strong autocorrelation | Medium* |

*Medium confidence due to methodological questions about DFA on short discrete sequences.

### The Core Insight
**Texas 42's complexity concentrates in count domino capture.** The trick-taking mechanics serve primarily to determine which team captures which of five special dominoes. Once count outcomes are known, V is nearly deterministic.

---

## 7.2 Interconnections Between Findings

### Why Symmetry Fails
The count dominoes use 6 of 7 pip values (0,1,2,3,4,5). This leaves almost no room for pip-permutation symmetries that preserve game value. **The count structure breaks symmetry.**

### Why Topology is Fragmented
Level sets are fragmented because V changes with count capture outcomes. States with the same V but different count histories form disconnected components. **The count structure fragments topology.**

### Why Temporal Correlations Are Strong
V evolves smoothly between count captures and jumps at captures. The 4-depth periodicity (trick boundaries) reflects when counts can be captured. **The count structure drives temporal correlations.**

### The Unifying Theme
**Count dominoes explain the structure across all analyses.**

---

## 7.3 What We Learned vs. What We Expected

| Question | Expectation | Reality |
|----------|-------------|---------|
| How much does count capture explain? | ~50% (it's 83% of points) | 76%, rising to 99%+ |
| Do symmetries help? | 2-4× compression | 1.005× (negligible) |
| Is the value function smooth? | Moderate continuity | Highly discontinuous |
| Are trajectories random? | Near-random walk | Strong persistence (H=0.925) |
| What's the best compression method? | Algebraic quotients | Feature-based (count capture) |

---

## 7.4 Practical Applications

### Neural Network Training (Achieved)
Our Transformer model achieves 97.8% move prediction accuracy. This analysis validates:
- Explicit count features (`count_value` = 0/5/10 per domino)
- Attention over game history (captures H=0.925 correlations)
- No symmetry augmentation needed

**Remaining issue**: Model occasionally fails on "robustness" decisions where two moves have equal V in one opponent configuration but different reliability across configurations.

### Simplified Oracles (Potential)
The 76% R² from 5 binary features suggests a "count-only" oracle is feasible:
- 2^5 = 32 count configurations per depth
- vs. millions of full states per depth
- ~99% accuracy in late game, ~75% overall

**Open question**: Is 75% overall accuracy useful for any application?

### Compression Strategies (Potential)
State-ordered V compresses to 8% of original size. Combined with the count structure, this suggests:
1. Store count basin membership (5 bits)
2. Within-basin lookup table (small for late game)
3. Hybrid: exact late-game, approximate early-game

**Open question**: What's the engineering trade-off curve?

---

## 7.5 Open Statistical Questions

### Methodology Questions

1. **Entropy estimation**: We treat V as discrete integers. With ~50 unique values and millions of samples, are our entropy estimates valid? What's the appropriate correction?

2. **DFA validity**: Standard DFA assumes continuous, stationary, infinite series. Our trajectories are discrete, non-stationary (trending), and short (~24 moves). What alternative methods apply?

3. **Regression weighting**: States at different depths have vastly different counts (1K to 10M). How should we weight when pooling across depths?

4. **Multiple comparisons**: We tested many hypotheses across 6 analysis sections. Should we adjust for multiple testing? Which findings need stricter validation?

### Model Questions

5. **Heteroscedasticity**: Residual variance depends strongly on depth (σ² = 33.5 at depth 5 vs 0.31 at depth 8). Should we fit depth-stratified models?

6. **Interaction effects**: Our count model assumes additivity. Is there evidence for interactions (e.g., capturing both 10-point counts)?

7. **Causal vs. correlational**: The learned coefficients differ from true point values. Is this a causal effect or confounding with trick wins?

### Structure Questions

8. **Theoretical compression bounds**: Given the game's rules, what's the minimum entropy of V? Can we derive this from first principles?

9. **Alternative representations**: The count basin representation works well. Are there other natural factorizations (by trick, by player, by trump suit)?

10. **The remaining 24%**: Count capture explains 76%. What explains the rest? Trick points (7 total)? Trump control? Something else?

---

## 7.6 What Would Help

### From a Statistics Perspective

1. **Better temporal analysis methods** for short, discrete, bounded sequences
2. **Clustering approaches** beyond k-means that might close the gap between 35.7% and 76%
3. **Formal tests** for the significance of our key findings (e.g., DFA difference)
4. **Heteroscedastic regression** frameworks for the depth-varying variance

### From a Game Theory Perspective

1. **Formal analysis** of why count dominoes break symmetry
2. **Bounds** on possible compression ratios given the rules
3. **Alternative solution concepts** beyond minimax (e.g., MaxMin, robust optimization)

### From a Machine Learning Perspective

1. **Curriculum learning** strategies exploiting the temporal correlation structure
2. **Uncertainty quantification** for the 24% unexplained variance
3. **Robustness training** for the edge cases where V doesn't distinguish good from risky moves

---

## 7.7 Conclusion

### What We Know

Texas 42 is fundamentally a count-capture game. The five count dominoes explain 76% of game value variance, rising to >99% in late-game positions. Exact symmetries provide no compression. The value function is highly discontinuous but shows strong temporal correlations along game trajectories.

### What We Built

A Transformer model achieving 97.8% move prediction accuracy, validated by this analysis. The model's architecture (count features, attention over history) aligns with the discovered structure.

### What We Seek

Statistical guidance on methodology (DFA validity, entropy estimation), better approaches we may have missed (clustering, dimensionality reduction), and formal frameworks for the questions we've raised.

---

## Report Navigation

- [00 Executive Summary](00_executive_summary.md) — Key findings and questions
- [01 Baseline](01_baseline.md) — V, Q, state count distributions
- [02 Information Theory](02_information.md) — Entropy, compression, mutual information
- [03 Count Dominoes](03_counts.md) — The 76% R² finding
- [04 Symmetry](04_symmetry.md) — Why algebraic methods fail
- [05 Topology](05_topology.md) — Level set fragmentation
- [06 Scaling](06_scaling.md) — State counts, temporal correlations
- [07 Synthesis](07_synthesis.md) — This document

---

## Appendix: Data Availability

All analysis notebooks and raw data are available in the project repository:
- Notebooks: `forge/analysis/notebooks/01_baseline/` through `07_synthesis/`
- Results: `forge/analysis/results/tables/` and `figures/`
- Code: `forge/analysis/utils/` for feature extraction and visualization

The complete game tree data (~300M states) is stored externally due to size. Processed summaries (CSVs, PNGs) are included in the report.
