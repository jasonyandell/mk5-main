# 04: Symmetry Analysis

## Context

We investigated whether permutation symmetries could compress the state space or enable data augmentation. **They can't.** This section explains a negative result that saved us from wasted effort.

## The Symmetry Hypothesis

Dominoes have mathematical symmetries. For example, if you swap all 2s and 3s throughout a position, the game value should be identical (assuming no suit is trump).

We expected this might yield 2-4x state space compression, enabling:
- Smaller training sets
- Data augmentation
- Canonical state representations

## The Surprising Result: 1.005x Compression

![Compression by Depth](../results/figures/04a_compression_by_depth.png)

| Metric | Value |
|--------|-------|
| Total states | 7,564 |
| Unique orbits | 7,528 |
| Compression | 1.005x |
| Fixed points | 99.5% |

**Nearly every state is its own orbit.** Only 36 non-trivial orbits exist—positions where permuting pips produces a different-but-equivalent state.

![Orbit Size Distribution](../results/figures/04a_orbit_size_dist.png)

## Why Symmetries Don't Help in Practice

The symmetries are mathematically valid but *practically irrelevant*:

1. **Trump breaks most symmetries** — Once a suit is trump (6s typically), only non-trump pip swaps are valid
2. **Played cards constrain positions** — The trick history eliminates symmetric configurations
3. **Natural play avoids symmetric positions** — Random deals rarely produce swappable configurations

**Example**: Swapping 2s↔3s requires:
- No 2 or 3 has been played yet
- 2s and 3s aren't trump
- The swap doesn't change any count domino ownership

These conditions rarely hold simultaneously.

## Orbit V-Consistency: Symmetries Are Correct, Just Rare

![Orbit V Analysis](../results/figures/04b_orbit_v_analysis.png)

When non-trivial orbits do exist, V is consistent within them (99.5% of the time). The symmetries are mathematically correct—they just don't occur.

![Canonical V Distribution](../results/figures/04b_canonical_v_dist.png)

## K-Means Clustering: Approximate Methods Win

Since exact symmetries failed, we tried approximate clustering:

![Clustering Quality](../results/figures/04c_clustering_quality.png)

| Method | Variance Reduction |
|--------|-------------------|
| Exact Symmetry | ~0.5% |
| K-Means (k=200) | 35.7% |

![Cluster V Analysis](../results/figures/04c_cluster_v_analysis.png)

Clustering on *features* (depth, counts, hand balance) beats *algebraic* structure by 70x.

**Model relevance**: This validates the feature-engineering approach. Our model uses learned features, not mathematical symmetries—and that's the right choice.

## What This Means for the Model

| Finding | Implication |
|---------|-------------|
| 1.005x compression | Don't bother with symmetry augmentation |
| 99.5% fixed points | Natural gameplay isn't symmetric |
| K-means beats algebra | Feature learning > mathematical structure |
| Symmetries correct but rare | Not a modeling failure, just irrelevant |

**Bottom line**: We were right not to invest in symmetry-based approaches. The 97.8% accuracy came from count features and attention, not group theory.

---

*Next: [05 Topology Analysis](05_topology.md)*
