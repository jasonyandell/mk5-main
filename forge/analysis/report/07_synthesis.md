# 07: Synthesis and Conclusions

## What We Learned

This analysis examined the structure of Texas 42's game tree to understand why our 97.8% accurate model works and what remains to improve.

## The Core Insight: Texas 42 is Count Poker

The single most important finding: **five count dominoes explain 76% of game value variance**.

| Component | Variance Explained |
|-----------|-------------------|
| Count domino ownership | 76% |
| Trick dynamics (attention) | ~20% |
| Other factors | ~4% |

Texas 42 looks like a complex trick-taking game but is fundamentally about count capture. Our model succeeds because:
1. `count_value` is an explicit feature (captures the 76%)
2. Transformer attention captures trick dynamics (the 20%)
3. Rich training data handles edge cases (the 4%)

## Why Each Analysis Mattered

| Analysis | Finding | Model Validation |
|----------|---------|------------------|
| Baseline | Balanced V, many forced moves | Clean training data ✓ |
| Information | ~40% compression | Learnable structure ✓ |
| Counts | R²=0.76 | Count feature critical ✓ |
| Symmetry | 1.005x compression | Don't bother with augmentation ✓ |
| Topology | Fragmented level sets | Explains value head limits ✓ |
| Scaling | α=31.5 correlations | Transformer architecture ✓ |

## The 2.2% Error Rate

Our remaining errors (2.2% accuracy gap, 0.072 Q-gap) cluster in:

1. **Ambiguous midgame positions** — Multiple reasonable moves, similar Q-values
2. **Robustness decisions** — Where reliability (6-6 vs 2-2) matters more than expected value
3. **Rare configurations** — Unusual hands underrepresented in training data

The trump-heavy hand bug (t42-pa69) exemplifies type 2: the model can't distinguish "always wins" from "usually wins" when trained on single opponent distributions.

## Minimal Representation

Based on analysis, the minimal feature set for Texas 42:

```
Required features:
├── count_value (0/5/10)     # Explains 76% variance
├── depth                     # Game phase context
├── trick_history             # For temporal attention
└── trump_rank               # Suit strength
```

Our model includes these plus additional features (player_id, is_partner, etc.) that handle the remaining 24%.

![Representation Comparison](../results/figures/07b_representation_comparison.png)

## What's Working

| Component | Evidence |
|-----------|----------|
| Count feature encoding | 76% R² → 97.8% accuracy |
| Transformer architecture | α=31.5 temporal structure captured |
| 817K parameters | Sufficient for 10M sample complexity |
| bfloat16 training | H100-efficient, no precision loss |

## What Needs Work

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Trump-heavy hand errors | Single opponent distribution | Marginalized Q-values |
| Value head MAE 7.4 | Fragmented topology | MC simulation for bidding |
| Rare hand coverage | Training distribution | More seeds/declarations |

## Recommendations

### Keep Doing
- **Count-explicit features** — The 76% R² validates this design
- **Transformer attention** — Captures the α=31.5 temporal structure
- **Large training sets** — More data → lower Q-gap (0.11 → 0.07)

### Do Next
- **Marginalized training** — Address robustness errors (t42-pa69)
- **MC bidding** — Replace value head regression with simulation
- **Error analysis** — Characterize the remaining 2.2% mistakes

### Don't Bother
- **Symmetry augmentation** — 1.005x compression means no benefit
- **Algebraic representations** — Simple features beat group theory
- **Smooth value regression** — Topology is too fragmented

## Final Summary

![Findings Dashboard](../results/figures/07a_findings_dashboard.png)

**Texas 42's structure is dominated by count domino capture (76% R²), with remaining complexity in trick dynamics (temporal correlations α=31.5).** Our Transformer architecture is well-matched to this structure, explaining the 97.8% accuracy.

The remaining 2.2% error rate concentrates in robustness decisions where marginalized training is needed. The value head's limitations (MAE 7.4) reflect fragmented topology that Monte Carlo handles better than regression.

**The model works because the architecture matches the game's structure.** Count features capture the dominant effect; attention captures the sequential dependencies; sufficient data covers the long tail.

---

## Report Navigation

- [Executive Summary](00_executive_summary.md) — Key findings and implications
- [01 Baseline](01_baseline.md) — V distribution, Q-structure
- [02 Information](02_information.md) — Entropy, compression
- [03 Counts](03_counts.md) — The 76% R² finding
- [04 Symmetry](04_symmetry.md) — Why symmetries don't help
- [05 Topology](05_topology.md) — Level set fragmentation
- [06 Scaling](06_scaling.md) — State counts, temporal correlations
- [07 Synthesis](07_synthesis.md) — This document
