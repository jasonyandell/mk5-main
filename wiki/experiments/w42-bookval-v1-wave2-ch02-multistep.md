---
title: W42 Book Validation Wave 2.G — ch02 Bid-Only-Enough Multi-Step Extension
bead: t42-ey88
epic: t42-4zi6
status: complete
date: 2026-05-03
tags: [book-validation, ch02, bidding, oracle, wave2]
backlinks:
  - [[winning42-ch02-bidding]]
  - [[w42-bookval-v1-wave2-bid-aware-atlas]]
  - [[w42-phase4-bidding-count-exposure-tests]]
---

## Summary

Wave 2.B.2 promoted `ch02-bid-only-enough` from `not-yet-tested` to `context-limited` based on a single step (bid=32 vs bid=30, N=14,000). Wave 2.G extends this to all adjacent bid steps in the 30–42 range and finds the overbidding penalty is **universal, monotone, and robust**. The claim is promoted to **`supported`** on the same-hand bid-margin slice.

## Claim Under Test

> Bid only what you need to win the hand. Overbidding raises your threshold and reduces your expected mark value. Don't bid 35 when 32 wins, don't bid 39 when 35 wins.
> — [[winning42-ch02-bidding]], paraphrased

## Method

Same approach as [[w42-bookval-v1-wave2-bid-aware-atlas]], restricted to `is_actual_action==1`. Paired contrasts on `(seed, decl_id, decision_idx)` so the same hand and game position is compared across bid thresholds. Delta convention: `lower_bid − higher_bid` (positive = lower bid is better for the bidding team).

Metrics: `mark_ev` (primary), `p_make`, `threshold_mass`, scalar Q mean. Bootstrap CIs: 2000 iterations, percentile method.

## Per-Step Results (mark_ev)

| Step   |    N | Delta  | 95% CI                | Cohen d | Supports book? |
|--------|-----:|-------:|----------------------:|--------:|----------------|
| 30↔32  | 10052| +0.074 | [+0.067, +0.082]     | +0.186  | YES            |
| 32↔35  |  9800| +0.115 | [+0.107, +0.122]     | +0.299  | YES            |
| 35↔36  |  9888| +0.048 | [+0.042, +0.054]     | +0.158  | YES            |
| 36↔39  |  9584| +0.103 | [+0.097, +0.110]     | +0.337  | YES            |
| 39↔42  |  8168| +0.146 | [+0.139, +0.153]     | +0.472  | YES            |

All five CIs exclude zero in the book direction. p-values are numerically zero (< machine epsilon) for all steps. The penalty is not constant: it grows as the bid approaches 42, consistent with rising threshold difficulty.

The p_make metric shows the same pattern at roughly half the magnitude (mark_ev = 2 × p_make − 1 identity holds within each bid level).

## Monotonicity

Yes. No reversal across any step. Effect size (Cohen d) grows from 0.16 at the 35↔36 boundary (smallest single-mark step) to 0.47 at 39↔42. The 35↔36 step is notably smaller than neighboring steps — this aligns with the fact that 36 and 35 differ by only a count point, making the threshold change minimal.

## Transitive Check

Does the direct 30→42 delta equal the sum of the five steps?

| Metric         | Direct 30→42 | Sum of Steps | Deviation |
|----------------|-------------:|-------------:|----------:|
| mark_ev        |       +0.482 |       +0.486 |    −0.81% |
| p_make         |       +0.241 |       +0.243 |    −0.81% |
| threshold_mass |       +0.012 |       +0.011 |    +7.7%  |

Additivity holds within measurement noise. The bid-only-enough penalty is approximately linear in bid step size — no evidence of nonlinear amplification. This simplifies instruction: every extra mark you bid costs roughly the same marginal penalty, regardless of where in the 30–42 range you are.

## Slice Analysis

All 85 slice cells (5 pairs × [10 decls + 4 seat roles + 3 phases]) show the book direction. Not a single cell reversed.

**By declaration:** The penalty is largest for "blanks" and "ones" contracts (+0.105) and smallest for "doubles" (+0.084). Doubles contracts are already highly constrained, so extra overbidding changes the Q distribution less.

**By seat role:** Bidder's partner suffers the largest penalty (+0.111 mean across steps) — the partner is most exposed to the raised threshold since they can't compensate by changing strategy.

**By phase:** Overbid penalty is largest early in the hand (+0.123) when the full game trajectory is still open. It shrinks to +0.037 in late tricks where counterfactual trajectories have mostly collapsed. Even the worst slice (39↔42, late game) has CI=[+0.0003, +0.006] — barely excludes zero but is directionally consistent.

## Bid=84 (separate analysis)

Bid=84 is an all-tricks contract with a 2× mark multiplier. It is treated as a categorically different claim and not part of the 30–42 step series.

Compared to bid=42: mark_ev delta = +1.000 (the entire 2-mark vs 1-mark gap). The scalar Q mean difference is only +0.065 (CI includes zero), meaning the underlying game value doesn't differ much — the penalty is purely contractual.

Compared to bid=30: mark_ev delta = +1.490, CI=[+1.479, +1.502]. p_make delta = +0.245. The oracle makes bid=30 roughly 25 points more often than bid=84 on the same hand, and the mark value difference is 1.5 marks on average.

Conclusion for 84: the overbid penalty is massive and qualitatively different from the 30–42 range. The 84 claim belongs in a separate chapter analysis (ch07/ch08 scope).

## Status

**Proposal: `supported`** on the same-hand bid-margin slice.

All five adjacent step pairs in [30, 32, 35, 36, 39, 42] show the overbidding penalty in mark_ev and p_make with 95% CIs excluding zero. The effect is monotone, holds in 85/85 slice cells, and the transitive check confirms near-additivity. The claim does not generalize to cross-contract comparison (choosing between bid=30/blanks vs bid=32/sixes), which is a separate analysis.

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-ey88_ch02_multistep/
├── README.md
├── manifest.json
├── summary.json
├── step_pair_deltas.csv     (20 rows: 5 pairs × 4 metrics)
├── slice_breakdown.csv      (85 rows)
├── transitive_check.csv     (3 rows)
└── run_ch02_multistep.py
```

Input: `w42/book_validation_v1/wave2/bid_aware_atlas/bid_aware_actions.csv` (read-only)

Command: `python3 w42/book_validation_v1/wave2/probes/t42-ey88_ch02_multistep/run_ch02_multistep.py`
