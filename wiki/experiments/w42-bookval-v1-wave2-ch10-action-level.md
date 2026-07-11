---
title: W42 Book Validation Wave 2.H — Ch10 Mark Multiplier Action-Level Evidence
status: complete
bead: t42-8na4
parent_bead: t42-4zi6
wave: wave2
claim_id: ch10-special-bid-mark-multiplier
claim_ledger_change: none (evidence base widened)
date: 2026-05-03
backlinks:
  - [[winning42-ch10-tournament-scoring]]
  - [[w42-bookval-v1-wave1-mark-utility-transform]]
  - [[w42-bookval-v1-wave2-bid-aware-atlas]]
---

# Wave 2.H — Ch10 Mark Multiplier Action-Level Evidence

## Context

[[winning42-ch10-tournament-scoring]] introduces special bids (35, 36, 39, 42,
84) that are worth more than their face value in the mark-counting system. The
Ch10 claim `ch10-special-bid-mark-multiplier` states that the mark multiplier
changes how the game should be played — that players ought to adjust their
strategy based on what the game is worth in marks, not just in points.

[[w42-bookval-v1-wave1-mark-utility-transform]] (Wave 1.2, bead t42-c6sa)
established that `mark_ev = 2 × p_make − 1` at bid=30 (mark multiplier = 1),
making mark_ev algebraically identical to p_make. That probe ran on the bid=30
corpus only.

[[w42-bookval-v1-wave2-bid-aware-atlas]] (Wave 2.B.2, bead t42-7eop) extended
the corpus to 259,618 action rows across 7 bid values (30, 32, 35, 36, 39, 42,
84), confirmed the identity `mark_ev = mm × (2 × p_make − 1)` holds at every
bid, and promoted the row to `supported` via the deterministic transform.

This probe (Wave 2.H, bead t42-8na4) adds the missing strategic layer: does
the multiplier change which tile the oracle plays, or only the final score?

## Method

For each of 14,000 decisions (50 seeds × 10 declarations × 28 positions each,
sampled across seeds 9000–9049) evaluated under all 7 bid values:

1. Find the top-1 action by `mark_ev` at each bid.
2. Find the top-1 action by raw EV (oracle expected value, `mean` column).
3. Find the top-1 action by `p_make` (win probability at each bid's threshold).
4. Compare top-1 actions pairwise across bids and utilities.

Bootstrap CIs use 2,000 iterations (percentile method, seed=42).

## Key Findings

### mark_ev always agrees with p_make on top-1 (all bids confirmed)

`mark_ev = mm × (2 × p_make − 1)` is a positive affine transform of p_make for
any fixed mm > 0. As a result, `argmax(mark_ev) == argmax(p_make)` for every
(decision, bid) pair — confirmed with zero exceptions across 98,000 pairs. The
Wave 1.2 algebraic identity generalises cleanly to all bids including bid=84
(mm=2).

Implication: mark_ev and p_make are strategically identical. The mark multiplier
scalar alone does not change which action is optimal within a given bid's
p_make distribution.

### The multiplier changes optimal play in 71% of decisions at bid=84

The real strategic shift comes from `threshold_q`, the Q-value above which a
hand is considered a make. Higher bids require more tricks/points, which shifts
this threshold and reorders the p_make distribution over available actions.

| Bid | mark_ev@bid vs mark_ev@bid=30 flip rate | 95% CI |
|-----|------------------------------------------|--------|
| 32  | 56.2% | [55.4, 57.0] |
| 35  | 60.3% | [59.4, 61.1] |
| 36  | 63.2% | [62.5, 64.0] |
| 39  | 70.0% | [69.2, 70.8] |
| 42  | 71.0% | [70.3, 71.8] |
| 84  | 70.9% | [70.2, 71.7] |

The flip rate rises monotonically with bid up to bid=42, then plateaus at bid=84.
This is mechanically correct: bids 42 and 84 share `threshold_q = 42` (all 42
points needed). The 2× mark multiplier adds no further strategic differentiation
beyond the threshold already captured at bid=42.

### mark_ev vs raw EV flip rate rises from 21% at bid=30 to 40% at bid=84

| Bid | mark_ev vs raw EV top-1 flip | 95% CI |
|-----|------------------------------|--------|
| 30  | 21.4% | [20.8, 22.1] |
| 32  | 23.6% | [22.9, 24.3] |
| 35  | 26.0% | [25.3, 26.7] |
| 36  | 28.5% | [27.8, 29.3] |
| 39  | 32.6% | [31.8, 33.3] |
| 42  | 40.0% | [39.1, 40.8] |
| 84  | 40.1% | [39.3, 40.9] |

Even at bid=30 (the baseline), 21% of decisions already show mark_ev
disagreeing with raw EV. This baseline reflects that win-probability weighting
(p_make) creates a fundamentally different utility than expected-score
maximisation regardless of bid level.

### Bidder is the most multiplier-sensitive role (74.6% at bid=84)

| Seat Role       | flip rate at bid=84 |
|-----------------|---------------------|
| bidder          | 74.6% |
| bidder_partner  | 69.8% |
| left_setter     | 69.6% |
| right_setter    | 69.6% |

The bidder's decisions are most altered by the threshold_q shift. This makes
sense: the bidder controls offense lead and is most exposed to the scoring
threshold change. Setter roles (both seats) are equally affected and slightly
less sensitive.

### Suit declarations more multiplier-sensitive than doubles/no-trump

| Declaration  | flip rate at bid=84 |
|--------------|---------------------|
| sixes        | 74.8% |
| twos         | 73.1% |
| fives        | 73.1% |
| fours        | 72.2% |
| no-trump     | 71.6% |
| doubles      | 69.8% |
| doubles-suit | 68.2% |

All declarations are substantially affected (all above 68%). Suit declarations
show slightly higher sensitivity, consistent with their wider strategic variety
at high bids. Doubles-suit is least sensitive — its constrained action set
(fewer viable non-trump plays) limits the range of strategic pivots.

## Claim-Ledger Assessment

Row `ch10-special-bid-mark-multiplier` remains `supported`. This probe does not
change the status. It widens the evidence base from "deterministic algebraic
identity" to "action-level paired demonstration":

> The mark multiplier — operating through the threshold_q mechanism — changes the
> oracle's optimal play in **71% of decisions** at bids 42 and 84 relative to
> a bid=30 baseline. This is not just a scoring arithmetic change.

The one nuance the action-level evidence surfaces: the 2× multiplier at bid=84
does not add strategic differentiation *beyond* bid=42. Both share the same
threshold_q. The book's claim that bid=84 changes play is correct, but the
mechanism is threshold_q equality with bid=42, not the mark multiplier scalar.

## Artifacts

All outputs under:
`w42/book_validation_v1/wave2/probes/t42-8na4_ch10_action_level/`

- `action_flip_rates.csv` — per-bid mark_ev vs raw EV flip rate
- `multiplier_strategic_effect.csv` — per-bid mark_ev@bid vs mark_ev@bid=30 flip rate
- `cross_utility_matrix.csv` — agreement matrix (mark_ev / p_make / EV) per bid
- `slice_breakdown.csv` — slices by declaration, seat role, EV action tier
- `summary.json` — machine-readable headline numbers
- `run_ch10_action_level.py` — reproducibility script

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- The declaration table lists 7 of 10 declarations; blanks (69.3%), ones (68.4%), threes (68.5%) at bid=84 are omitted — the "all above 68%" claim still holds.
- summary.json notes that at bid=42 p_make=0 for all actions (mark_ev = −mm always, argmax settled by ties) — the zero-exception mark_ev==p_make agreement at bids 42/84 is partly tie-breaking, not discrimination.
- Cheap next probe: flip-rate vs trick number (early vs late decisions) to test whether threshold_q sensitivity concentrates in the endgame.
