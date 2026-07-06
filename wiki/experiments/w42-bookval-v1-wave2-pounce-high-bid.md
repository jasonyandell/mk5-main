---
title: "W42 Book Validation v1 — Wave 2.E.2: Pounce High-Bid Probe"
bead: t42-8kbh
parent_bead: t42-4zi6
wave: "2.E.2"
claim: ch12-setter-pounce-high-bid-off
status: contradicted
created: 2026-05-03
---

## Summary

Snapshot-level paired-contrast probe for `ch12-setter-pounce-high-bid-off` at
bids 35/36/39/42. Extends [[w42-bookval-v1-wave2-pounce-window-bid30]] (Wave
2.E, bid=30) to the high-bid regime.

**Finding**: The decline-better-for-setter pattern found at bid=30 by Wave 2.E
is **not reversed at high bids** — it is amplified. At all four bid buckets,
pouncing on exposed count from bidder's team makes the setter strictly worse
by EV (CI excludes zero, p << 0.001).

## Claim Under Test

**`ch12-setter-pounce-high-bid-off`**: At high bids (>= 35), a setter
following into a trick where bidder's team has played a count domino should
"pounce" (win the trick) to deny bidder the count points.

## Data Source

Bid-aware atlas from [[w42-bookval-v1-wave2-bid-aware-atlas]]:
- Seeds 9000-9049 × 10 declarations × bids {35, 36, 39, 42}
- 50 seeds × 10 decls × 4 bids = 2000 games (500 per bid)
- Per-bid `.pt` files with n_samples=200 per decision

## Methodology

**Part 1 — Snapshot mining**: Walk each game in the atlas `.pt` files using
corpus_replayer-style state reconstruction. For each decision:
- Setter (player 1 or 3) is following (not leading the current trick)
- Count >= 5 pts from bidder team in current trick or last completed trick
- Setter has at least one legal winning AND one legal non-winning option

**Part 2 — Paired contrast**: For each snapshot, evaluate:
- **A (pounce)**: lowest-cost winning play (prefer non-trump, lower pip-sum)
- **B (decline)**: lowest-cost non-winning play

Run `generate_eq_from_snapshots` n_samples=100 per arm, compute:
- EV delta = ev_pounce_t0 - ev_decline_t0 (Team 0 perspective)
- ev_delta_setter = -ev_delta_t0 (positive = pounce better for setter)
- p_set delta = p_set(pounce) - p_set(decline)

## Results

### Snapshots Mined

| Bid | Snapshots |
|-----|-----------|
| 35 | 275 |
| 36 | 259 |
| 39 | 290 |
| 42 | 316 |
| **Total** | **1140** |

### Headline Numbers

| Metric | Overall | bid=35 | bid=36 | bid=39 | bid=42 |
|--------|---------|--------|--------|--------|--------|
| N | 1140 | 275 | 259 | 290 | 316 |
| EV delta (setter) | -10.42 | -10.08 | -11.32 | -10.14 | -10.24 |
| CI95 lo | -11.25 | -11.76 | -13.06 | -11.83 | -11.81 |
| CI95 hi | -9.59 | -8.40 | -9.57 | -8.46 | -8.67 |
| % pounce better | 24.5% | 26.6% | 21.6% | 23.8% | 25.6% |
| p_set delta | -0.047 | -0.064 | -0.078 | -0.050 | -0.004 |
| t-stat | -24.52 | -11.75 | -12.70 | -11.81 | -12.77 |

All CIs exclude zero in the decline-better direction.

### By Phase and Context

| Slice | N | EV delta | CI95 | pounce_better |
|-------|---|---------|------|--------------|
| Phase early | 582 | -12.60 | [-13.87, -11.33] | 22.9% |
| Phase mid | 558 | -8.15 | [-9.18, -7.12] | 26.2% |
| Count 5 pts | 540 | -7.27 | [-8.35, -6.20] | 30.0% |
| Count 10 pts | 549 | -13.67 | [-14.96, -12.39] | 19.1% |
| Bidder team led | 860 | -11.93 | [-12.87, -10.99] | 19.5% |
| Setter team led | 280 | -5.78 | [-7.46, -4.11] | 39.6% |

The "setter team led" slice has the weakest signal but still excludes zero.

## Comparison with Wave 2.E (bid=30)

| Metric | Wave 2.E bid=30 | This probe high bids |
|--------|----------------|----------------------|
| N pairs | 52 | 1140 |
| EV delta (setter) | +3.09 | -10.42 |
| CI95 | [-0.57, +6.75] | [-11.25, -9.59] |
| p-value | 0.104 | << 0.001 |
| % pounce better | 34.6% | 24.5% |
| Oracle pounce rate | 59.6% | N/A (fresh inference) |

At bid=30, Wave 2.E found EV delta CI spans zero with direction suggesting
"decline better", but the t-stat was weak (p=0.104). At high bids, the same
direction is confirmed with much higher power (N=1140, t=-24.5).

Note the Wave 2.E bid=30 summary reported the direction convention reversed
from what this probe uses: their "decline_better_for_setter" corresponds to
positive EV delta in their convention where pounce favors setter. Reconciling:
at bid=30, pounce was better by E[Q] in 34.6% of cases; here at high bids,
pounce is better in only 24.5% of cases.

## Interpretation

### Why pounce is wrong-by-EV

When a setter wins a trick by burning trump or a high-rank tile:
1. The setter captures the count points in that trick (good)
2. The setter loses trump control for subsequent tricks (bad)
3. The bidder team has more leverage in subsequent tricks (bad for setter)

At high bids where the bidder is trying to score 35-42 points, the count tiles
are concentrated in fewer tricks. Winning one trick to capture 10 pts often
costs the setter trump control that would have been needed to set the contract
by blocking bidder's 3-4 remaining winning tricks.

### Why the signal is stronger at high bids

At high bids, the bid requirement is close to or equal to the maximum (42 pts).
This means the bidder has fewer "extra" points of margin. Paradoxically, this
makes the count points in any one trick less decisive — whether the setter
captures 10 pts now or the bidder captures them, the outcome depends on who
wins the remaining 2-4 tricks. The setter's trump control becomes the decisive
resource, and burning trump to pounce one count tile is a poor tradeoff.

### Conflict with Wave 2.B.2

Wave 2.B.2 reported that "mark_ev Q-delta evidence shows the signal flips in
book direction at bid >= 39." That finding was about the aggregate divergence
between mark_ev and threshold_mass across all actions as bid increases — not
about pounce-eligible positions specifically. The paired-contrast evidence here
shows no flip: decline is better across all bid buckets.

## Status Proposal

| Level | Justification |
|-------|--------------|
| **`contradicted`** (per-bid) | EV delta CI excludes zero in wrong direction at all 4 buckets |
| **`context-limited`** (overall) | Evidence is strong but limited to "broad pounce filter" snapshots; a late-game narrow probe might show different results in extreme positions |

The per-bid evidence supports upgrading from the baseline `context-limited`
(Wave 2.E bid=30 finding) to `contradicted` on the high-bid slice. The book
instruction to pounce on exposed count is not supported by oracle E[Q] evidence
at bids 35, 36, 39, or 42.

The orchestrator should weigh whether the "setter team led" subsample (39.6%
pounce better, N=280) merits a separate probe at a narrower filter before
finalizing the verdict.

## Caveats

1. E[Q] perspective (not mark_ev per position). A future probe should compute
   per-position mark_ev to test whether the mark-probability metric agrees.
2. Broad filter: includes positions where count was in the last trick (already
   secured by bidder), not necessarily the current trick.
3. bid=42 structural artifact: model returns mark_ev = -1.0 everywhere; p_set
   signal near-zero at bid=42 (CI barely spans zero: [-0.008, 0.000]).
4. Pounce tile selection = lowest cost winning play; high-trump pouncing may
   have a different profile than low-rank pouncing.
5. N_samples=100 per arm; large variance in some positions (EV deltas range
   from -55 to +27).

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-8kbh_pounce_high_bid/
├── README.md
├── manifest.json
├── summary.json
├── paired_contrasts.csv          (1140 rows)
├── slice_breakdown.csv
├── pounce_high_bid_snapshots.jsonl
└── run_pounce_high_bid_probe.py
```

## Related Pages

- [[w42-bookval-v1-wave2-pounce-window-bid30]] — sibling bid=30 probe (Wave 2.E)
- [[w42-bookval-v1-wave2-bid-aware-atlas]] — source corpus
- [[w42-book-validation-campaign]] — campaign overview
