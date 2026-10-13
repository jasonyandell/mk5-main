---
kind: experiment
title: "W42 Book Validation v1 — Wave 2.E.2: Pounce High-Bid Probe"
bead: t42-8kbh
parent_bead: t42-4zi6
wave: "2.E.2"
claim: ch12-setter-pounce-high-bid-off
status: complete
created: 2026-05-03
first_seen: 2026-05-03
last_updated: 2026-10-12
---

## Summary

Snapshot-level paired-contrast probe for `ch12-setter-pounce-high-bid-off` at
bids 35/36/39/42. Extends [[w42-bookval-v1-wave2-pounce-window-bid30]] (Wave
2.E, bid=30) to the high-bid regime.

**Verdict: withdrawn pending a sign check (#122).** The script treats the
generator's E[Q] as Team 0's view and negates it for the setter
(`run_pounce_high_bid_probe.py:20,426,438`); forge documents E[Q] from the
acting player's view (`forge/eq/generate/actions.py:113,160`), and the setter
is the acting player. If the sign is inverted, the recorded −10.42 is +10.42
for the pounce, matching every actor-perspective corpus pounce contrast
(+4.10, +3.85, +4.93). The ledger row `ch12-setter-pounce-high-bid-off` was
demoted to `contradicted` on this probe; that verdict is withdrawn with it.

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
- EV delta = ev_pounce_t0 - ev_decline_t0 (read as Team 0 perspective; unverified, #122)
- ev_delta_setter = -ev_delta_t0 (positive = pounce better for setter, if the
  Team 0 reading is right)
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

All CIs exclude zero; which direction they favour depends on the unverified sign (#122).

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

At bid=30 the contrast was null (p=0.104). Both probes share the unverified
sign convention.

Note the Wave 2.E bid=30 summary reported the direction convention reversed
from what this probe uses: their "decline_better_for_setter" corresponds to
positive EV delta in their convention where pounce favors setter. Reconciling:
at bid=30, pounce was better by E[Q] in 34.6% of cases; here at high bids,
pounce is better in only 24.5% of cases.

## Interpretation

None stands until the sign is checked (#122). The mechanism the probe
offered (pouncing burns trump control the setter needs later) explained a
direction that may be inverted. Independent of the sign:

- The test compares two hand-picked actions ("lowest-cost winning play" vs
  "lowest-cost non-winning play"), not the book's act, and does not measure
  how often the book's action is the argmax.
- The evaluator sees all four hands in every sampled world; the book's clause
  is a hedge for a defender who cannot (caveat 0).
- Labels come from the learned Q network (qMAE 0.94 points), in points; CIs
  resample decisions, not the 50 seeds.
- The Wave 2.B.2 aggregate setter-team Q delta that first promoted the row is
  not a pounce contrast either.

## Status Proposal

The probe proposed `contradicted` per bid and `context-limited` overall; the
orchestrator took `contradicted`. Both proposals are withdrawn pending #122.
The "promotion guard" rule in `w42/book_validation_v1/AGENTS.md` was drawn
from this reversal; its precedent is withdrawn with it.

## Caveats

0. **Information-regime challenge (2026-07-07, [[w42-book-second-pass]] §4).**
   The book's actual clause (p. 112) is "pouncing when you can, **regardless of
   whether you know who will win the trick**… Play it or lose it!" — an
   explicitly imperfect-information hedge for a defender who cannot see whether
   partner takes the trick. This probe's E[Q] arms evaluate under
   sampled-complete-worlds where the oracle always knows the trick winner, a
   regime in which "pounce regardless" is worse by construction. The
   `contradicted` verdict therefore may be a category error rather than a wrong
   book claim; the proposed re-adjudication is a belief/PIMC defender that does
   not see hands, scoped to bid ≥ 35 and bidder ≤ 2 offs. The Zeb-era study
   ([[book-study-2026-10-12]]) priced a different act (count before partner
   can play) and did not reproduce this test.
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
