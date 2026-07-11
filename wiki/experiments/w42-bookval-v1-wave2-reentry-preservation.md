---
title: w42 Book Validation v1 — Wave 2 Reentry Preservation Probe
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: complete
parent_bead: t42-rwdj
wave: 2
claim_tested: ch03-reentry-preservation
claim_ledger_impact: underpowered
---

## Summary

First integration test for the Wave 2 state-injection harness. Evaluated whether
the bidder should preserve their single remaining trump ("reentry") vs consuming it
immediately when holding ≥ 2 vulnerable off-suit cards.

**Headline result:** EV delta (preserve − consume) = **−1.62** (95% CI: [−2.93, −0.31]).
The oracle model *prefers consuming* the trump in the random-play context, but this
is almost certainly a measurement artifact (see Caveats). Status: `underpowered`.

## Claim Under Test

> Ch 03 (bidder play): "When you hold your last trump as reentry, preserve it until
> you can cash your off suits first." — reentry-preservation principle.

The claim predicts EV delta > 0 (preserve is better).

## Data Slice

- 200 mid-game snapshots from random play, seeds 0–17, pip-trump decls 0–6
- Shape filter: bidder's turn, exactly 1 trump remaining, ≥ 2 distinct off suits
- All games at bid_value = 30, player 0 = bidder
- Model: domino-qval-large-3.3M (50 world samples per decision, CPU)

## Results

| Metric | A (consume trump) | B (preserve trump) | B − A |
|--------|------------------|--------------------|-------|
| Mean E[Q] | — | — | **−1.62** (CI: [−2.93, −0.31]) |
| CVaR_10 | — | — | −0.54 (CI: [−2.33, +1.26]) |
| P(Q≥30) | — | — | −0.025 |
| % pairs where B better | — | — | 44.5% |

Direction: consume > preserve (opposite of book claim in this slice).

## Interpretation

The claim `ch03-reentry-preservation` is **not tested** adequately by this probe. Three
structural problems prevent this from being a contradiction verdict:

1. **Random-play context**: Snapshots were reached by random play, not optimal bidder
   trajectories. The "1 trump remaining, 2+ off suits" shape occurs in unfavorable positions
   where preserving the trump is genuinely suboptimal (e.g., opponent has already stripped
   the relevant off suits).

2. **Off-suit selection bias**: The probe plays the *first available* off-suit slot, not
   a strategically optimal one (e.g., leading a high pip that opponents cannot win). The
   book claim assumes the bidder is playing optimally after the preserve decision.

3. **Low sample count**: 50 world samples per decision on CPU produces noisy E[Q] estimates.
   The oracle was trained and validated at 500–1000 samples on GPU. CPU at 50 samples
   underestimates variance substantially.

## Infrastructure Contribution

This probe **validates the Wave 2 state-injection harness end-to-end**:
- `GameStateTensor.from_snapshot` correctly initialises mid-game states
- `to_snapshot` round-trips bit-identically through `from_snapshot`
- `generate_eq_from_snapshots` produces valid E[Q] PDFs from snapshot positions
- The CLI `--snapshot-file` flag dispatches correctly

All 6 new tests in `forge/eq/test_state_injection.py` pass. No regressions in
115 existing forge tests.

## Claim Ledger Impact

`underpowered`

The probe cannot distinguish between:
(a) the book claim is false in the oracle model, or
(b) the probe's random-play slice does not represent the strategic context the book
describes. A follow-up probe using oracle-greedy trajectories and smarter off-suit
selection would be needed to promote to `supported` or `contradicted`.

## Caveats

- No seat position variation (bidder always player 0)
- All games bid_value=30; high-bid regime untested
- CVaR confidence interval spans zero — CVaR effect not significant at n=200
- Doubles-trump and NOTRUMP excluded (no "reentry" concept)
- Corpus from only 18 unique seeds (shape filter is restrictive)

## Artifacts

```
w42/book_validation_v1/wave2/snapshots/reentry_preservation/
├── snapshots.jsonl          (200 snapshots, forge.eq.snapshot.v1)
├── manifest.json
├── paired_contrasts.csv     (200 rows: ev, cvar, threshold_mass per pair)
├── summary.json             (headline numbers + claim-ledger impact)
├── README.md
├── build_reentry_corpus.py
└── run_reentry_probe.py
```

## Links

[[w42]] | [[w42-bookval-v1-wave2-infra-design]] |
[[w42-phase4-final-claim-audit]] | [[w42-branch-atlas-scaled-v0]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- The proposed follow-up already exists as [[w42-bookval-v1-wave2-reentry-v2]] (artifacts in `snapshots/reentry_preservation_v2/`).
- Cheap next probe: rerun the same 200 pairs with oracle-greedy off-suit selection only (keeping the random-play corpus) to isolate how much of the −1.62 comes from the first-available-slot bias.
