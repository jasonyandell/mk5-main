---
title: w42 Book Validation v1 — Wave 2.A.3 Reentry Preservation v2 Probe
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: complete
parent_bead: t42-v9lu
wave: 2.A.3
claim_tested: ch03-reentry-preservation
claim_ledger_impact: context-limited
predecessor: "[[w42-bookval-v1-wave2-reentry-preservation]]"
---

## Summary

Re-run of the reentry-preservation probe on oracle-greedy source material, correcting the
context bias that made Wave 2.A `underpowered`. With 222 viable pairs from 500 oracle-greedy
snapshots (N=200 world samples each, MPS device), the overall CI barely spans zero but the
late-game slice shows a clear signal in the **wrong** direction.

**Headline result (all phases):** EV delta = **−1.23** (95% CI: [−2.62, +0.16]). CI spans
zero. Overall verdict: inconclusive.

**Late-game result (tricks 5–6, n=60):** EV delta = **−3.97** (95% CI: [**−6.76, −1.17**]).
CI excludes zero. Consuming the trump is significantly better in late tricks. This **directly
contradicts** the book's reentry-preservation advice in the late-game context.

**Status: `context-limited`**

## Claim Under Test

> Ch 03 (bidder play): "When you hold your last trump as reentry, preserve it until you can
> cash your off suits first." — reentry-preservation principle.

The claim predicts EV delta > 0 (preserve > consume).

## Improvement Over Wave 2.A

The predecessor probe ([[w42-bookval-v1-wave2-reentry-preservation]]) suffered from three
structural problems that this probe corrects:

1. **Random-play context** → now oracle-greedy: snapshots reached by oracle-optimal play are
   strategically valid positions, not the distorted "stuck with 1 trump" states that random
   play produces.

2. **Naive off-suit selection** → now highest-pip: instead of "first available off-suit slot,"
   we pick the highest pip-sum non-trump tile in the bidder's legal hand. This is the book's
   notion of "cashing a high tile" — the tile most worth protecting and most relevant to the
   reentry dilemma.

3. **CPU with 50 samples** → **MPS with 200 samples**: proper inference budget on GPU-class
   hardware.

## Data Slice

- **Source**: 500 oracle-greedy snapshots (`reentry_preservation_v2` corpus)
  - Mined from 22,801 candidates in 9 corpus chunks
- **Declarations**: pip-trump decls 0–6 (NOTRUMP/doubles excluded — no reentry concept)
- **Shape filter**: bidder's turn, exactly 1 trump remaining, ≥ 2 distinct off suits, trick ≥ 2
- **Viable pairs**: 222 of 500 (278 skipped: trump not legal in follower position = 208;
  no legal off-suit available = 70)
- Bid value: 30 | Player 0 = bidder (corpus invariant)
- Model: domino-qval-large-3.3M | N=200 world samples per snapshot | device=mps

## Results

### Overall

| Metric | Mean | 95% CI |
|--------|------|--------|
| EV delta (preserve − consume) | **−1.23** | [−2.62, +0.16] |
| CVaR_10 delta | −0.51 | [−2.34, +1.32] |
| P(Q≥30) delta | −0.015 | — |
| % pairs where preserve better | 49.1% | — |

Direction: consume > preserve (opposite book claim direction).
Overall CI spans zero — verdict at whole-sample level: inconclusive.

### Phase Slices

| Phase | Tricks | N | EV delta mean | 95% CI | Verdict |
|-------|--------|---|---------------|--------|---------|
| early | 1–2 | 0 | — | — | (no data) |
| mid | 3–4 | 162 | −0.22 | [−1.79, +1.36] | underpowered |
| **late** | **5–6** | **60** | **−3.97** | **[−6.76, −1.17]** | **contradicted** |

The late-game slice is the most striking: in tricks 5–6, the oracle consistently prefers
consuming the trump rather than preserving it. This may reflect that by trick 5, the tactical
situation has evolved: the bidder has likely already established their book on off suits, and
holding the trump adds little value while costing a play.

## Interpretation

The book's reentry advice is context-sensitive. In the oracle model:

- **Mid-game (tricks 3–4)**: No significant signal either way. The position is genuinely
  uncertain; preserving or consuming the trump has similar expected outcomes on average.
- **Late-game (tricks 5–6)**: Strong signal favoring consumption. By trick 5, the bidder
  typically needs their remaining tiles to score, not to "re-enter" — the reentry window has
  likely already closed or the off suits have been cashed.

The book's advice "preserve your reentry" is probably sound in very early play (tricks 2–3
after bidding), but the oracle data suggests the window closes by trick 5. The 0-early-pairs
gap in our corpus means we cannot validate the claim in the regime where it is most likely
to hold.

## Claim Ledger Impact

**`context-limited`**

The evidence is phase-dependent:
- Late-game (tricks 5–6): CI excludes zero, direction contradicts book (consume > preserve).
- Mid-game (tricks 3–4): CI includes zero, underpowered.
- Early-game (trick ≤ 2): no data from this corpus.

The claim cannot be promoted to `supported`. It may warrant re-evaluation as "preserve early,
consume late" — a more nuanced version of the book's advice.

## Caveats

- Bid value fixed at 30; higher bid regimes (84, 35+) untested.
- Player 0 is always the bidder; seat generalization untested.
- 278 of 500 snapshots skipped: the shape filter allowed follower positions where trick
  suit-following made trump non-legal. Only positions with genuine choice evaluated.
- No early-game pairs (trick ≤ 2) produced by this corpus — the book's primary advice
  context is untested.
- EV delta from a single E[Q] estimate at 200 samples; no cross-run variance estimate.
- CVaR CI spans zero — the downside risk metric is not significant at this N.

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-v9lu_reentry_v2/
├── README.md
├── manifest.json
├── summary.json
├── paired_contrasts.csv     (222 rows)
├── slice_by_phase.csv       (mid / late)
└── run_reentry_v2_probe.py
```

Wall time: 135 seconds (2.25 minutes) on Apple MPS.

## Links

[[w42]] | [[w42-bookval-v1-wave2-reentry-preservation]] |
[[w42-bookval-v1-wave2-infra-design]] | [[w42-phase4-final-claim-audit]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Cheap next probe: relax the shape filter to trick ≥ 1 (or mine a dedicated early-game corpus) to fill the 0-pair early slice — the book's primary regime remains untested.
- The 208 "trump not legal in follower position" skips suggest the miner's shape filter could require bidder-leads positions upfront, doubling yield per corpus pass.
