# Reentry Preservation v2 Probe — Wave 2.A.3 (t42-v9lu)

## Question

Does preserving the bidder's single remaining trump (the "reentry") produce better expected
outcomes than consuming it immediately when the bidder holds at least 2 vulnerable off-suit
cards? Tests book claim `ch03-reentry-preservation`.

## Slice

- **Source**: 500 oracle-greedy snapshots from `reentry_preservation_v2` corpus
  - Mined from 22,801 candidates in 9 oracle-greedy corpus chunks
- **Declarations**: pip-trump decls 0–6 only (no NOTRUMP/doubles — no reentry concept)
- **Shape filter**: bidder's turn, exactly 1 trump remaining, >=2 distinct off suits,
  trick >= 2
- **Viable pairs**: 222 of 500 (278 skipped: trump not legal as follower = 208,
  no legal off-suit = 70)
- **Bidder**: player 0 (corpus invariant)
- **Bid value**: 30 (corpus default)

## N

**222 paired contrasts** (oracle-greedy snapshots, N=200 world samples per snapshot).

Branch A (consume): trump slot — play the single remaining trump now.
Branch B (preserve): highest-pip legal non-trump slot — cash a high off-suit tile instead.

## Paired vs Unpaired

Paired: each contrast is a single decision point evaluated for both actions simultaneously
by the oracle (one forward pass, two slots read).

## Metric

- **EV delta**: E[Q](preserve) − E[Q](consume); positive = book direction
- **CVaR_10 delta**: difference in bottom-10% conditional value at risk
- **Threshold-mass delta**: P(Q >= 30) difference

## Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| EV delta mean | **−1.23** | [−2.62, +0.16] |
| CVaR_10 delta mean | −0.51 | [−2.34, +1.32] |
| Threshold-mass delta | −0.015 | — |
| % preserve better | 49.1% | — |

**Overall CI spans zero** (barely, upper bound +0.16).

### Phase slices

| Phase | Tricks | N | EV delta mean | 95% CI |
|-------|--------|---|---------------|--------|
| early | 1–2 | 0 | — | — |
| mid | 3–4 | 162 | −0.22 | [−1.79, +1.36] |
| late | 5–6 | 60 | **−3.97** | [**−6.76, −1.17**] |

Late-game CI **excludes zero in the wrong direction** (consume > preserve), directly
contradicting the book's reentry advice in tricks 5–6.

## Claim-Ledger Impact

**`context-limited`**

The overall paired CI spans zero (inconclusive). However, the late-game slice (trick 5–6,
n=60) shows a significant signal that *contradicts* the book: consuming the trump is
reliably better in late tricks, with CI entirely negative [−6.76, −1.17]. Mid-game (trick
3–4, n=162) is underpowered. No early-game (trick ≤ 2) pairs were available.

The book's reentry claim is not supported at the overall level, and is specifically
contradicted in late-game positions.

## Caveats

- All snapshots use bid_value=30; no bid-value variation tested.
- Player 0 is always the bidder; seat generalization untested.
- The `best_off_suit` selection picks the highest-pip legal non-trump tile. In some follower
  positions, the legal set is constrained by suit-following — these are correctly excluded
  when trump itself is not legal.
- 278 of 500 snapshots were skipped because the contrast was not viable (trump not legal as
  follower, or no legal off-suit). Only positions where the bidder has a genuine choice were
  evaluated.
- EV delta uses a single Q-value estimate per action (200 world samples). Bootstrap CI is
  normal approximation (SE method), not bootstrap percentile.
- Wave 2.A predecessor used random-play context; these results use oracle-greedy context,
  which is materially better. The context bias from Wave 2.A is corrected.

## Command

```bash
cd /Users/jason/code/mk5-main
python w42/book_validation_v1/wave2/probes/t42-v9lu_reentry_v2/run_reentry_v2_probe.py \
    --samples 200 \
    --device mps \
    --output-dir w42/book_validation_v1/wave2/probes/t42-v9lu_reentry_v2
```

## Artifacts

- `paired_contrasts.csv` — 222 rows, one per viable snapshot
- `slice_by_phase.csv` — phase breakdown (mid / late)
- `summary.json` — machine-readable headline numbers
- `run_reentry_v2_probe.py` — probe script
