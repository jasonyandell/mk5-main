# t42-ey88 — ch02 Bid-Only-Enough Multi-Step Extension

**Wave:** 2.G
**Bead:** t42-ey88 (under epic t42-4zi6)
**Claim:** ch02-bid-only-enough

## Question

Wave 2.B.2 showed that bidding 32 vs 30 hurts the bidding team (mark_ev delta −0.076, p_make delta −0.038). The book's claim is broader: never overbid regardless of magnitude. Does the "overbidding hurts" penalty persist across **all** adjacent bid steps (30→32, 32→35, 35→36, 36→39, 39→42), and does it hold under slice analysis?

## Slice

- Corpus: `bid_aware_actions.csv`, seeds 9000–9049 (50 seeds × 10 decls)
- Restricted to `is_actual_action == 1` (played actions only, matching Wave 2.B.2's contract)
- Paired on `(seed, decl_id, decision_idx)` — same hand, same game position, different bid threshold
- N per step pair: 8,168–10,052 (pairing reduces from 14,000; see N column in step_pair_deltas.csv)

## Method

- **Paired delta:** delta = value(lower_bid) − value(higher_bid) for each paired triplet
- **Bootstrap CI:** 2000 iterations, percentile method, seed 42
- **Metrics:** mark_ev, p_make, threshold_mass, scalar Q mean
- **Slices:** declaration name, seat role (bidder/partner/setter), game phase (early/mid/late by trick_idx)
- **Transitive check:** direct 30↔42 delta vs sum of 5 step deltas

## Results

### Step Pair Deltas (mark_ev, lower − higher; positive = lower bid better)

| Pair   |    N | Delta  | 95% CI                | Cohen d | p-value | Book? |
|--------|-----:|-------:|----------------------:|--------:|--------:|-------|
| 30↔32  | 10052| +0.0743| [+0.0665, +0.0819]   | +0.186  | <1e-300 | YES   |
| 32↔35  |  9800| +0.1146| [+0.1066, +0.1221]   | +0.299  | <1e-300 | YES   |
| 35↔36  |  9888| +0.0476| [+0.0418, +0.0536]   | +0.158  | <1e-300 | YES   |
| 36↔39  |  9584| +0.1032| [+0.0973, +0.1095]   | +0.337  | <1e-300 | YES   |
| 39↔42  |  8168| +0.1459| [+0.1393, +0.1529]   | +0.472  | <1e-300 | YES   |

All 5 step pairs: CI excludes zero in book direction. No reversal anywhere.

### p_make (same direction as mark_ev)

| Pair   | Delta  | 95% CI                |
|--------|-------:|----------------------:|
| 30↔32  | +0.0372| [+0.0334, +0.0411]   |
| 32↔35  | +0.0573| [+0.0533, +0.0612]   |
| 35↔36  | +0.0238| [+0.0209, +0.0268]   |
| 36↔39  | +0.0516| [+0.0486, +0.0547]   |
| 39↔42  | +0.0729| [+0.0696, +0.0762]   |

### Transitive Check (30↔42)

| Metric          | Direct 30→42 | Sum of Steps | Deviation | Additive? |
|-----------------|-------------:|-------------:|----------:|-----------|
| mark_ev         |      +0.4817 |      +0.4856 |    −0.81% | Yes       |
| p_make          |      +0.2409 |      +0.2428 |    −0.81% | Yes       |
| threshold_mass  |      +0.0124 |      +0.0114 |     +7.7% | Yes       |

Penalty is ~additive (deviations < 8%). No evidence of nonlinear amplification or dampening.

### Slice Breakdown (mark_ev, averaged across all 5 step pairs)

**By declaration (best → worst overbid penalty):**
- blanks: +0.105, ones: +0.105, no-trump: +0.103 (worst penalty = most harmed by overbidding)
- doubles: +0.084 (least harmed — doubles contract is already restrictive)

**By seat role:**
- bidder_partner: +0.111 (highest penalty), bidder: +0.104
- right_setter: +0.088, left_setter: +0.086

**By phase:**
- early tricks (0-1): +0.123 (overbid penalty largest early — bidder pays most at start)
- mid tricks (2-4): +0.105
- late tricks (5-6): +0.037 (small but CI still excludes zero: [+0.0003, +0.0005])

**All 85 slice cells support the book direction (85/85 CIs exclude zero).**

Best slice: 39↔42 / early phase, delta = +0.298, CI=[+0.285, +0.311]
Worst slice: 39↔42 / late phase, delta = +0.003, CI=[+0.0003, +0.006] (barely significant)

### Bid=84 (treated separately — all-tricks double-mark contract)

**bid=42 vs bid=84:**
- mark_ev delta: +1.000 (ceiling: bid=42 always makes 0 marks, bid=84 always loses 2 marks)
- p_make: 0.000 (both bids fail equally on p_make — both have 0% make rate)
- scalar Q mean: +0.065, CI=[−0.139, +0.274] (not significant — Q distribution similar)

**bid=30 vs bid=84:**
- mark_ev delta: +1.490, CI=[+1.479, +1.502] (huge overbid penalty, as expected)
- p_make: +0.245, CI=[+0.239, +0.251]

Interpretation: bid=84 is categorically different — it carries double mark risk. The +1.0 mark_ev gap vs bid=42 reflects the 2× multiplier. The scalar Q gap is small, meaning the underlying game difficulty doesn't change much — the penalty is purely contractual.

## Monotonicity

YES. Every step pair shows a positive, significant overbid penalty. The effect sizes trend upward at higher bids (d = 0.16 to 0.47), consistent with the intuition that the gap to the threshold widens faster at higher bids.

## Status Proposal

**`supported`** on the same-hand bid-margin slice.

Justification: All 5 adjacent step pairs show overbidding penalty in mark_ev and p_make with 95% CIs excluding zero. The effect is monotone (no reversal), holds across all 85 slice cells (decl × seat × phase), and the transitive 30→42 check confirms near-additivity (≤8% deviation). This is a stronger result than Wave 2.B.2's `context-limited` promotion — the claim holds broadly, not just at the 30→32 step.

## Caveats

1. Bid-only-enough applies on the same contract (same seed × decl_id). Cross-contract comparisons (e.g., choosing between bid=30/blanks vs bid=32/sixes) are not covered here.
2. The late-game slice is barely significant (p≈0.05 at 39↔42/late). By the time few tricks remain, the counterfactual matters less.
3. Bid=84 results assume bid=42 as the "minimum winning" baseline — the 84-specific strategy is not covered by ch02.
4. These are oracle-based (perfect-info) counterfactuals. Human play and auction constraints are not modeled.

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-ey88_ch02_multistep/
├── README.md                      # this file
├── manifest.json                  # provenance
├── summary.json                   # machine-readable headline
├── step_pair_deltas.csv           # N=20 rows (5 pairs × 4 metrics)
├── slice_breakdown.csv            # N=85 rows
├── transitive_check.csv           # N=3 rows
└── run_ch02_multistep.py          # reproducibility script
```

Reproducibility command:
```bash
python3 w42/book_validation_v1/wave2/probes/t42-ey88_ch02_multistep/run_ch02_multistep.py
```
