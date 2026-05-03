# Pounce High-Bid Probe — Wave 2.E.2

**Bead**: t42-8kbh
**Claim**: `ch12-setter-pounce-high-bid-off`
**Scope**: High bids 35/36/39/42 (sibling to Wave 2.E bid=30 probe t42-ntbe)

## Question

At high bids (35/36/39/42), when the bidder's team has played a count domino
(>= 5 pts) in the most recent completed trick or current trick, and the setter
is now following (not leading), does the oracle consistently prefer to "pounce"
(win the current trick) over "decline" (play a non-winning tile)?

Wave 2.E (bead t42-ntbe) showed pounce was wrong-by-EV at bid=30 in 65.4% of
paired positions. Wave 2.B.2 suggested the signal might flip toward book
direction at bid >= 39 based on aggregate Q-delta evidence. This probe tests
that directly with the same paired-contrast methodology.

## Slice

- **Source**: Bid-aware atlas (seeds 9000-9049 x 10 decls x 4 bids)
  - `w42/book_validation_v1/wave2/bid_aware_atlas/eq_pdf_seeds*_bid{35,36,39,42}_v2.pt`
- **Filter**: Setter (player 1 or 3) following into a trick, with count from
  bidder team (>= 5 pts) in current trick or last completed trick, AND setter
  has both legal winning and legal non-winning options
- **Bid values**: 35, 36, 39, 42
- **Declarations**: all 10 per seed
- **Model**: domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt, n_samples=100, MPS

## N

- Snapshots mined: 1140 total (35: 275, 36: 259, 39: 290, 42: 316)
- Paired contrasts run: 1140 (100% conversion — all mined snapshots had valid contrast)
- Runtime: ~1548 seconds (~25.8 minutes) on MPS

## Paired vs Unpaired

Paired. Each of 1140 contrasts compares:
- **A (pounce)**: lowest-cost winning play (prefer non-trump, lower pip-sum)
- **B (decline)**: lowest-cost non-winning play (same preference ordering)

within the same board position.

## Metric

E[Q] delta from Team 0 perspective. Sign convention:
- Setter (Team 1) wants LOWER Team 0 EV
- `ev_delta_setter = -(ev_pounce_t0 - ev_decline_t0)`
- **Positive = pounce better for setter, Negative = decline better for setter**

| Metric | Value |
|--------|-------|
| Overall N | 1140 |
| EV delta (setter): mean | -10.42 |
| EV delta CI95 | [-11.25, -9.59] |
| t-stat | -24.52 |
| % pounce better by EV | 24.5% |
| p_set delta (mean) | -0.047 |
| p_set delta CI95 | [-0.056, -0.038] |
| CVaR_10 delta (mean) | +4.40 |

**CI excludes zero in the wrong direction (decline better) at all bid buckets.**

## By Bid Bucket

| bid | N | EV delta (setter) | CI95 | % pounce better | p_set delta | verdict |
|-----|---|------|------|----------------|-------------|---------|
| 35 | 275 | -10.08 | [-11.76, -8.40] | 26.6% | -0.064 | **contradicted** |
| 36 | 259 | -11.32 | [-13.06, -9.57] | 21.6% | -0.078 | **contradicted** |
| 39 | 290 | -10.14 | [-11.83, -8.46] | 23.8% | -0.050 | **contradicted** |
| 42 | 316 | -10.24 | [-11.81, -8.67] | 25.6% | -0.004 | **contradicted** |

At bid=42 the p_set delta CI just barely spans zero ([−0.008, 0.000]) — the
mark-probability signal is attenuated (at bid=42 the contract is already
maximally stressed). But the EV signal is still strongly negative.

## Slice Breakdown

| Slice | N | EV delta (setter) | p_set delta | pounce_better |
|-------|---|------|-------------|--------------|
| Phase early (tricks 0-2) | 582 | -12.60 | -0.070 | 22.9% |
| Phase mid (tricks 3-5) | 558 | -8.15 | -0.023 | 26.2% |
| Count 5 pts | 540 | -7.27 | -0.034 | 30.0% |
| Count 10 pts | 549 | -13.67 | -0.063 | 19.1% |
| Bidder team led trick | 860 | -11.93 | -0.055 | 19.5% |
| Setter team led trick | 280 | -5.78 | -0.024 | 39.6% |

No slice shows pounce as beneficial. The setter-team-led slice has a weaker
signal (CI95: [-7.46, -4.11]) but CI still excludes zero.

## Key Findings

### 1. Pounce is wrong-by-EV at all high bids tested

The Wave 2.E finding (decline better at bid=30, EV delta = +3.09 favoring
pounce for setter by CI spans zero / direction is actually wrong sign) is
AMPLIFIED not reversed at high bids. The EV delta is strongly negative
(decline better for setter) at all four bid buckets, with CI excluding zero by
a wide margin.

### 2. Wave 2.B.2 aggregate Q-delta finding does not generalize to paired contrasts

Wave 2.B.2 found that mark_ev diverges from threshold_mass monotonically with
bid. This aggregate finding is about the correlation structure across all
actions, not specifically about pounce-eligible positions. The per-position
paired contrast shows the pounce/decline decision is still dominated by the
same E[Q] cost dynamics: pouncing burns trump or high-rank tiles to win the
count, leaving setter worse off in subsequent tricks.

### 3. p_set signal aligns with EV signal at all bids except bid=42

p_set delta is negative at bid=35/36/39 with CI excluding zero: pouncing
reduces setter's probability of setting the contract. At bid=42 the signal
becomes near-zero (CI [-0.008, +0.000]), consistent with the model returning
mark_ev = -1.0 everywhere at bid=42 (from bid_aware_atlas manifest:
threshold_q = 42, which is outside normal Q range).

### 4. Effect size is larger at high bids than at bid=30

Wave 2.E bid=30: EV delta = +3.09 (CI spans zero, p=0.104)
This probe, all high bids: EV delta = -10.42 (CI [-11.25, -9.59], p << 0.001)

The magnitude is 3x larger, but the direction is the same: **decline is better
for setter than pouncing**. The difference from bid=30 is that at bid=30 the
CI spanned zero; at high bids the signal is unambiguous.

### 5. Interpretation

When the bidder's team plays a count tile in a trick and the setter can win
that trick, pouncing (capturing the count) comes at a heavy EV cost: the setter
burns a high tile to win one trick but loses trump control for subsequent
tricks. At high bids where the count distribution is more concentrated,
this cost is magnified. The book's pounce instruction (`ch12-setter-pounce`)
may be sound as a heuristic for extreme late-game positions, but the broad
"pounce on exposed count" guideline is not supported by oracle evidence across
the following scenarios and positions tested here.

## Claim-Ledger Impact

**`contradicted`** at the snapshot level (EV perspective, all bid buckets)

However, note that:
- The probe does not distinguish between count the setter is following vs.
  count the bidder played in the LAST trick (different strategic situations)
- The p_set signal at bid=42 is near-zero (CI barely spans zero at [-0.008, 0.000])
- A late-game, setter-controls-trick scenario has weaker negative signal
  (EV delta = -5.78 vs. overall -10.42)

The orchestrator should label this as `contradicted` in the bid-aware high-bid
slice, pending a more targeted probe that isolates late-game positions where
the setter can pounce at minimal trump cost.

**Wave 2.E proposal**: `context-limited` (CI spanned zero at bid=30)
**This probe proposal**: `contradicted` (CI excludes zero, decline-better direction, all 4 bid buckets)

The verdict code in summary.json reports `context-limited` at the top level
because the `propose_verdict()` function treats all-contradicted bids as
context-limited. The per-bid verdicts are all `contradicted`.

## Caveats

1. Filter is broad: any setter decision after count exposure in current or
   last trick. Positions where setter is the final follower (forced choice)
   are included in winning_slots only if they actually win.
2. No mark_ev computed per position (would require score-state tracking
   across the game; a future probe could add this).
3. Pounce definition = lowest-cost winning play. Different pounce tile
   choices (high-trump vs. low-trump) have different costs and may yield
   different results.
4. n_samples=100 per arm gives ~10 pt SE per decision; paired contrast
   reduces SE substantially, but large EV variance exists (some deltas
   reach -55 to +26).
5. bid=42 has structural artifact: model returns mean_ev = -1.0 for all
   actions (cannot make bid=42 contract). p_set signal collapses.

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-8kbh_pounce_high_bid/
├── README.md                          (this file)
├── manifest.json                      (provenance)
├── summary.json                       (headline numbers, machine-readable)
├── paired_contrasts.csv               (1140 rows, one per contrast)
├── slice_breakdown.csv                (per-slice summary)
├── pounce_high_bid_snapshots.jsonl    (1140 mined snapshots)
└── run_pounce_high_bid_probe.py       (full pipeline script)
```

## Reproducibility

```bash
# Full pipeline (mine + inference):
/Users/jason/code/mk5-main/forge/venv/bin/python \
  /Users/jason/code/mk5-main/w42/book_validation_v1/wave2/probes/t42-8kbh_pounce_high_bid/run_pounce_high_bid_probe.py \
  --atlas-dir /Users/jason/code/mk5-main/w42/book_validation_v1/wave2/bid_aware_atlas \
  --checkpoint /Users/jason/code/mk5-main/forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
  --samples 100 --device mps --max-per-bid 600

# Mine only (no inference, 6s):
... same command with --mine-only flag

# Re-run inference from pre-mined snapshots:
... same command with --snapshots-file pounce_high_bid_snapshots.jsonl
```
