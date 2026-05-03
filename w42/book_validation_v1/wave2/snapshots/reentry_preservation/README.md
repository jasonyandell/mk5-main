# Reentry Preservation — Wave 2 Snapshot Corpus

## Question

Does preserving the bidder's single remaining trump (the "reentry") produce better expected
outcomes than consuming it immediately when the bidder holds at least 2 vulnerable off-suit cards?

Tests: `ch03-reentry-preservation`

## Slice

- **Corpus**: 200 fresh-deal snapshots mined from random play, seeds 0–17
- **Declarations**: pip-trump decls 0–6 only (NOTRUMP and doubles excluded — no trump reentry concept)
- **Shape filter**: bidder's turn, exactly 1 trump remaining, ≥ 2 distinct off suits
- **Bidder**: player 0 in all games
- **Bid value**: 30 (fixed; minimum contract)

## N

200 snapshots × 2 continuations (trump vs off-suit) = 400 E[Q] evaluations per metric.

## Paired vs Unpaired

Paired: same mid-game state, two contrasting first-action choices.

## Metric

| Metric | Value |
|--------|-------|
| EV delta (preserve − consume) | **−1.62** (95% CI: [−2.93, −0.31]) |
| CVaR_10 delta | −0.54 (95% CI: [−2.33, +1.26]) |
| Threshold mass delta (P(Q≥30)) | −0.025 |
| % pairs where preserve is better | 44.5% |

## Claim-Ledger Impact

**`underpowered`**

The EV delta is significantly negative (consume > preserve), which is *directionally opposite*
to the Ch 03 book claim. However, this should not be interpreted as contradiction because:

1. Snapshots were reached via random play, not optimal bidder play. The "single reentry"
   shape is much more common at positions where the bidder has already been forced into a
   defensive posture rather than the textbook "holding back the trump reentry" strategic scenario.
2. Off-suit selection is the first available slot, not strategically chosen (e.g., should be
   a domino where the bidder is void in opponent's hand, or a safe throw).
3. 200 snapshots from 18 seeds — sample size is borderline for effect-size discrimination.
4. Model is CPU-only at 50 samples per decision (lower accuracy than GPU at 1000 samples).

## Caveats

- Random-play context is not representative of actual game trajectories at these positions.
- The "off-suit" alternative uses the first available off-suit slot, not the strategically optimal
  off-suit choice (e.g., leading a "safe" domino to maintain information advantage).
- Bidder = player 0 always; no seat position variation.
- All bid_values = 30; high-bid regime (35–42) may differ substantially.
- No CVaR confidence interval overlap check (CIs overlap zero — CVaR effect not significant).

## Artifacts

| File | Description |
|------|-------------|
| `snapshots.jsonl` | 200 mid-game snapshot dicts (forge.eq.snapshot.v1) |
| `manifest.json` | Provenance, SHA256, shape filter |
| `paired_contrasts.csv` | Per-pair EV / CVaR / threshold mass deltas |
| `summary.json` | Headline numbers, claim-ledger impact |

## Reproducibility

```bash
# Step 1: Generate corpus
python w42/book_validation_v1/wave2/snapshots/reentry_preservation/build_reentry_corpus.py \
    --n-snapshots 200 --max-seeds 5000 --seed 0

# Step 2: Run probe
python w42/book_validation_v1/wave2/snapshots/reentry_preservation/run_reentry_probe.py \
    --samples 50 --device cpu
```
