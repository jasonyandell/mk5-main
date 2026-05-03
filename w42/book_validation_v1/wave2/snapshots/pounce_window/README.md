# Pounce Window — Wave 2.A.2 Snapshot Corpus (bid=30 slice)

## Question

When the setter is on lead or following and the bidder's team played a count domino
(≥ 5 pts) in the most recent trick, does the setter gain by playing aggressively
to capture that count exposure?

Tests: `ch12-setter-pounce` (bid=30 slice)
Note: `ch12-setter-pounce-high-bid-off` (high-bid slice) deferred to Wave 2.B.2.

## Slice

- **Source**: 111GB legacy oracle-greedy corpus (`gus/data/corpus_train_chunk_*.pt`)
- **Declarations**: all 10
- **Shape filter**: setter (player 1 or 3) on lead or following, bidder's team played a count domino (≥ 5 pts) in the immediately preceding completed trick, trick ≥ 1
- **Bid value**: 30 (corpus default)

## N

500 snapshots (oracle-greedy decisions).

## Paired vs Unpaired

Unpaired (snapshots only). Paired probe pending.

## Metric

E[Q] delta: aggressive setter play vs passive setter play after count exposure.

## Claim-Ledger Impact

**`underpowered`** (pending probe run)

## Caveats

- Filter is broad: any setter decision after bidder's count exposure qualifies.
- Does not distinguish between count domino still in play vs already captured.
- High hit rate (~14%) may include many non-actionable positions (setter can't actually pounce).
- High-bid-off slice (ch12-setter-pounce-high-bid-off) is explicitly deferred — legacy corpus has no bid-aware context.

## Command

```bash
python w42/book_validation_v1/wave2/snapshots/pounce_window/build_pounce_window_corpus.py
```

## Status

`snapshots_ready` — awaiting probe run.
