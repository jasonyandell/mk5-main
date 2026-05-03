# Reentry Preservation v2 — Wave 2.A.2 Snapshot Corpus

## Question

Does preserving the bidder's single remaining trump (the "reentry") produce better expected
outcomes than consuming it immediately when the bidder holds at least 2 vulnerable off-suit cards?

Replaces Wave 2.A's random-play version with oracle-greedy source material.

Tests: `ch03-reentry-preservation`

## Slice

- **Source**: 111GB legacy oracle-greedy corpus (`gus/data/corpus_train_chunk_*.pt`)
- **Declarations**: pip-trump decls 0–6 only (NOTRUMP and doubles excluded — no trump reentry concept)
- **Shape filter**: bidder's turn, exactly 1 trump remaining, ≥ 2 distinct off suits with ≥ 1 tile each, trick ≥ 2
- **Bidder**: player 0 in all games (corpus invariant)
- **Bid value**: 30 (corpus default)

## N

500 snapshots (oracle-greedy decisions).

## Paired vs Unpaired

Snapshots are unpaired here. Paired E[Q] contrasts require a probe run (not included).

## Metric

E[Q] distributions at the decision point. Probe analysis pending.

## Claim-Ledger Impact

**`underpowered`** (pending probe run with these snapshots)

## Caveats

- Legacy corpus bid_value=None mapped to bid=30; no bid-value variation.
- Player 0 is always the bidder in this corpus; seat generalization untested.
- Filter hit rate: ~1.7% of decisions (~50 per 2800-decision chunk).
- Wave 2.A used random-play; this v2 uses oracle-greedy, materially improving source quality.

## Command

```bash
python w42/book_validation_v1/wave2/snapshots/reentry_preservation_v2/build_reentry_preservation_v2_corpus.py
```

Or via unified driver:

```bash
python w42/book_validation_v1/wave2/snapshots/mine_legacy_snapshots.py
```

## Status

`snapshots_ready` — awaiting probe run to generate paired E[Q] contrasts.
