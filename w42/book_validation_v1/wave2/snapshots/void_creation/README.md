# Void Creation — Wave 2.A.2 Snapshot Corpus

## Question

When the setter (opponent) is on lead and holds a suit with only one tile, does leading
that tile (to void the suit) produce better outcomes than leading from longer suits?

Tests: `ch05-void-creation`

## Slice

- **Source**: 111GB legacy oracle-greedy corpus (`gus/data/corpus_train_chunk_*.pt`)
- **Declarations**: all 10 (setter strategy applies across all decl types)
- **Shape filter**: setter (player 1 or 3) on lead, holds ≥ 2 distinct suits, at least 1 suit has exactly 1 domino (void-creation candidate), trick ≥ 1
- **Bid value**: 30 (corpus default)

## N

500 snapshots (oracle-greedy decisions).

## Paired vs Unpaired

Unpaired (snapshots only). Paired probe pending.

## Metric

E[Q] delta: leading singleton (voiding) vs leading from longer suit.

## Claim-Ledger Impact

**`underpowered`** (pending probe run)

## Caveats

- Filter is broad: any setter on lead with a singleton suit qualifies.
- Does not verify the singleton is in an off suit vs trump.
- Does not capture the follow-on impact of voidness in subsequent tricks.
- Filter hit rate: ~8.6% of decisions.

## Command

```bash
python w42/book_validation_v1/wave2/snapshots/void_creation/build_void_creation_corpus.py
```

## Status

`snapshots_ready` — awaiting probe run.
