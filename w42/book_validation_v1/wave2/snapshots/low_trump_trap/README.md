# Low Trump Trap — Wave 2.A.2 Snapshot Corpus

## Question

When the bidder must play trump (void in led suit or led suit is trump) and holds both
the dominant trump and at least one lower trump, does playing the dominant trump
(or saving it for later) produce better outcomes?

Tests: `ch04-low-trump-trap-against-count-dump`

## Slice

- **Source**: 111GB legacy oracle-greedy corpus (`gus/data/corpus_train_chunk_*.pt`)
- **Declarations**: pip-trump decls 0–6 only
- **Shape filter**: bidder is following (not on lead), bidder is void in led suit OR led suit is trump, bidder holds the dominant trump (highest unplayed) AND ≥ 1 other low trump, trick ≥ 1
- **Bid value**: 30 (corpus default)

## N

300 snapshots (oracle-greedy decisions).

## Paired vs Unpaired

Unpaired (snapshots only). Paired probe pending.

## Metric

E[Q] delta: playing dominant trump vs playing lower trump.

## Claim-Ledger Impact

**`underpowered`** (pending probe run)

## Caveats

- Rarer shape: ~0.6% of decisions. Took ~19 chunks to hit 300.
- "Dominant trump" defined as highest-ranking unplayed trump overall (not just in player's hand).
- Does not model partner's trump holdings.
- Requires probe run to generate paired contrasts.

## Command

```bash
python w42/book_validation_v1/wave2/snapshots/low_trump_trap/build_low_trump_trap_corpus.py
```

## Status

`snapshots_ready` — awaiting probe run.
