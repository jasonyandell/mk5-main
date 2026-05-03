# Eighty-Four Throwaway — Wave 2.A.2 Snapshot Corpus

## Question

In an 84-bid (plunge) context, when the bidder holds throwaway tiles, what is the
optimal discard strategy to ensure all 7 tricks are made?

Tests: (deferred) `ch07-eighty-four-throwaway`

## Slice

**DEFERRED TO WAVE 2.B.2**

The legacy corpus (`gus/data/corpus_train_chunk_*.pt`) was generated with `bid_value=None`
(implicit bid=30). There is no 84-bid auction context.

## Blocker

The legacy corpus does not contain true 84/plunge-bid games. All 100,000 games use
bid=30 (minimum contract). The hand shapes used in 84 analysis are fundamentally different:
- 84 requires winning all 7 tricks
- E[Q] semantics shift to all-or-nothing (42 points or 0)
- The corpus oracle was trained on bid=30 incentives

A placeholder extraction found **22 hands** with ≥ 4 doubles + ≥ 2 non-double trumps in
pip-trump declarations, but these hands played under bid=30 rules. The E[Q] values in these
snapshots reflect bid=30 oracle decisions, not 84-bid strategy.

## N

22 shape-matching snapshots (bid_value=84 as placeholder; actual oracle context is bid=30).
These are NOT valid 84-bid snapshots.

## Claim-Ledger Impact

**`not-yet-tested`** — blocked on corpus with explicit 84 bid context.

## Caveats

- 22 snapshots have `bid_value=84` in the schema field, but the underlying E[Q] values
  were computed under bid=30 incentives.
- DO NOT use these for 84-bid claim validation without re-running oracle under 84 context.
- Wave 2.B.2 must generate or identify a corpus with explicit 84-bid games.

## Command

```bash
python w42/book_validation_v1/wave2/snapshots/eighty_four_throwaway/build_eighty_four_throwaway_corpus.py
```

## Status

`deferred` — Wave 2.B.2 required to generate 84-bid corpus.
