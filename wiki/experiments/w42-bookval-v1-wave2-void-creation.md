---
title: W42 Book Validation v1 Wave 2 — Void Creation
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-10-12
status: complete
bead: t42-26j8
parent_epic: t42-4zi6
---

## Summary

Probe for [[winning42-ch05-setter-defense]] claim that setters should deliberately
void themselves in an off-suit by leading their last tile from a singleton suit, so
they can later trump in or punish a count tile led to that suit.

Verdict: **withdrawn pending a sign check (#122)**. The probe script treats E[Q] as Team 0's view and
negates it for the setter; forge documents E[Q] from the acting player's view
(`forge/eq/generate/actions.py:113,160`). If the sign is inverted, the recorded
−2.63 is +2.63 for the void lead. Until the sign is checked on data, this page
carries no verdict about the book.

## Claim Under Test

From ch05-setter-defense: a setter on lead should recognize when they hold exactly
one tile in an off-suit and deliberately lead it to void themselves, gaining future
trump-in or count-punish opportunities.

See [[winning42-ch05-setter-defense]] for the original book exposition.
See [[w42-bookval-v1-wave2-infra-design]] for campaign methodology.

## Data Slice

| Dimension | Value |
|-----------|-------|
| Corpus | oracle-greedy legacy (`gus/data/corpus_train_chunk_*.pt`, on HF as flat basenames — [tree](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus/tree/main); see [[huggingface-assets]]) |
| Role | setter (player 1 or 3; bidder fixed at player 0) |
| Position | on lead (trick_plays empty), trick >= 1 |
| Filter | setter holds >= 2 distinct suits, >= 1 singleton suit |
| Declarations | all 10 (pip 0-6, doubles-trump, doubles-suit, no-trump) |
| Bid value | 30 (fixed by legacy corpus) |
| Input snapshots | 500 |
| Valid contrasts produced | 276 |

## Methodology

For each snapshot, a paired contrast is formed:

- **A (preserve suit)**: lead the lowest-sum tile from the most-populated suit
  (multi-tile, non-trump, non-void-suit).
- **B (create void)**: lead the last tile from the smallest singleton off-suit,
  voiding the setter in that suit.

`generate_eq_from_snapshots` is called independently per action (100 world samples,
MRV sampler, greedy continuation). Q-PDFs are read at the action slot from
`decisions[0]`. The script assumes Q values are in Team 0 (bidder)
perspective and negates them for setter metrics; forge documents them as
acting-player perspective, and the setter is the acting player (#122). Labels
come from the learned Q network (qMAE 0.94 points), in points, bid 30. `p_set`
uses Q ≥ 30 on the ±42 differential scale, where the make line is Q ≥ 18, so
it is partly saturated. The void tile and the preserve tile also differ in
identity (count, pips), which confounds the contrast.

## Headline Results

| Metric | B-A delta | 95% CI | Direction |
|--------|-----------|--------|-----------|
| p_set delta | -0.0155 | [-0.028, -0.003] | preserve > void |
| EV delta (setter) | -2.63 | [-3.42, -1.84] | preserve > void |
| % contrasts void better (EV) | 32.6% | — | minority |
| CVaR_10 delta | -0.213 | — | preserve safer |

The deltas are as recorded under the script's sign convention, which is
unverified (#122).

## Slice Breakdown

| Slice | N | p_set delta | EV delta (setter) | % void better |
|-------|---|-------------|-------------------|---------------|
| phase=early (tricks 1-2) | 156 | -0.0165 | -3.79 | 28.8% |
| phase=mid (tricks 3-5) | 120 | -0.0143 | -1.13 | 37.5% |
| count_exposed (bidder holds count in void suit) | 18 | -0.0528 | -3.28 | 22.2% |
| no_count | 258 | -0.0129 | -2.58 | 33.3% |

The sign of every row depends on the same unverified convention. The
count-exposed sub-group has N=18.

## Verdict

**withdrawn pending a sign check (#122)**. The recorded "contradicted" rested on the
setter-side sign; if the sign is inverted the same data favour the void lead.
Either way the test is a two-action contrast in oracle points with a
perfect-information evaluator, on positions from the April Gus v2 corpus later
found 27–67% malformed ([[otis-phase-r]]), with CIs that do not cluster by
deal. No ledger row was ever registered for it.

## Caveats

1. **Leading position only.** All snapshots show a setter leading a trick (they
   won the prior one). The book's primary void scenario may be a following
   (off-suit discard) position, which requires a different snapshot corpus.

2. **Oracle-greedy baseline.** Greedy oracle continuation may have already adapted
   its subsequent play to not rely on the void. The counterfactual comparison is
   first-move only; downstream play quality is matched by the oracle.

3. **Singleton-only mining.** The filter requires exactly 1 tile in the voided
   suit. Two-tile suits played down to one (partial voiding) are not captured.

4. **Bid value = 30 only.** p_set metrics use bid_value=30 throughout. Higher
   bids are not tested.

5. **Late-game absent.** No trick-6/7 samples exist in the count (most positions
   are early/mid).

6. **Single-corpus provenance.** Oracle-greedy play may not reflect human play
   patterns that motivate the book strategy.

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-26j8_void_creation/   # git-tracked receipts
├── README.md
├── manifest.json
├── summary.json
├── paired_contrasts.csv      (276 rows)
├── slice_breakdown.csv       (5 slice rows)
└── run_void_creation_probe.py
```

## Links

- [[winning42-ch05-setter-defense]] — source book chapter
- [[w42-bookval-v1-wave2-infra-design]] — wave 2 campaign design
- [[w42]] — project entity page
