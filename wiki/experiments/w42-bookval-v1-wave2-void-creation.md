---
title: W42 Book Validation v1 Wave 2 — Void Creation
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: complete
bead: t42-26j8
parent_epic: t42-4zi6
---

## Summary

Probe for [[winning42-ch05-setter-defense]] claim that setters should deliberately
void themselves in an off-suit by leading their last tile from a singleton suit, so
they can later trump in or punish a count tile led to that suit.

Verdict: **contradicted** on this corpus slice.

Oracle-greedy play systematically favors preserving suit over creating a void.
Both p_set delta and EV delta (setter perspective) are negative with 95% CIs that
exclude zero.

## Claim Under Test

From ch05-setter-defense: a setter on lead should recognize when they hold exactly
one tile in an off-suit and deliberately lead it to void themselves, gaining future
trump-in or count-punish opportunities.

See [[winning42-ch05-setter-defense]] for the original book exposition.
See [[w42-bookval-v1-wave2-infra-design]] for campaign methodology.

## Data Slice

| Dimension | Value |
|-----------|-------|
| Corpus | oracle-greedy legacy (`gus/data/corpus_train_chunk_*.pt`) |
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
`decisions[0]`. All Q values are in Team 0 (bidder) perspective; setter metrics
are sign-adjusted.

## Headline Results

| Metric | B-A delta | 95% CI | Direction |
|--------|-----------|--------|-----------|
| p_set delta | -0.0155 | [-0.028, -0.003] | preserve > void |
| EV delta (setter) | -2.63 | [-3.42, -1.84] | preserve > void |
| % contrasts void better (EV) | 32.6% | — | minority |
| CVaR_10 delta | -0.213 | — | preserve safer |

p_set delta is negative: creating the void lowers the probability of the bidder
failing to make contract. The EV delta (setter perspective, negated from Team 0)
is -2.63 points, meaning preserve-suit actions produce ~2.6 more setter-favorable
points on average.

## Slice Breakdown

| Slice | N | p_set delta | EV delta (setter) | % void better |
|-------|---|-------------|-------------------|---------------|
| phase=early (tricks 1-2) | 156 | -0.0165 | -3.79 | 28.8% |
| phase=mid (tricks 3-5) | 120 | -0.0143 | -1.13 | 37.5% |
| count_exposed (bidder holds count in void suit) | 18 | -0.0528 | -3.28 | 22.2% |
| no_count | 258 | -0.0129 | -2.58 | 33.3% |

The negative signal is consistent across phases. The count-exposed sub-group shows
the strongest contradiction (p_set delta = -0.053, only 22% of contrasts favor
void), though N=18 warrants caution.

## Verdict

**contradicted** — on this corpus slice, oracle-greedy play favors preserving suit
over creating a void. The effect is consistent across phases and count-exposure
sub-groups. 95% CIs for both primary metrics exclude zero on the negative side.

Justification: The oracle does not find systematic value in voiding in a suit when
leading a trick in the positions captured here. The "future trump-in" benefit does
not outweigh the immediate cost of leading a low-value singleton tile in these
states.

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
w42/book_validation_v1/wave2/probes/t42-26j8_void_creation/
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

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- The following/discard-position caveat is covered by a sibling probe: `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/`.
- Cheap next probe: rerun at higher bid values (35/42) to test whether the negative void signal holds with less bidder slack — bid_value=30 came from the corpus snapshot conversion, not the probe.
