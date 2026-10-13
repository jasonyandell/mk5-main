---
kind: experiment
title: "W42 Book Validation v1 — Wave 2 — Void Creation Follow Position"
bead: t42-z31l
parent_bead: t42-4zi6
wave: 2
status: complete
claim: ch05-void-creation (follow-position sub-claim)
verdict: withdrawn pending sign check (#122)
date: 2026-05-03
first_seen: 2026-05-03
last_updated: 2026-10-12
---

# W42 Book Validation v1 Wave 2 — Void Creation Follow Position

## Background

[[winning42-ch05-setter-defense]] describes the setter's void-creation strategy: when
you cannot follow a non-trump lead and hold the last tile of some non-led suit, you
should discard that tile to void yourself in that suit. The future payoff is that when
an opponent later leads to that suit, you can trump in.

Wave 2.C (bead t42-z31l's predecessor probe [[w42-bookval-v1-wave2-void-creation]])
tested the *lead-position* variant of void creation — where the setter is on lead and
chooses to expose their last tile in a suit (N=276, recorded EV delta −2.63). Its
verdict is withdrawn pending the same sign check as this page's (#122).

The wave 2.C.2 probe (this bead) tests the *follow-position* variant: the canonical
scenario from the book. Here the setter is **following** a non-trump trick they cannot
follow, and must discard. The contrast is whether to discard the singleton tile (creating
a void) vs. discarding a tile from a different multi-tile suit.

## Slice

- **Corpus:** oracle-greedy legacy corpus (3 of 100 chunks, bid=30 throughout)
- **Position:** setter following a non-trump trick, unable to follow led suit
- **Condition:** setter holds exactly 1 tile in some non-led, non-trump suit
- **Contrast A (preserve):** discard from a different held suit (keeps singleton alive)
- **Contrast B (void):** discard the singleton (voids that suit)

## Results

**N = 500 paired contrasts** (100 world samples per branch, MPS device)

| Metric | Mean | 95% CI | Direction |
|--------|------|--------|-----------|
| EV delta (setter) | **+0.77** | [+0.12, +1.42] | void > preserve |
| p_set delta | +0.0087 | [−0.0019, +0.0192] | void > preserve (CI straddles 0) |
| CVaR_10 delta | +0.17 | — | void > preserve |
| % contrasts favoring void | 52.6% | — | slightly above chance |

### Slice breakdown

| Slice | N | EV delta (setter) | p_set delta | % void better |
|-------|---|-------------------|-------------|---------------|
| phase=early | 208 | +0.37 | +0.0067 | 52.9% |
| phase=mid | 292 | +1.05 | +0.0102 | 52.4% |
| count_exposed=True | 30 | +0.46 | +0.0633 | 43.3% |
| count_exposed=False | 470 | +0.79 | +0.0052 | 53.2% |
| bidder_winning_trick | 326 | +0.79 | +0.0051 | 52.8% |
| setter_winning_trick | 174 | +0.72 | +0.0155 | 52.3% |

## Verdict: withdrawn pending sign check (#122)

The probe script negates E[Q] for the setter on the assumption that it is Team 0's
view (`run_void_creation_follow_probe.py:411-413`); forge documents E[Q] from the
acting player's view (`forge/eq/generate/actions.py:113,160`). If the sign is
inverted, the recorded +0.77 is −0.77 and the book's direction is weakly
contradicted. The verdict is withdrawn until the sign is checked on data.

Independent of the sign, the result is not evidence for the book:

1. +0.77 points is below the ≈0.93-point minimum detectable effect, the sign test
   gives p=0.26, and 52.6% of contrasts favour void.
2. On the clean subset (no count, no trump) the delta is +0.08 [−0.37, +0.53].
3. The p_set CI straddles zero, and `p_set` uses Q ≥ 30 on the ±42 scale where the
   make line is Q ≥ 18, so it is partly saturated.
4. When all legal actions are ranked, the EV argmax is the void tile on only 29.4% of
   these snapshots ([[w42-bookval-v3-utility-argmax-divergence]], which maximizes raw
   e_q and does not share the sign question).
5. Labels come from the learned Q network (qMAE 0.94 points), in points, at bid 30, on
   the April corpus later found 27–67% malformed ([[otis-phase-r]]); CIs do not
   cluster by deal.

The earlier reading of the lead and follow probes as a "position-dependent reversal"
is withdrawn with both verdicts.

## Scope caveat

This probe only tests bid=30 from the oracle-greedy corpus. The book's void-creation
advice was never tested at higher bids.

## Artifacts

| Path | Description |
|------|-------------|
| `w42/book_validation_v1/wave2/snapshots/void_creation_follow/snapshots.jsonl` | 500 snapshots |
| `w42/book_validation_v1/wave2/snapshots/void_creation_follow/manifest.json` | Mining provenance |
| `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/paired_contrasts.csv` | 500 paired rows |
| `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/slice_breakdown.csv` | Slice aggregates |
| `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/summary.json` | Headline numbers |

## Related pages

- [[w42-bookval-v1-wave2-void-creation]] — lead-position probe (N=276; withdrawn pending #122)
- [[winning42-ch05-setter-defense]] — source chapter

## Reproducibility

```bash
# Part 1: mine corpus
python w42/book_validation_v1/wave2/snapshots/void_creation_follow/build_void_creation_follow_corpus.py

# Part 2: run probe
python w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/run_void_creation_follow_probe.py \
  --snapshots w42/book_validation_v1/wave2/snapshots/void_creation_follow/snapshots.jsonl \
  --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
  --samples 100 \
  --device mps \
  --output-dir w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/
```
