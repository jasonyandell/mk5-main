---
kind: experiment
title: "W42 Book Validation v1 — Wave 2 — Void Creation Follow Position"
bead: t42-z31l
parent_bead: t42-4zi6
wave: 2
status: complete
claim: ch05-void-creation (follow-position sub-claim)
verdict: context-limited
date: 2026-05-03
first_seen: 2026-05-03
last_updated: 2026-07-11
---

# W42 Book Validation v1 Wave 2 — Void Creation Follow Position

## Background

[[winning42-ch05-setter-defense]] describes the setter's void-creation strategy: when
you cannot follow a non-trump lead and hold the last tile of some non-led suit, you
should discard that tile to void yourself in that suit. The future payoff is that when
an opponent later leads to that suit, you can trump in.

Wave 2.C (bead t42-z31l's predecessor probe [[w42-bookval-v1-wave2-void-creation]])
tested the *lead-position* variant of void creation — where the setter is on lead and
chooses to expose their last tile in a suit. That probe found the claim **contradicted**
at N=276: the oracle consistently prefers preserving the suit over voiding it from the
lead position (mean EV delta = −2.63 from setter perspective, CI entirely negative).

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

## Verdict: context-limited

The EV CI for the follow-position scenario ([+0.12, +1.42]) **excludes zero on the
positive side**, pointing weakly in the direction the book predicts. This is a meaningful
difference from the lead-position probe where the CI was entirely negative. However:

1. The p_set CI straddles zero (−0.0019 to +0.0192) — the setting-probability signal
   is not significant.
2. Only 52.6% of contrasts favor void creation — barely above chance.
3. The absolute EV advantage (+0.77) is modest compared to the strong contradiction in
   the lead-position probe (−2.63).
4. The count_exposed sub-slice (N=30) shows the largest p_set delta (+0.063) but is
   underpowered for a sub-slice verdict.

**The follow-position void scenario shows a small, real positive signal — the oracle
agrees the book direction is marginally correct here — but the effect is too noisy and
too small to endorse as "supported" across the bid=30 corpus.**

## Comparison to Lead-Position Probe

| Dimension | Lead-position (t42-26j8) | Follow-position (t42-z31l) |
|-----------|--------------------------|----------------------------|
| N pairs | 276 | 500 |
| EV delta (setter) | **−2.63** (CI: [−3.42, −1.84]) | **+0.77** (CI: [+0.12, +1.42]) |
| p_set delta | −0.0155 (CI: [−0.028, −0.003]) | +0.0087 (CI: [−0.002, +0.019]) |
| % void better | 32.6% | 52.6% |
| Verdict | **contradicted** | **context-limited** |

The lead-position void creation claim is robustly contradicted. The follow-position
claim is the book's canonical scenario, and the oracle weakly supports the direction
but the effect is small enough that the verdict is context-limited rather than supported.

## Interpretation

The mechanism difference matters: when the setter is on **lead**, discarding their
singleton creates a void but also announces their weakness and may invite opponents to
lead into them before the void pays off. When the setter is **following** a trick they
cannot follow, the discard is forced anyway — the only choice is *which* suit to deplete.
The positive signal here suggests the oracle finds mild truth in the book's advice: among
forced discards, voiding a singleton suit is marginally better than depleting a
multi-tile suit. But the effect is small because the oracle trades away both options
at near-equal value in most positions.

## Scope caveat

This probe only tests bid=30 from the oracle-greedy corpus. The book's void-creation
advice is likely more compelling in high-bid games (35+) where the contract margin is
tighter and opponent void exploitation is more decisive. Wave 2.E.2 (high-bid testing)
would be the natural follow-on.

## Artifacts

| Path | Description |
|------|-------------|
| `w42/book_validation_v1/wave2/snapshots/void_creation_follow/snapshots.jsonl` | 500 snapshots |
| `w42/book_validation_v1/wave2/snapshots/void_creation_follow/manifest.json` | Mining provenance |
| `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/paired_contrasts.csv` | 500 paired rows |
| `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/slice_breakdown.csv` | Slice aggregates |
| `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/summary.json` | Headline numbers |

## Related pages

- [[w42-bookval-v1-wave2-void-creation]] — lead-position probe (contradicted, N=276)
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
