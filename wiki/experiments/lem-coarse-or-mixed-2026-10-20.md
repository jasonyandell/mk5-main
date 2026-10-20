---
title: "Already-clean test 7b: coarse or mixed, by the essay's own discriminator — routes in, uses out, against planted folds (Lem, #229)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). [[lem-family-statistic-2026-10-20]]
showed that our family statistic cannot tell trained units from random ones. This test asks the
squeeze's question the way the essay does. Is a narrow net's broadened unit a merge of things the
net treats alike (coarse), or a fold of unrelated things that do not interfere (mixed)?

Scope: the field net's recipe on pip pours, one position against uniform beliefs, no history,
concept list v1, signals conditioned on the play count.

## The review before sealing (Jason's rule, 2026-10-20)

Two turns by a fresh-context reviewer who read Cammarata's essay itself. The quotes below were
checked against the page.

- **Turn 1** found that the v1 draft tested a straw version.
  - Our hand-made concept families are not the essay's "related", which is "from the network's
    point of view".
  - A co-occurrence-weighted null subtracted the essay's own mechanism.
  - A consequence clause made the thesis rest on a statistic the essay never used.
  - Families are now descriptive only.
- **Turn 2** found that the v2 positive control was circular: the pair was picked by the measured
  quantity. It also found that an input-route split alone is not mixing.
  - The essay's offshoot "arrives by its own route" and is still generalization.
  - The essay reads both sides: "upstream at how it's built and downstream at what it does, and
    both line up."
  - So mixed requires disjoint routes in **and** distinct use out.
  - The reviewer also asked for a random-split ceiling, a sealed tail share instead of medians,
    a two-sided untrained guard, and the same observer clause on every consequence.
- **One departure from the review.** Co-occurrence lift is not the bar. Related coarse merges such
  as `late` = `third` ∪ `last` are mutually exclusive.

**A note on the review, added 2026-10-20 after the run.** Both review turns read the essay
through WebFetch, which returns a small model's summary in slices, not the text. So this review
did not read the paper itself. The quotes used here were later checked against the local verbatim
copy, and all of them appear exactly. The test was void by its own rule regardless, so nothing it
concluded rests on the review. [[lem-unit-origin-2026-10-20]] was reviewed on the full text.

## Pre-registration

- **Nets.** The drop-out nets ([[lem-dropout-order-2026-10-17]]): trained and untrained at 16×8 and
  8×4 (three seeds each) for the main arm, and the trained 32×16 nets as extra fold donors. The
  sample is the 434,535-row pip audit sample, with list v1.
- **Units read: layer 1 only.** A layer-2 unit's effect on the tile scores is its activation times
  one fixed weight vector, so its downstream direction is the same on every row. The
  downstream-use measure is meaningful only for layer 1.
- **Broadened pole.** A live `+` pole of a layer-1 unit whose best valid two-concept union {a, b}
  beats its best single concept by ≥ 0.05 in precision on the top 1% of its conditioned signal.
  Its top set is split into A (a holds, b does not) and B (b holds, a does not). Both need at least
  30 rows.
- **Route in.** The mean over the side's rows of the unit's input contribution x ⊙ w₁[:, j],
  centred on its mean over all rows. `r` is the cosine between side A's vector and side B's.
- **Use out.** The mean over the side's rows of the change in the 28 tile scores when the unit is
  ablated (after its ReLU, recomputed downstream). `d` is the cosine between A's and B's.
- **Ceiling.** The same two cosines with the A/B assignment shuffled within A ∪ B, sizes kept,
  median of 20 shuffles. This is the shared-direction bias every split carries.
- **Planted folds, the positive controls.** In a copy of a trained net, unit k is merged into unit
  j: w₁[:, j] += w₁[:, k], b₁[j] += b₁[k], w₂[:, j] += w₂[:, k], and unit k is removed. The merged
  unit is a real unit with its own ReLU. The instrument has to find its routes and uses through
  those weights. Pairs are picked by their labels only, never by a measured quantity:
  - **Mixed fold**: two clean `+` layer-1 poles whose labels a and b have co-occurrence lift
    P(a∧b)/(P(a)P(b)) < 0.5 and are not values of one variable. That is unrelated and
    non-interfering by construction.
  - **Coarse fold**: two clean `+` layer-1 poles whose labels are values of one variable, either
    the mover's seat (`lead`, `second`, `third`, `last`) or the declaration (`trump-is-*`, list v2
    only, so this control reads v2 labels for its split). That is one concept the list splits.

  For a fold, a and b are the two donors' labels. Up to 40 folds of each kind, seed 20261020.
- **Mixed-like.** A pole is mixed-like when both `r` and `d` are below the mixed folds' 75th
  percentile.
- **Predictions.**
  1. **The controls separate.** The mixed folds' median `r` and median `d` are both below the
     coarse folds' 25th percentile. If not, the instrument is void and nothing is concluded about
     coarse or mixed.
  2. **Narrow units are not mixed.** Pooled over 16×8 and 8×4 trained nets, the share of broadened
     layer-1 poles that are mixed-like is below 0.15.
  3. **The two-sided untrained guard.** Report the untrained twins' mixed-like share.
     - If trained and untrained are both below 0.15 and within 0.05 of each other, prediction 2 is
       void as evidence, because the instrument cannot tell learned from random.
     - Trained well above untrained would suggest learned mixing.
  4. **The known coarse merge.** If at least 5 trained broadened poles have the union {`third`,
     `last`} under conditioning, at most 0.2 of them are mixed-like. If fewer than 5, this
     prediction is void.
- **Consequence, sealed.** A mixed-like pole is a *candidate* for mixing, never a finding of it,
  until three observer explanations are ruled out and recorded per pole:
  - a missing concept: does the union read as one concept that list v2 or a reader names?
  - missing inputs or history (#202);
  - the percentile window: does it hold at the top 5%?
- **What would count against the thesis here, sealed.** Narrow broadened poles that are
  mixed-like on both routes in and uses out, in a share above 0.15 and clearly above the untrained
  twins'. Each must also survive the three observer checks. A disjoint input route alone, with
  shared use out, is an offshoot in the essay's sense, not mixing.
- **Cost.** About 40 forward passes per net for ablations and folds, under a minute each, on
  the Mac's CPU through `scripts/cpu-slot`.

## Results

Run 2026-10-20 as registered. Evidence is `lem/coarse-or-mixed-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/9583f00503a38c88fa8e6fe752429512a33ad32c/lem/coarse-or-mixed-2026-10-20):
one JSON per net with every broadened pole and every fold, plus `grade.py`, `grade.txt` and
`grade.json`. The donor nets were the trained 32×16, 16×8 and 8×4 nets. There were 27 mixed folds
and 26 coarse folds with both sides of at least 30 rows.

**The controls, prediction 1.**

| | `r` (routes in): median [q25, q75] | `d` (uses out): median [q25, q75] |
|---|---|---|
| mixed folds (27) | 0.199 [−0.107, 0.488] | 0.777 [0.457, 0.876] |
| coarse folds (26) | 0.363 [−0.015, 0.550] | 0.930 [0.633, 0.973] |

**Prediction 1 is wrong.** The mixed folds' medians are not below the coarse folds' 25th
percentiles (0.199 against −0.015; 0.777 against 0.633). **By the sealed rule the instrument is
void, and nothing is concluded about coarse or mixed.** Predictions 2–4 are not graded as
evidence. Their numbers are recorded below for the next instrument only.

**Why the controls overlap.** Routes in do not separate the controls, and the essay says why: an
offshoot can arrive "by its own route". The coarse folds join mutually exclusive values of one
variable: `lead` and `last`, `trump-is-fives` and `trump-is-fours`. The input features that say
"I lead" and "I play last" are different features by construction, so even a merge of one concept
has disjoint routes in. Uses out does separate the controls in the expected direction (0.930
against 0.777), but not past the sealed bar. The random-split ceilings are about 1.0 for both
measures, so any real split reads below its ceiling.

**Recorded, not read** (for the next instrument):
- the 37 trained broadened layer-1 poles at 16×8 and 8×4 have medians `r` 0.369 and `d` 0.943;
- 0.216 of them fall in the mixed-like region as defined, and so do 0.000 of the 70 untrained;
- there are 4 {`third`, `last`} poles, none mixed-like.

**Next.** Uses out as the primary measure, read more sharply (on the legal tiles, or the chosen
move's probability), with more folds and a threshold sealed from the controls: #231. It will be
reviewed against the essay copy before sealing.

Scope: the field net's recipe on pip pours, list v1, layer-1 units, signals conditioned on the
play count.

## Links

[[lem-family-statistic-2026-10-20]] · [[lem-squeeze-conditioned-2026-10-20]] · [[lem-dropout-order-2026-10-17]] ·
[[lem-causal-2026-10-17]] · [[clean-neurons-42]]
