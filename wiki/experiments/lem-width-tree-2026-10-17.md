---
title: "Already-clean test 2: the width tree — does a narrow net's coarse unit split into a family of fine units in a wider net? (Lem, #195)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[lem-the-teacher]]). This tests the essay's interpretation
directly ([Already Clean](https://already-clean.goodfire.com/), [[clean-neurons-42]]). If a
squeezed net's unit is a coarse merge of a concept family, then in a wider net trained the same
way, that unit should correspond to several fine units of the same family. If it is a mixture of
unrelated concepts, its counterparts should come from unrelated families. The nets are the
squeeze's ([[lem-squeeze-2026-10-16]]), retrained from scratch at each width. Registered before
any unit was matched.

## Pre-registration

- **Nets.** Narrow: the squeeze's 16×8 (seeds 0, 1, 2) and 8×4. Wide: its 128×64 (seed 0) and,
  as the middle rung, 32×16. All are retrained from scratch at their own width on the same data
  by the same recipe, so no wide net was pruned into a narrow one.
- **Rows.** 40,000 rows drawn at random (seed 20261017) from the audit's 434,535 held-out rows,
  split by deal into 20,000 fit and 20,000 eval rows.
- **Units.** Every live pole of every net; a pole's signal is its neuron's pre-activation, negated
  for a `−` pole. A unit's **family** is the family of its sealed-audit label, using the squeeze's
  11 families (Silas labels only; poles labelled by a raw feature are excluded from the family
  counts and reported apart).
- **Match.** For each narrow pole, the wide poles whose signals correlate best with it on the fit
  rows.
  - **Statistic 1, the tree.** The share of a narrow pole's 3 best-matched wide poles that are in
    the narrow pole's own family, against 200 random reassignments of the wide poles' families
    (sizes kept).
  - **Statistic 2, the split.** The narrow pole's signal regressed by least squares on its 5
    best-matched wide poles from its own family, against its 5 best from other families.
    R² is scored on the eval rows.
- **Predictions, before matching.**
  1. The tree: the within-family share of the 3 best matches is at least twice the null mean for
     every narrow net against the 128×64 net.
  2. The split: own-family R² exceeds other-family R² for at least 0.75 of the narrow poles that
     have five same-family wide poles to draw on.
  3. The late-in-the-trick units: of the narrow poles whose best union is `last` ∪ `third`, at
     least half have both a `last`-labelled and a `third`-labelled wide pole among their 5 best
     matches. A coarse unit should split into the two seats.
  4. The middle rung: matched against 32×16, the 8×4 net's within-family share is at least as high
     as against 128×64. A coarse unit has finer counterparts at every wider width.
- **What would break the thesis here.** A within-family share at the null: narrow units
  correspond to unrelated wide units.
- **Cost.** Forward passes of seven nets over 40,000 rows and small least-squares fits. Minutes, one slot call.

## Results

Run 2026-10-17 as registered on the squeeze's nets. Evidence (one JSON per net pair, with every
narrow pole's matches): `lem/width-tree-2026-10-17/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10).

| narrow vs wide | narrow poles (Silas-labelled) | own family in the 3 best matches | random families (mean, 97.5%) | ratio | own-family R² beats other-family R² | mean R² own / other | late units with both seats in their 5 best |
|---|---:|---:|---|---:|---:|---|---:|
| 16×8 s0 vs 128×64 | 31 | **0.516** | 0.217, 0.302 | 2.38 | 0.35 (of 31) | 0.459 / 0.535 | 1 of 9 |
| 16×8 s1 vs 128×64 | 23 | **0.565** | 0.232, 0.333 | 2.44 | 0.57 (of 23) | 0.542 / 0.533 | 0 of 4 |
| 16×8 s2 vs 128×64 | 30 | **0.533** | 0.210, 0.300 | 2.54 | 0.40 (of 30) | 0.539 / 0.574 | 1 of 7 |
| 8×4 vs 128×64 | 19 | **0.544** | 0.186, 0.281 | 2.92 | 0.21 (of 19) | 0.486 / 0.574 | 1 of 3 |
| 8×4 vs 32×16 | 19 | 0.439 | 0.209, 0.316 | 2.10 | 0.35 (of 17) | 0.458 / 0.505 | 0 of 3 |

**The predictions, graded.**

| | predicted | measured | |
|---|---|---|---|
| 1. the tree: own family in the 3 best matches ≥ 2× the null, every narrow net vs 128×64 | | 2.38, 2.44, 2.54, 2.92 (all above the null's 97.5%) | holds |
| 2. the split: own-family R² beats other-family R² for ≥ 0.75 of narrow poles | | 0.21–0.57 | **wrong** |
| 3. late units: both a `last` and a `third` wide unit in the 5 best, for ≥ half | | 3 of 23 | **wrong** |
| 4. 8×4 against 32×16 at least as high as against 128×64 | | 0.439 vs 0.544 | wrong |

**Reading.**
- **The tree is there.** A narrow unit's closest counterparts in a wide net, which was trained
  from scratch and never pruned from it, come from the narrow unit's own concept family 2.4–2.9
  times as often as random families give. Narrow units are not mixtures of unrelated wide units.
  This is the essay's interpretation, measured directly, and it holds for every narrow net
  against the 128×64 net.
- **It splits along facets, not siblings.** The 128×64 net has 47 units labelled `last` and 6
  labelled `third`. A narrow "late in the trick" unit's best matches are several wide `last` units,
  sometimes with `partner-winning`, `can-take-it` or `holds-master` beside them. So the wide net
  does not split the coarse unit into its two seats. It spreads `last` into many units, one per
  facet of the last seat's decision. That is closer to the essay's offshoots (one concept growing
  facets) than to the seat split this page predicted.
- **The split regression did not favour the family, and the test was unfair.** Five units from
  the other ten families, chosen from a pool about four times larger, fit a narrow unit as well as
  or better than five from its own family. The R² means are close (about 0.5 each). The larger
  pool and concepts that co-occur across families (last seat, partner winning, can take it) both
  favour the other-family side. A fair version matches pool sizes. That belongs to #200, the
  sharper statistic, and should be registered before it runs.
- **The middle rung is weaker** (2.10 against 32×16, against 2.92 against 128×64): the 32×16 net has
  only 54 Silas-labelled units to match against.

Caveats: one wide seed; correlation matching on pre-activations; labels from the v1 list.

## Links

[[lem-squeeze-2026-10-16]] · [[lem-formation-2026-10-17]] · [[clean-neurons-42]] · [[lem-the-teacher]]
