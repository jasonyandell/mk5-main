---
title: "Already-clean test 2: the width tree — does a narrow net's coarse unit split into a family of fine units in a wider net? (Lem, #195)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
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

Not yet run.

## Links

[[lem-squeeze-2026-10-16]] · [[lem-formation-2026-10-17]] · [[clean-neurons-42]] · [[lem-the-teacher]]
