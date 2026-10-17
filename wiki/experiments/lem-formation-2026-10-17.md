---
title: "Already-clean test 1: watch the field net's units form across training (Lem, #194)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[lem-the-teacher]]). Cammarata's central evidence is
formation: neurons watched across checkpoints, each forming around one concept and then sprouting
offshoots along that concept's facets ("generalization is fractal";
[Already Clean](https://already-clean.goodfire.com/), [[clean-neurons-42]]). The squeeze found
Zeb's narrow nets cleaner than its wide ones ([[lem-squeeze-2026-10-16]]). This test watches the
same nets form. Registered before any checkpoint was taken.

## Question

When does each of the field net's clean units form, in what order, and how does its concept
change as it forms? Recursive generalization predicts that a unit keeps its concept family as it
grows. Superposition, or units that jump between unrelated concepts, predicts relabels across
families.

## Pre-registration

- **Runs.** The squeeze's recipe exactly: the five pip pours (250,000 deals), margin target, MLX,
  60 epochs. Widths **16×8** and **128×64**, seeds **0, 1, 2**: six runs from one dataset build.
  Checkpoints come from inside each run (`scenario.field_train.train(checkpoints=...)`, steps of
  the optimizer, 0 being the initial net). There are 15 checkpoints a run, roughly log-spaced:
  steps 0, 30, 100, 300, 1,000, 3,000 and 10,000, then the ends of epochs 1, 2, 3, 5, 10, 20,
  40 and 60. One epoch is about 15,300 steps.
- **Per checkpoint.** The sealed audit, on the same 434,535 held-out rows and the 48 v1 labels.
  The 20 label shuffles run at the final checkpoint only, since chance has been 0.000 in every
  audit so far. Also regret under the teacher's plan on those rows, from the teacher values G7
  collected ([[lem-fidelity-by-concept-2026-10-16]]).
- **Units are tracked by index within a run**, as the same neuron over time. Each pole gets:
  - its **first clean step**, the earliest checkpoint from which it is clean at every later one;
  - its **first label**, the label at that step;
  - its **final label**, the label at epoch 60.

  Families are the squeeze's 11 (`lem.probe.squeeze.FAMILIES`).
- **Predictions, before training.**
  1. **Early.** By the end of epoch 1 the clean share is at least 0.8 of its epoch-60 value, at
     both widths and in every seed.
  2. **Order.** Units with a position or trick-state label (`lead`, `last`, `partner-winning`,
     `winner-cannot-be-beaten`, ...) reach their first clean step before units with a count or
     mastery label. This is measured as the median first clean step by family, over the three
     seeds pooled, at 128×64.
  3. **Recursive generalization.** Of the poles clean at epoch 60, at least 0.7 have a first label
     in the same family as their final label. The null is how often two labels drawn independently
     from the epoch-60 clean labels share a family.
  4. **Broadening grows.** The 16×8 "late in the trick" units (best union `last` ∪ `third`) are
     single-label clean (`last` or `third`) earlier than they are broadened: the union's gain over
     the best single label rises across checkpoints.
  5. **Determinism.** The 16×8 seed-0 run's epoch-60 net is bit-identical to the squeeze's
     `field_h16x8_s0`. Checkpointing does not change a run (tested), and the recipe is the same.
  6. **A trump-to-master chain** (exploratory, counted not graded): at 128×64, at least one pole
     whose first label is a trump-held concept (`holds-boss-trump`, `holds-sure-trump`) and whose
     final label is a mastery concept (`holds-master`, `masters-2plus`, `can-take-it`). This is
     the epic's "trump double → highest unplayed trump → master of any suit".
- **What would break the thesis here.** Under prediction 3, final labels as often outside their
  first family as the null gives: units that wander between unrelated concepts as they form.
- **Cost.** The dataset build is about six minutes; the runs are about 10 minutes each at 60
  epochs, so about an hour on the Mac's GPU. There are 90 audits at about half a minute each
  without shuffles. One slot call trains all six runs, with the reason written in the runner:
  one dataset build serves every run.

## Results

Run 2026-10-17 as registered: six runs, 15 checkpoints each, 90 audits. Validation regret at
epoch 60 is 0.0130, 0.0134 and 0.0130 at 16×8, and 0.0081, 0.0084 and 0.0083 at 128×64. Evidence
(every checkpoint net, its audit, `formation.json`, runners, logs, `SHA256SUMS`):
`lem/formation-2026-10-17/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10).

**Clean share across training** (selected checkpoints; one epoch is 15,270 steps):

| run | step 0 (untrained) | 1,000 | 10,000 | epoch 1 | epoch 2 | epoch 60 | regret: step 0 → epoch 1 → 60 |
|---|---:|---:|---:|---:|---:|---:|---|
| 128×64 s0 | 0.195 | 0.276 | 0.424 | 0.447 | 0.489 | 0.610 | 0.0423 → 0.0102 → 0.0081 |
| 128×64 s1 | 0.148 | 0.275 | 0.451 | 0.466 | 0.518 | 0.654 | 0.0421 → 0.0102 → 0.0084 |
| 128×64 s2 | 0.161 | 0.279 | 0.458 | 0.462 | 0.501 | 0.636 | 0.0437 → 0.0103 → 0.0083 |
| 16×8 s0 | 0.188 | 0.289 | 0.533 | 0.578 | 0.553 | 0.833 | 0.0432 → 0.0161 → 0.0130 |
| 16×8 s1 | 0.283 | 0.357 | 0.575 | 0.600 | 0.756 | 0.652 | 0.0424 → 0.0162 → 0.0134 |
| 16×8 s2 | 0.208 | 0.340 | 0.581 | 0.614 | 0.644 | 0.792 | 0.0401 → 0.0153 → 0.0130 |

**A new baseline, and it matters for the thesis.** An untrained net already scores a clean share
of 0.15–0.28: 57–75 of 384 poles at 128×64, most labelled by a raw feature (30–51) and some by a
Silas concept (24–27). A random unit with a large weight on one input bit fires hardest exactly
when that bit is on. The shuffled-label chance (0.000) does not see this, because it breaks the
labels, not the units. **The untrained net is the honest floor for "clean".** Training lifts the
share from about 0.2 to 0.61–0.83, and lifts the Silas-labelled poles from about 25 to about 133
at 128×64. The trained nets stay far above the floor. Every earlier clean share should now be
read against it (#193, #200).

**First clean step by family** (128×64, three seeds pooled; median step, poles):

| family | median first clean step | poles |
|---|---:|---:|
| time | 6,500 | 6 |
| position | 61,080 | 154 |
| trick | 76,350 | 33 |
| trump state | 76,350 | 50 |
| trump held | 152,700 | 130 |
| count | 152,700 | 9 |
| mastery | 152,700 | 13 |
| side | 305,400 | 3 |
| doubles | 763,500 | 2 |

**Family kept from first clean label to final label:**

| width | poles (Silas both ends) | family kept | same label | null (independent draws) |
|---|---:|---:|---:|---:|
| 128×64, pooled | 355 | **0.820** | 0.806 | 0.279 |
| 16×8, pooled | 58 | **0.741** | 0.724 | 0.311 |

By run: 0.800, 0.814 and 0.846 at 128×64; 0.840, 0.692 and 0.650 at 16×8.

**The predictions, graded.**

| | predicted | measured | |
|---|---|---|---|
| 1. by epoch 1, clean share ≥ 0.8 of its epoch-60 value, every run | | 0.73, 0.71, 0.73, 0.69, 0.92, 0.78 | **wrong**: about three quarters by epoch 1; the rest forms over the next 59 epochs |
| 2. position and trick units clean before count and mastery units | | position 61k and trick 76k steps vs count and mastery 153k | holds |
| 3. ≥ 0.7 of final clean units keep their first family | | 0.820 and 0.741 pooled, against nulls of 0.28 and 0.31; 4 of 6 runs individually | holds pooled; two 16×8 runs at 0.69 and 0.65 |
| 4. late-in-the-trick units single-clean before broadened | | the measure is ill-posed: at step 0 any unit's union of `last` and `third` beats either alone by > 0.05 | not graded; see below |
| 5. the 16×8 seed-0 net is bit-identical to the squeeze's | | identical | holds |
| 6. a trump-held → mastery chain (exploratory) | ≥ 1 | 0 in all six runs | none seen |

**Exploratory, not registered: coarse first, then fine.** Prediction 4 cannot be graded, so here is
one reading of the same 20 late-in-the-trick units. It compares the checkpoint at which the union
`last` ∪ `third` first reaches the 0.9 clean bar with the one at which a single label does. The
union gets there first for 14 units and at the same checkpoint for 6; it never comes second. These
units form coarse ("late in the trick") and some sharpen to one seat later. The essay describes
growth from a seed concept outward ("iy" to "diy" to "do it yourself"). Here the direction is the
reverse, coarse to fine, while the family is kept throughout.

**Reading.**
- **Units keep their concept family as they form.** 0.82 of the wide net's units end in the
  family they first became clean in, against 0.28 for independent draws. The net does not assign
  a unit one concept and then reassign it to an unrelated one. This is the essay's
  recursive-generalization signature, and the first formation evidence for the core thesis
  (#193) on 42.
- **The trick in front of you forms first:** time and position, then the trick's state and
  trump state, then trump held, count and mastery, and last the side and doubles. That is the
  narrow nets' surviving vocabulary ([[lem-squeeze-2026-10-16]]) in order of formation: what
  forms first is what survives the squeeze.
- **No chain from trump to mastery.** The epic's candidate "trump double → highest unplayed trump
  → master of any suit" did not appear as one unit relabelling. Mastery units start as mastery.
- **The untrained floor (0.15–0.28) belongs in every clean share from now on.**

Caveats: units are tracked by index, so a unit that changes concept counts as one unit; labels are
the 48-concept v1 list; three seeds per width; regret is to the finite teacher.

## Links

[[lem-squeeze-2026-10-16]] · [[clean-neurons-42]] · [[lem-the-teacher]] ·
[[lem-field-net-audit-2026-10-15]] · [[strategy-fusion-fence]]
