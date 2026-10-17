---
title: "Already-clean test 1: watch the field net's units form across training (Lem, #194)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
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

Not yet run (waits on the trainer's checkpoint hook, a gated branch with the Mayor).

## Links

[[lem-squeeze-2026-10-16]] · [[clean-neurons-42]] · [[lem-the-teacher]] ·
[[lem-field-net-audit-2026-10-15]] · [[strategy-fusion-fence]]
