---
title: "Already-clean test 7c: coarse or mixed by how a unit grew — bridge rows, shared inputs and union readers across training (Lem, #231)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). [[lem-coarse-or-mixed-2026-10-20]]
was void: planted controls could not separate coarse from mixed. The essay says why. Whether
something is generalization or superposition depends on "why its things are together in the first
place", more than on telling it apart from a control. So this test reads origin: for each
broadened unit of a narrow net, how did its second concept get there?

Scope: the field net's recipe on pip pours, one position against uniform beliefs, no history,
list v1, signals conditioned on the play count. What we learn here is about this model, not all of
42.

## The review before sealing (two turns, a fresh reader of the essay's local copy)

- **Turn 1** dropped the planted folds as deciding controls. They are edited, not grown: an
  unrelated fold is interference damage no reader learned to decode, and a same-label fold is
  trivially coarse or a merge the net never made. It also dropped uses out as the bar, because a
  single unit's ablation differs between its halves only through downstream context. It pointed
  to the essay's primary test, watching the unit grow.
- **Turn 2** corrected the meaning of "non-local".
  - In the essay it is semantic distance, not timing. A neuron "drifts to neighboring things
    rather than picking up unrelated ones". A nine-step change (umbrella to bee) was local.
  - Locality enters through a bridge: rows that hold both, or shared inputs.
  - "Born coarse" is the thesis's own expectation for a squeezed net, not a neutral case, so the
    earliest steps must be read.
  - Calibrating abruptness on clean poles selects on the outcome, and a share rise forces
    turnover by construction, so abruptness is descriptive only.
  - Packing means nothing reads the union. A union reader with specialists beside it is a coarse
    unit differentiated downstream.

## Pre-registration

- **Runs.** The six formation runs of [[lem-formation-2026-10-17]] (16×8 and 128×64, seeds 0–2).
  Each has 15 in-run checkpoints: steps 0, 30, 100, 300, 1k, 3k and 10k, then the ends of epochs
  1, 2, 3, 5, 10, 20, 40 and 60. Units are tracked by index. The rows are the 434,535 held-out
  rows with list v1, and every checkpoint's signals are conditioned on the play count.
- **Units.** Layer-1 `+` poles that are broadened at epoch 60: their best valid union {a, b} on
  the top 1% beats their best single concept by ≥ 0.05. The reference set is the layer-1 `+`
  poles that are clean at epoch 60. In each pair, `a` is the concept with the larger share of
  the top rows at step 30 and `b` the other.
- **Per checkpoint.** The top 1% rows, and s_a and s_b, the shares of those rows holding a and
  holding b.
- **Born coarse.** At the first checkpoint from step 30 on where max(s_a, s_b) ≥ 0.3, both are
  ≥ 0.3. This is *consistent with the thesis* ("coarse-to-fine wins right away").
- **b's entry**, for poles that are not born coarse. t_b is the first checkpoint where s_b reaches
  half its epoch-60 value. The new b rows N are the top rows at t_b that hold b and were not top
  rows at the checkpoint before.
  - **Bridge share:** the share of N that also holds a.
  - **Input commonality:** the cosine between N's mean input and the mean input of the previous
    checkpoint's top a-rows, both centred on the all-rows mean ("what the neuron's inputs have in
    common").
  - **Reference:** the same input cosine for every epoch-60 clean pole between its newly entering
    rows and its previous top rows, at the same checkpoint interval.
  - **Non-local arrival:** bridge share < 0.1 *and* input commonality below that reference's 5th
    percentile at the same interval.
  - **Sprout:** an entry that is not non-local.
  - **Unresolved:** fewer than 20 new rows, or t_b at the epoch-40 or epoch-60 checkpoint (20-epoch
    intervals cannot resolve an entry). These are reported apart and left out of the
    denominators.
  - Abruptness (top-row Jaccard per step) is reported descriptively.
- **Readers at epoch 60**, read on side A (a, not b) and side B (b, not a). A layer-2 unit i with
  w₂[i, j] > 0 is a **union reader** if it is active on ≥ 0.3 of both sides' rows, and a
  **specialist** if it is active on ≥ 0.3 of one side and < 0.1 of the other.
- **Predictions.**
  1. **Grown or born coarse.** At 16×8, pooled over seeds, at least 0.6 of the resolved broadened
     poles are sprouts or born coarse.
  2. **Few non-local arrivals.** At 16×8, at most 0.15 of the resolved broadened poles are
     non-local arrivals.
  3. **The same principle at width.** At 128×64, at most 0.15 are non-local arrivals.
  4. **Someone reads the union.** At 16×8, at least 0.6 of the broadened poles have at least one
     union reader.
- **What a failed prediction means, sealed.**
  - Failing 1 through "unresolved" is not evidence of mixing.
  - Failing 2 or 3 sends each non-local arrival into the refutation clause as a *candidate*.
  - Failing 4 means the union is read only by specialists, a candidate pattern for packing, still
    subject to the clause.
- **Refutation clause, sealed.** A convincing mixed unit needs all four:
  1. halves unrelated from the net's point of view: no list-v2 concept or reader names the union,
     and the a and b rows share little input (low input commonality);
  2. b's entry is non-local as defined (no bridge rows, no shared inputs);
  3. no layer-2 reader takes both halves positively, and every substantial reader uses one half
     only;
  4. it survives the observer checks: missing inputs or history (#202), the top-5% window, and
     the negative pole.

  Even one such unit is reported as a finding. In a 24-neuron MLP over hand-built features it is
  scoped as possibly the toy-world or bad-learning boundary that the essay concedes, not a
  refutation for large nets.
- **Cost.** 90 forward passes of 434,535 rows (15 checkpoints × 6 runs) on the Mac's CPU through
  `scripts/cpu-slot`.

## Results

Not yet run.

## Links

[[lem-formation-2026-10-17]] · [[lem-coarse-or-mixed-2026-10-20]] · [[lem-family-statistic-2026-10-20]] ·
[[lem-conditioned-audit-2026-10-20]] · [[clean-neurons-42]]
