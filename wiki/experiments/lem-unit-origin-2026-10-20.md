---
title: "Already-clean test 7c: coarse or mixed by how a unit grew — bridge rows, shared inputs and union readers across training (Lem, #231)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
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

Run 2026-10-20 as registered, on all six runs. Evidence is `lem/unit-origin-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/fab6b4c1506e0bcb2da7e5a136a0a693ad094ce0/lem/unit-origin-2026-10-20):
one JSON per run with every broadened pole's shares, Jaccard series, entry measures and readers,
plus `grade.py` and `grade.txt`.

| width | broadened layer-1 `+` poles | sprout | born coarse | non-local | unresolved | with a union reader |
|---|---:|---:|---:|---:|---:|---:|
| 16×8 (3 seeds) | 27 | 10 | 6 | 5 | 6 | 24 of 27 |
| 128×64 (3 seeds) | 280 | 217 | 31 | 23 | 9 | 224 of 280 |

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. 16×8: sprout or born coarse ≥ 0.6 of resolved | 16 / 21 = 0.762 | holds |
| 2. 16×8: non-local ≤ 0.15 of resolved | 5 / 21 = 0.238 | **wrong** |
| 3. 128×64: non-local ≤ 0.15 of resolved | 23 / 271 = 0.085 | holds |
| 4. 16×8: ≥ 0.6 have a union reader | 24 / 27 = 0.889 | holds |

**The refutation clause on the non-local candidates.** By the sealed rule, prediction 2's failure
sends each non-local arrival in as a candidate, not a finding.
- **Most candidates are values of one variable**, the mover's seat: (`third`, `last`),
  (`second`, `third`), (`last`, `second`). These are mutually exclusive, so their bridge share is
  0 by construction. That is a limit of the bridge measure, which the input-commonality reference
  then has to carry alone.
  - (`third`, `last`) is named by list v1's own coarse `late`, so it fails part 1.
  - Nearly all of these candidates have union readers, so they fail part 3.
- **The 16×8 `trick-1` → `to-set-le-5` unit** (seed 0, unit 3) has 3 union readers, so it fails
  part 3.
- **Two units pass parts 1–3:**
  - 16×8 seed 0, unit 14, `lead` → `ten-on-trick`: entry at step 30, no union reader, no
    specialist;
  - 128×64 seed 0, unit 59, `lead` → `safe-trick`: entry at step 100, no union reader, 7
    specialists.

  They are **candidates**. Part 4's observer checks are not done: a v2 or reader name for the
  union, the top-5% window, the negative pole, and missing inputs or history (#202). Those are
  #232. Neither is a finding yet.

**Reading.**

- **Read the way the essay reads them, by origin, the broadened units are mostly grown.** At 16×8,
  16 of 21 resolved units sprouted their second concept or were born coarse, and 24 of 27 have a
  layer-2 unit that reads both halves together: someone reads the union. At 128×64, 248 of 271 are
  sprouts or born coarse.
- **Prediction 2's miss is mostly the instrument meeting mutually exclusive values.** Seat
  concepts never share a row, so no bridge can exist. Their unions are named (`late`) or read by a
  union reader, which is how the essay says a coarse unit looks downstream.
- **Two candidates remain**, one per width, both starting from `lead`. In the essay's terms each
  is worth a careful look. One convincing example would matter, scoped to this toy setting.

Scope: the field net's recipe on pip pours, list v1, layer-1 `+` poles, signals conditioned on
the play count. What we learned from this model, not all of 42, yet.

## Correction, 2026-10-20: refutation clause part 3

Part 3 of the sealed clause treated "no layer-2 reader takes both halves" as a mark of mixing. The
review of #232, which read the essay's full text, showed that this misreads the essay. There, "nothing
reads that combination" was evidence *against* a hiding place. A packed code needs downstream units
that *separate* or *cancel* the halves, and specialists that each read one half can be
differentiation. So the two candidates' lack of a union reader does not count toward mixing. #232
replaces part 3 with a cancelling-reader test and requires a positive packing signature.

## Links

[[lem-formation-2026-10-17]] · [[lem-coarse-or-mixed-2026-10-20]] · [[lem-family-statistic-2026-10-20]] ·
[[lem-conditioned-audit-2026-10-20]] · [[clean-neurons-42]]
