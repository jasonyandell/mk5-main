---
title: "Already-clean test 5: causal — do the narrow nets' named units do what their names say? (Lem, #198)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]], [[lem-the-teacher]]). So far every
clean unit has been read by correlation, the precision of a concept on its top 1%. This is the
first test of use: switch a named unit off, or force it on, and see whether the play changes
where its name says it should. It is G3 of the paper, on the 24-neuron nets, where every unit
can be tested. A unit named "late in the trick" that changes play only late in the trick is the
thesis working. A unit whose ablation hurts everywhere is a warning.

Scope: this net, these inputs (one position, uniform beliefs, no player-behaviour history), and
list v2. A unit with no v2 name is not tested here; its reader name ([[lem-naming-2026-10-17]])
is not executable.

## Pre-registration

- **Nets.** The three 16×8 seeds of [[lem-dropout-order-2026-10-17]], with their v2 audits from
  [[lem-naming-2026-10-17]]. The subjects are every pole that is clean under v2 and whose neuron
  fires on at least 200 rows.
- **Rows.** The audit sample (434,535 held-out rows), with the teacher's q for every legal move
  from G7 (`fidelity-2026-10-16/q`). The plumbing reproduces G7's regret on the shipped net,
  0.00844.
- **Interventions** (`python -u -m lem.probe causal`). A neuron is set after its ReLU and the
  rest of the net is recomputed in the referee's integers.
  - **Ablate** (`+` poles) sets it to 0.
  - **Clamp** (`−` poles) sets it to its 99th-percentile value over the rows where it fires.
- **The comparison removes the circularity.** Ablation touches only rows where the neuron fires,
  and the audit named the `+` pole from those rows. So the `+` test is made **among the rows
  where the neuron fires**: is the move change more frequent where the pole's concept holds than
  where it does not? For a `−` pole the rows are those where the neuron is off, and the
  intervention is the clamp. Groups with fewer than 200 rows are not scored.
- **The null.** The same rates, split instead by the concept of another clean pole of the same
  net, drawn at random (seed 20261017).
- **Measures per named pole.**
  - the move-change rate where the concept holds and where it does not;
  - the change in teacher regret, likewise;
  - the change in regret over all rows;
  - the null's rates.
- **Predictions, before any intervention runs on these nets.**
  1. **Selective change (`+` poles).** Among rows where the neuron fires, ablation changes the
     move more often where its concept holds than where it does not, for at least 0.7 of named
     `+` poles.
  2. **Its own concept, not any concept.** The pole's in/out ratio of change rates beats the
     null concept's ratio for at least 0.7 of named `+` poles.
  3. **Used, not decorative.** Ablation raises the teacher's regret where the concept holds, for
     at least 0.8 of named `+` poles.
  4. **No hubs.** For at least 0.6 of named `+` poles, the regret increase where the concept
     does not hold is at most half of the increase where it does.
  5. **Steering (`−` poles).** Among rows where the neuron is off, clamping it on changes the move
     more often where the `−` pole's concept holds than where it does not, for at least 0.6 of
     named `−` poles.
- **Cost.** One forward pass of a 24-neuron net per neuron and intervention, about 150 passes in
  all, minutes on one core.

## Results

Not yet run.

## Links

[[lem-naming-2026-10-17]] · [[lem-dropout-order-2026-10-17]] · [[lem-fidelity-by-concept-2026-10-16]] ·
[[lem-squeeze-2026-10-16]] · [[clean-neurons-42]]
