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

Run 2026-10-17 as registered. Evidence: `lem/causal-2026-10-17/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/51a73fb797c0048c97d07304f24cfc62486b52cc/lem/causal-2026-10-17).
It holds one JSON per net with every named pole's rates, `grade.py` and `grade.txt`.

**Scope.** These are three 24-neuron nets, read on one position, against uniform beliefs, with
no history, and named with list v2. Of the 108 named poles, 45 `+` poles and 57 `−` poles had
at least 200 rows on each side of their concept.

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. ablation changes the move more where the concept holds (among rows where the unit fires), ≥ 0.7 of `+` poles | 39 of 45 = **0.867** | holds |
| 2. own concept's in/out ratio beats a random other concept's, ≥ 0.7 | 33 of 40 = **0.825** (median ratio 1.49 vs null 0.94) | holds |
| 3. ablation raises teacher regret where the concept holds, ≥ 0.8 | 45 of 45 = **1.000** | holds |
| 4. no hubs: regret rise outside the concept ≤ half the rise inside, ≥ 0.6 | 28 of 45 = 0.622 | holds, narrowly |
| 5. steering: clamping a `−` pole's unit on changes the move more where its concept holds, ≥ 0.6 | 18 of 57 = 0.316 | **wrong** |

**Reading.**

- **The named units are used, and used where their names say.** Every named `+` unit that was
  switched off made the play worse where its concept holds. The median rise in teacher regret is
  0.0093 there, against 0.0037 where the concept does not hold, on a base regret of 0.013. For 39
  of 45, the move moved more often inside the concept than outside, among the rows where the unit
  was firing either way. A unit's own concept separates its effect better than another unit's
  concept for 33 of 40 (null median 0.94, about no separation).
  - Clear cases: `lead` units (moves changed 0.64 / 0.66 / 0.62 on lead rows, against 0.14–0.29
    elsewhere), `partner-winning`, `holds-boss-trump`, `count-on-trick` (regret +0.055 and +0.050
    where count is on the trick, against +0.009 elsewhere), and the declaration detectors
    (`trump-is-fives`, `trump-is-fours`).
  - This is the first evidence of use, not only correlation, and it is the thesis's prediction.
- **The exceptions.** Six `+` poles changed play more outside their concept than inside:
  - `third` (s2 L1+004);
  - `trump-led` (s2 L1+008);
  - two of four `winner-cannot-be-beaten` (s0 L1+006, L1+015);
  - one `no-trump-held` (s1 L1+002);
  - one raw-input unit, `x:winning:2` (s0 L1+000).

  These are candidates for walking the ladder. A unit named by a narrow precision on its top 1%
  may do its work at moderate activations, where the label does not hold. Rung 1 asks whether
  the top 1% is the right window for these units. Nothing here calls for superposition.
- **Steering failed, and the tool is the reason.** Clamping a unit at its 99th-percentile value
  is not a nudge:
  - it changes the move on a median 46% of all rows;
  - it raises regret by about 0.017 everywhere, inside the concept and out;
  - the regret rise is higher where the `−` concept holds for only 22 of 57.

  The ladder stops at rung 1, the tools. The essay steers by adding a measured amount. A
  dose-response clamp (small increments, matched across units) is the honest version, and #198
  stays open for it. This failure does not say the `−` poles are mis-named. It says a clamp this
  large cannot tell.

Caveats:
- one width, 24 neurons;
- the ablation is all-or-nothing;
- the regret is the teacher's under its own plan, so it measures fidelity, not strength;
- the move-change rate counts any change, good or bad.

## Links

[[lem-naming-2026-10-17]] · [[lem-dropout-order-2026-10-17]] · [[lem-fidelity-by-concept-2026-10-16]] ·
[[lem-squeeze-2026-10-16]] · [[clean-neurons-42]]
