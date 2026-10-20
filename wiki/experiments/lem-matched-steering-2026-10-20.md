---
title: "Already-clean test 5b: steering with matched rows — does a named unit's signal belong where its concept holds? (Lem, #205)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). In [[lem-causal-2026-10-17]],
ablation showed that named units are used where their names say. Steering by a matched dose was
unresolved: comparing its cost inside and outside a concept mixed rows of different difficulty.
This test compares rows only within the same play count, and compares each unit against other
units steered on the same rows. Row difficulty cancels in both comparisons.

Scope: the field net's recipe, three 24-neuron nets, one position against uniform beliefs, no
history, list v2.

## Pre-registration

- **Nets, names, rows and teacher q.** As in [[lem-causal-2026-10-17]]: the three 16×8 seeds,
  their v2 audits from [[lem-naming-2026-10-17]] (unconditioned names), the 434,535-row sample,
  and G7's q.
- **Intervention** (`lem.probe causal --matched`). Add k·σ to a unit after its ReLU, for
  k ∈ {0.5, 1}. σ is the unit's standard deviation over the rows where it fires. Every live unit
  is steered once per dose and its regret rise kept.
- **The statistic.** For a named pole with concept C, on the rows where its unit was off: within
  each play count, take the mean regret rise where C holds minus where it does not. Average over
  play counts, weighted by min(n_in, n_out), over play counts with at least 20 rows on each side.
  This is `own`. `random` is the same statistic, on the same rows and the same C split, for 5
  other live units of the same net drawn at random (seed 20261017), each at its own k·σ.
- **Predictions.**
  1. **`+` poles: the signal belongs where the concept holds.** `own` is below the random median,
     for at least 0.6 of named `+` poles, at k = 0.5 and at k = 1.
  2. **`−` poles: the signal is wrong where the concept holds.** `own` is above the random median,
     for at least 0.6 of named `−` poles, at k = 0.5 and at k = 1.
  3. **The effect is the unit's, not the rows'.** Pooled over all named poles and both doses,
     the sign-corrected difference (own − random median for `−` poles, random median − own for
     `+` poles) has a median above 0.
- **Cost.** 2 doses × 24 neurons × 3 nets = 144 forward passes, minutes on one core.

## Results

Run 2026-10-20 as registered. Evidence is `lem/matched-steering-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/02c07602e6828c5e79b2034b2e73b38556c73f8b/lem/matched-steering-2026-10-20):
one JSON per net with every pole's `own` and five `random` gaps per dose, plus `grade.py` and
`grade.txt`.

| | k = 0.5 | k = 1 | needed |
|---|---:|---:|---:|
| 1. `+` poles with `own` below the random median | 15 / 37 = 0.405 | 17 / 37 = 0.459 | 0.6 |
| 2. `−` poles with `own` above the random median | 23 / 43 = 0.535 | 21 / 43 = 0.488 | 0.6 |
| 3. pooled sign-corrected median (n = 160) | | **−0.000102** (0.475 positive) | > 0 |

**All three predictions are wrong. Steering cost, compared fairly, shows no unit-specific
signal.** The stratified gaps are of the order 10⁻⁴ in regret, a named unit's own gap sits
inside the spread of five random units' gaps on the same rows, and the pooled median is at
chance.

**Reading.**

- **What it rules out.** The earlier near-chance steering result
  ([[lem-causal-2026-10-17]] addendum) was not hiding a real effect behind mixed row difficulty.
  With difficulty matched within the play count and the comparison made against other units on
  the same rows, the effect is still absent.
- **What it does not rule out.** That steering moves the play *in the concept's direction*. The
  statistic reads the cost of an added signal, not where the move goes. Three readings remain
  open, and this test cannot choose between them:
  1. Cost is the wrong reading: a unit's added signal changes the move toward the concept's play,
     and the teacher's regret, on rows where the concept does not hold, is about equally hurt by
     any change.
  2. The rows are the wrong rows: the rows where a unit is off and its concept holds are the
     unit's own judgement that the concept-like play is wrong, so adding its signal there need not
     help.
  3. The units are named but their effect in this net is mostly to gate, not to vote. Ablation
     shows they are used where their names say, and steering a gate open on the wrong rows has
     no specific cost.
- **Where it leaves the causal case.** Ablation is the evidence of use (45 of 45 named units raise
  regret where their concept holds; 39 of 45 selective). Steering, read by cost, adds nothing for
  or against. A direction-based steering test needs a definition of "the concept's play" first,
  and it will be filed when it has one.

Scope: three 24-neuron field nets, one position against uniform beliefs, no history, list v2.

## Links

[[lem-causal-2026-10-17]] · [[lem-naming-2026-10-17]] · [[lem-conditioned-audit-2026-10-20]] ·
[[clean-neurons-42]]
