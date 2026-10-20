---
title: "Already-clean test 5b: steering with matched rows — does a named unit's signal belong where its concept holds? (Lem, #205)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
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

Not yet run.

## Links

[[lem-causal-2026-10-17]] · [[lem-naming-2026-10-17]] · [[lem-conditioned-audit-2026-10-20]] ·
[[clean-neurons-42]]
