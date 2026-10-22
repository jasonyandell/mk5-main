---
title: "Gus decision-class probe — does the representation carry the class?"
kind: experiment
first_seen: 2026-10-21
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

[[gus-decision-quotient-2026-10-21]] counts the decision classes of the contingent course: at each decision, the weak order of exact Q over the menu (13 realizable at three-way menus, 3 at two-way). The raw interface never repeats a fresh root, so a student that transfers must compute the class from the history. Does [[gus-tiny-gpt]] already compute it? At the query token, does its representation linearly encode (a) the class, (b) the exact Q contrasts between menu slots, (c) flat versus nonflat? And did the 512 masked updates of [[gus-contingent-maskbreadth-2026-10-20]] move it? This is Night-Probe's first question for the night of 2026-10-21. It is a measurement: no model is changed, nothing trains.

## Instrument

- **Rows**: the 6,466 discretionary unsettled decisions of the 256-root course, as `lem/quotient.py` reads them. 223 three-way rows (roots only) and 6,243 two-way rows (33 roots, 6,210 second decisions). Slots are menu tiles ranked by tile id, as in the quotient.
- **Models**: the Splat parent (`gus-splat-v1` best, step 195,000), the wide student (maskbreadth wide step 512, 128 TRAIN roots), the narrow student (flatmask masked step 512, the first 64 of those roots), and three untrained twins of the same architecture (torch seeds 0, 1, 2).
- **Features at the query token**: both blocks' MLP pre-activations (256 each, `lem.probe.gus.forward`); the residual stream after each block (64 each); the tile logits at the menu slots, centred. The observer does not expose its residual stream. The probe recomputes it by calling the model's own submodules in order, and requires the final logits to equal `forward`'s exactly.
- **Baselines**: majority class (chance); the play count (sequence-length one-hot), so free positional structure is not credited; the menu tiles (legal-tile one-hot).
- **Probes**: linear only. Standardized ridge for contrasts, standardized L2 logistic for classes and flatness. Each is fit on the course's 128 TRAIN roots, its penalty chosen by 4-fold CV grouped by physical root inside TRAIN, then scored once on the 128 TEST roots. Intervals are 95% percentiles over 1,000 resamples of TEST roots, with the same resamples for every model, so differences are paired.
- **Metrics**: balanced accuracy for the class (ties are the majority: 54 of 116 three-way and 3,000 of 3,350 two-way TEST rows); R² for the contrasts; AUC for flatness.
- **Policy readout**: separately, at nonflat TEST decisions, the share where the net's own argmax over the menu lies in the maximizing set.

## Result

The representation does not linearly carry the class or the contrasts on fresh roots. It carries two-way flatness, and that came from pretraining, not from the masked lessons. What the lessons moved is the net's own choice at nonflat roots.

**(a) Class.** Three-way, 107 TRAIN and 116 TEST rows over 13 classes: every probe on every model collapses to the majority class (triple tie). Balanced accuracy is 0.083 [0.083, 0.100] for the parent, wide and narrow at every layer, equal to chance. The untrained twins average 0.069 to 0.083. With 107 fitting rows for 13 classes this is underpowered, not evidence of absence. Two-way: the best probe is wide res2 at 0.387 [0.328, 0.457] against chance 0.333; the parent's is 0.369 [0.332, 0.427].

**(b) Contrasts.** Out-of-sample R² lies between −0.035 and +0.068 everywhere. The single excursion above the twins is wide's menu logits at three-way: 0.068, +0.070 [+0.008, +0.117] over the twins.

**(c) Flatness**, AUC (chance 0.5):

| Two-way, TEST 3,350 rows / 128 roots | pre1 | pre2 | res1 | res2 |
| --- | ---: | ---: | ---: | ---: |
| Parent | 0.562 [0.485, 0.652] | **0.767** [0.689, 0.836] | 0.709 [0.623, 0.790] | 0.630 [0.495, 0.775] |
| Wide | 0.587 | 0.682 [0.591, 0.782] | 0.710 | 0.599 |
| Narrow | 0.571 | 0.725 | 0.715 | 0.680 |
| Untrained twins (mean of 3) | 0.585 | 0.610 | 0.592 | 0.620 |
| Play count / menu tiles | 0.545 / 0.542 | | | |

The parent's layer-2 pre-activations exceed the twins by +0.157 [+0.048, +0.281], and res1 by +0.116 [+0.032, +0.195]. Wide minus parent at pre2 is −0.085 [−0.156, −0.017]: the masked loss zeroes every all-tie term, and the student's layer-2 readout of flatness weakened. At three-way, flatness is not read by any probe: 0.415 to 0.545, and nothing differs from the twins.

A linear probe on the signed menu logits cannot express flatness, which depends on the gap's size, not its sign. Its AUCs below 0.5 (parent 0.347) are that limit, not a reversed signal.

**Policy readout** at nonflat TEST decisions:

| | Three-way (62 rows) | Two-way (350 rows / 58 roots) |
| --- | ---: | ---: |
| Uniform chance | 0.457 | 0.500 |
| Untrained twins (mean) | 0.435 | 0.453 |
| Parent | 0.581 [0.468, 0.694] | 0.540 [0.434, 0.669] |
| Narrow | 0.613 [0.484, 0.742] | 0.526 [0.421, 0.679] |
| Wide | **0.710** [0.597, 0.823] | 0.577 [0.476, 0.711] |
| Wide − parent | +0.129 [+0.016, +0.258] | +0.037 [−0.053, +0.120] |

On the TRAIN roots it was taught, wide reaches 0.945 (three-way) and 0.964 (two-way): the lessons were acquired. On fresh roots the gain is at three-way roots, where the class probes saw nothing.

## Reading

- The class is not a linear direction at the query token that a 107-row probe can find. Yet wide's own logits order the three-way menu better on fresh roots than the parent's do. The tile logits are slot-aligned by construction, while the hidden features are not: to read "slot 1 beats slot 0" off a hidden unit, a linear map must also decode which tiles hold the slots. The class and contrast probes on hidden features test that harder thing. The policy row is the fair slot-aligned readout, and it moved.
- Flatness is order-free, so the hidden-feature probes test it fairly, and pretraining put it there. The masked lessons never trained on a tie and wore the layer-2 readout down. That is a side effect the masked-loss studies ([[gus-contingent-maskfit-2026-10-20]]) did not measure.
- Many intervals were computed: 5 feature sets × 6 tasks × 6 pairs. Single excursions near zero (e.g. two-way class wide − parent at pre2, +0.031 [+0.002, +0.069]) are not findings. The flatness gap over the twins appears in two correlated feature sets of one model.

## Scope

This net, these inputs (one decision's history, the fixed G0 partner and pinned Zeb of the course, no belief input), and these targets: the course's 6,466 decisions, one fixture (three own tiles, acting last in trick five). What we learned from this model, not all of 42, yet. How much play the nets learned: fresh-root success is parent 37.3% and wide 39.6%, and on its TRAIN roots wide reaches 40.1% against the teacher's 40.4% ([[gus-contingent-maskbreadth-2026-10-20]]). Linear probes only. A nonlinear or slot-aligned probe of the hidden features (for example, through the unembedding of each menu tile) is the next instrument, and it is not run here (#240).

## Evidence

Tool `lem/probe/decision.py`, tests `lem/tests/test_decision_probe.py`; `lem/probe/gus.py` now reads a Splat checkpoint's backbone. Each model's probe run took under 10 s and the summary 79 s under `scripts/cpu-slot`. [Report and per-model TEST predictions](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/2858ebeedef9b204f713b2d8cdcdc903270dc26c/lem/gus-decision-class-probe-2026-10-21) (17 files, verified by sha256). The companion question, a tile-free label that routes, is [[gus-observation-labels-2026-10-21]].
