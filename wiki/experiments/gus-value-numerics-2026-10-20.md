---
title: "Gus fixed-row value/routing numerical diagnostic"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

For original-parent confirmation pair 09, positive row `292:6`, which returned
fields and conditions disagree between the frozen native-hook and functional
forward paths, and how does each discrepancy change in float64? The row was
selected because the separate [[gus-value-routing-2026-10-20]] instrument
stopped there; its failure receipt did not preserve the condition or numerical
discrepancy. This fixed-row diagnostic cannot resolve that omission by
retrofitting a result into the stopped study.

## Frozen diagnostic scope

The final registration called the original checkpoint's unchanged native-hook
and functional forward paths in float32 and float64 for the baseline and 20
conditions on this single row. With actor and ordinary model baselines, the
run used 90 trained forward calls total. It recorded every returned field's
shape, finite status, exact equality, maximum absolute difference and
coordinates, with raw arrays retained. The original strict `2e-5` criterion
was unchanged. The raw-shape capture gap found before registration was fixed
before the final freeze and independent review. The [registration, SHA256
`3a5ad6de…`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/13f8d44bad5ed328a615a047287b9ed208636b2c/lem/gus-value-numerics-2026-10-20/registration)
was anonymously verified before the sole trained run.

## Result and boundary

The complete 42-case readout is finite; all recorded structural invariants
passed, ordinary native baseline anchors were exact, hooks were cleaned, and
weights were unchanged. Exactly one float32 case crossed the unchanged strict
criterion: condition index 3, current-lead TILE, paired donor, routing and
value together at dose 0.25. The largest difference was at `logits[19]`:
native `-3.480842113494873` versus functional `-3.4808220863342285`, an absolute
difference of `2.002716064453125e-5`, only about `2.72e-8` above the boundary.
Both arrays were finite. The same case's float64 path difference at that logit
was `2.6645352591003757e-14`; the maximum across all float64 main cases was
`5.240252676230739e-14`.

For that condition, the maximum float32 differences at the final head read,
projected write and full layer-0 attention output were respectively
`1.49011612e-8`, `7.45058060e-9` and `7.45058060e-9`; the layer-1 pre/post
differences were `3.81469727e-6` and `1.90734863e-6`. These saved-array results
support precision/operation-order sensitivity with downstream amplification
on this fixed row. They do not isolate precision from operation order as the
unique cause or establish behavior on other rows.

This one-row diagnostic is complete as instrument diagnosis only. It does not
identify the unsaved within-row condition or value from the earlier failed
attempt, validate the other rows or model panel, or supply the stopped
value/routing study's primary comparison. There is no pair mean, bootstrap,
intervention-effect estimate, semantic confirmation, mediation, cleanliness or
playing-strength conclusion. The old threshold was not changed and the old
run was not resumed.
Its [full evidence archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/18c842cfc185e9b0708fbfd67585ae34ae5591df/lem/gus-value-numerics-2026-10-20/evidence)
was anonymously verified at immutable revision
`18c842cfc185e9b0708fbfd67585ae34ae5591df` after independent saved-array
review.

See [[gus-value-routing-2026-10-20]] for the original stopped instrument and
[[gus-book-silas-tools]] for the experiment sequence.
