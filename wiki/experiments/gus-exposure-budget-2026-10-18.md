---
title: "Gus: four times the outcome-training exposure"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Does four times the fresh on-policy outcome-training exposure improve
[[gus-tiny-gpt]] beyond its own short prefix and matched supervised rehearsal?
The preceding [[gus-staged-outcomes-2026-10-18]] supplies the training-only
motivation: sparse mixed-outcome groups despite diverse paths, with most prior
endgame skill retained. More exposure is a hypothesis, not an established cure.

## Registered comparison

Two continuous runs start from the original 185,652-parameter splat parent:
full-hand GRPO and supervised rehearsal alone. Each receives 1,024 fresh physical
training roots, eight draws per root, 128 rounds and 256 AdamW updates. Its own
round-32 prefix supplies the short-budget control: 256 roots and 64 updates.
The long run keeps AdamW and RNG continuously; a new optimizer is not introduced
at that boundary. This varies fresh data and update count together as an exposure
budget, without identifying their separate effects.

Architecture, clean own-seven/public-history inputs, bid30/pip task, fixed G0
partner, pinned Zeb opponents, original KL reference, seed42 and learning recipe
stay unchanged: lr3e-5, clip .2, KL .02, rehearsal weight1/batch32, eight roots per
round, two inner updates, full seven-turn credit and evaluation every eight rounds.
Both arms collect trajectories for matched diagnostics. Rehearsal has no reward
or KL gradient. Demon is a fenced integration competitor, never a label source.

The 128-root development panel selects greedy then sampled success within the
common original-loss bounds +.02 each, including initialization and earliest ties.
At round32, the saved formation checkpoint's best record freezes the short
selection. At round128, the long selection freezes. Long may legitimately retain
short. Only then are five labels evaluated on 512 untouched physical roots:
original parent, short/long GRPO and short/long rehearsal. Exact inference-state
identity can share lazy records; labels do not multiply independent evidence.

The development criterion requires long GRPO to gain at least 2/128 over both
short GRPO and long rehearsal. Promotion requires greedy gains at least 2pp over
parent and 1pp over each of short GRPO and long rehearsal, every paired 95% lower
bound above zero, with original-task retention. Confirmation uses 20,000 paired
physical-deal bootstrap draws, seed91642. Sampled outcomes are secondary; no
checkpoint is reselected after confirmation.

Each arm has at most 40 cooperative 45s segments with 90s outer caps; two live
arms maximum. Identity, numeric or replay failures stop work with state retained.
The registered budget cannot silently grow. Confirmation uses bounded lazy units;
fixed G0/Zeb/Demon integration faceoffs each retain the 60s cap.

## State and evidence

[Public preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/2dfc408d0971961d300da2297d1b7e43ddb894b0/lem/gus-exposure-budget-2026-10-18) preceded training. All 11 files
(12,201,421 bytes) passed fresh anonymous hashes. Registration SHA256:
`bcf1e93c7204d49044c66dedc0a71c5858082ec1067da146fae774f4e619b835`.

All 1,664 reserved physical deals exclude 600,622 prior deals, including every
reserved root and tournament schedule from the preceding study. All new reservations
become future exclusions. Root audit confirms split uniqueness and fixed seating.

Runtime source is `77e0dc44` (the unchanged trainer implementation
from `1dff0a63`, with 146 Python tests and GREEN gates). The bounded runs are started;
no result is claimed. Protocol and recovery are in [[gus-success-probability-planning]]
and the `gus-training` skill. The ongoing investigation is
[#223](https://github.com/jasonyandell/mk5-main/issues/223).

Live telemetry: [GRPO](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/4b35a82a)
and [rehearsal](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/e462da85).
