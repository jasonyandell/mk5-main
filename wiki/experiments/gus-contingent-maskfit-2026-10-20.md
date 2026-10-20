---
title: "Gus masked-CE TRAIN acquisition and occupied regret"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->
# Gus learned almost all of these known decisions

The fixed final masked student chooses an exact teacher-optimal action on **71 of 72 known corrections**, with **one new consequential error** among 133 other nonflat rows. Its complete finite TRAIN success is **43.0605%**, only **0.0521 percentage points** below the teacher. This establishes acquisition on these training positions; unseen transfer remains unestablished.

| Fixed specimen | Complete TRAIN success | Corrections repaired /72 | New other-nonflat errors /133 | Visited / unvisited suboptimal rows |
|---|---:|---:|---:|---:|
| Original parent | 36.3600% | 0 | 0 | 39 / 33 |
| Masked selected128 | 42.9636% | 67 | 3 | 3 / 5 |
| Masked final512 | 43.0605% | 71 | 1 | 2 / 0 |
| CE selected384 | 42.6656% | 60 | 0 | 9 / 3 |
| CE final512 | 42.5730% | 64 | 1 | 9 / 0 |
| Exact teacher | 43.1126% | — | — | — |

These are descriptive means over the same 64 original TRAIN roots, not independent test estimates or full-game win rates. All five specimens were fixed in advance. Final512 is a diagnostic specimen, not a replacement for the previously selected masked128. The previous flatmask held-out gate remains FAILED; no promotion or full-hand test follows.

## Why complete policies matter

A correct move can lead to another decision the student mishandles. We therefore evaluate the student's entire remaining policy, not just its first move. Exact occupancy weights record how often its paths reach each decision. Summing occupied teacher regret exactly equals the teacher's value minus that student's complete value under this finite model.

For masked final512, the strict-correction contribution is 1/3840 and the retention contribution is 1/3840, totaling 1/1920 (0.0521 percentage points). Both errors are reached; all 814 unvisited rows are optimal. CE final512 has 37/7680 strict-correction regret plus 1/1728 retention regret, totaling 373/69120 (0.5396 points). Its nine errors are all reached. CE selected384 has the different decomposition 103/23040, entirely strict-correction regret.

All-action ties have zero immediate teacher regret, but they can route Gus toward later mistakes. Teacher Q assumes optimal future decisions, whereas the student's continuation contains its own errors. This arithmetic does not identify the causal effect of masking or prove interference.

## Scope and verification

The registered diagnostic used 64 TRAIN roots, 1,356 prefixes and five fixed policies: 6,780 logical captures and 320 exact root values. It reused 622 old parent captures and 32 old parent values byte-identically, making 6,158 new native calls and 288 new root evaluations. One 15.499-second scientific segment closed normally. No new training, roots, field-model calls or development/test access occurred.

Independent saved-data audit passed **415,646 checks**, reconstructing all captures, input identities, exact value arithmetic, occupancy identities and closed manifest bindings. It did not instantiate models, replay gradients, or independently replay game transitions. Independent result review checked every one of 14,201 output hashes and all 320 saved root checks. Two review-prose transcription errors were corrected before publication; the original review is retained. Scientific outputs were unchanged.

Keep original clean bid30 inputs, fixed G0 partner and pinned Zeb. Root uncertainty is uniform over legal-consistent worlds and carried through observed field actions; indistinguishable worlds are summed before choosing one shared learner action. The planner stays offline. All 620,747 existing reservations remain future training exclusions.

## What this changes

The known lessons are largely acquired. Simply extending practice on these examples is no longer the obvious next lever. A fresh, separately registered transfer question is a candidate; no new learning recipe or comparison is selected here. It must not reselect from the old test or reinterpret this TRAIN diagnosis as a successful held-out gate.

## Sources and continuation

Follows [[gus-contingent-flatmask-2026-10-20]] and reuses [[gus-contingent-fit-2026-10-20]]. Live receipts: `scratch/lem/gus-contingent-maskfit-v1/CURRENT.md`.

[Verified registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/497d43b5ba84995eb323346f671c27ce08240aa4/lem/gus-contingent-maskfit-2026-10-20/registration), 10 files /54 members before measurement. [Verified evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/afa22e9d330667948ef6aa70469c58758e1c349c/lem/gus-contingent-maskfit-2026-10-20/evidence), 11 files /12955 members, freshly downloaded anonymously and verified. Original parent assets are referenced; 32 newly created local wrapper receipts are included.

[Owner-private report](https://gus-first-look.psychometrix.chatgpt.site/#maskfit). Capability refresh found unchanged Lem207, pending Book121 and open Silas166. Existing finite verifier suffices.
