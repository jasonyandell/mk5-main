---
title: "Gus: endgame initialization for full-hand outcome training"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and scope

Does the exact endgame curriculum in [[gus-fixed-field-curriculum-2026-10-18]]
provide a better initialization for subsequent full-hand GRPO than the original
splat parent or its matched endgame rehearsal control? That curriculum improved
fresh finite-model endgame regret without confirming stronger full-hand play.
This comparison tests the proposed transfer through another outcome-training
stage; it does not assume transfer has already happened.

[[gus-tiny-gpt]] keeps its 185,652 parameters, original own-seven/public-history
inputs, fixed bid30/pip task, G0 partner and pinned Zeb opponents. The changed
factor is the learner's starting weights. Original parent weights remain the
common frozen KL reference and common retention baseline in every arm. The
same supervised rehearsal, optimizer recipe, physical roots, action-stream
namespace and update counts are shared. The pinned Demon remains a separate
integration competitor and never supplies labels.

## Engineering

`lem.observer.outcomes` requires explicit `initial`, `reference`, `replay` and
their SHA256 identities. The starting policy receives fresh AdamW state;
`reference` supplies frozen legal distributions and the original-task retention
baseline. An initial policy that violates those declared bounds fails before
writing a usable training checkpoint. A changed reference invalidates exact
resume. Old studies require their published runtime rather than an implicit
reference default.

Real SIGTERM and hard-crash tests cover a distinct initial/reference pair.
The reference remains unchanged and has no gradients; initial snapshots preserve
the supplied warm-start weights. A warm start into the new phase is different
from `--resume`, which retains AdamW, RNG, source/plan identities and progress
within that phase. Completed resumes still preserve existing bytes.

## Study state

Implementation and tests precede preregistration. No staged-training result is
yet claimed. The planned bounded comparison uses three GRPO arms from the
original splat parent, the predeclared seed42 exact-endgame selection and its
matched rehearsal selection. A fourth, supervised-only continuation from the
endgame-trained start controls for additional rehearsal during this new phase.
Full parameters, disjoint roots, development
selection, material-gain rule and untouched confirmation will be published
before training. The established GRPO implementation is described in
[[gus-full-hand-outcomes-2026-10-18]]; the objective and lazy evidence contract
are in [[gus-success-probability-planning]].

The continuation is tracked by [issue #223](https://github.com/jasonyandell/mk5-main/issues/223).
