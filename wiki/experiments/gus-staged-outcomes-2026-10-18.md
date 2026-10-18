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

Source `1dff0a63` is merged with 146 Python tests and repository gates GREEN.
The [public preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/885cede1d7ac6ee021adcb28772236c273271823/lem/gus-staged-outcomes-2026-10-18)
contains all plans and exclusions; all 11 files (12,423,655 bytes) passed fresh
anonymous hashes. Registration SHA256:
`13c07ae7ad556c9199ee80550a817df896546ffbdd52c83527d85737e8a5df8e`.
All 896 reserved physical roots exclude 599,687 prior deals and become future
exclusions themselves.

| Arm | Initial policy | Phase objective |
|---|---|---|
|parent|Original splat parent|Full-hand GRPO + KL + rehearsal|
|endgame|Predeclared seed42 exact-success selection|Same full-hand GRPO + KL + rehearsal|
|rehearsal_start|Predeclared seed42 endgame rehearsal selection|Same full-hand GRPO + KL + rehearsal|
|rehearsal_only|Same start as endgame|Supervised rehearsal only|

Every arm uses 256 fresh physical training roots with eight draws per root,
32 rounds/64 AdamW steps, learning rate 3e-5, clip .2, KL .02, rehearsal weight 1
and batch 32. The seven-turn reward horizon is unchanged. Cooperative segments
are 45s with 90s outer caps; at most 16 per arm, with two live arms. This is
an initialization comparison plus a matched supervised-continuation control,
not a changed opponent population or new architecture.

The shared 128-root development panel selects greedy then sampled success within
the common original-loss bounds +.02 each, including initialization and retaining
the earliest tie. The development method criterion requires the endgame arm to
gain at least 1/128 over its start and 2/128 over each other selected arm. All four
selections freeze before reading the separate 512-root confirmation. Confirmation
measures all four selected functions and all three original starts, with greedy
and one coupled sampled rollout per root. Identical inference states may share
records under the same pinned runtime/field/seed recipe; labels are counted
separately from unique trajectories.

Promotion requires the selected endgame arm to gain at least 2pp over the original
parent and 1pp over its own start and each other selected arm, with positive paired
95% lower bounds and common-reference retention. This prevents an already-stronger
start from being mistaken for learning during the new phase. The study is running;
no result is claimed yet. The established objective is in
[[gus-full-hand-outcomes-2026-10-18]] and the lazy evidence contract is in
[[gus-success-probability-planning]].

The continuation is tracked by [issue #223](https://github.com/jasonyandell/mk5-main/issues/223).
