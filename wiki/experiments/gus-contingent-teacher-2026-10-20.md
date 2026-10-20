---
title: "Gus: discover better contingent decisions, then train on them"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: preparation
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## The question

[Issue #237](https://github.com/jasonyandell/mk5-main/issues/237) tracks this pilot.

Can a small exact offline planner discover better observation-contingent
decisions under the fixed G0/Zeb field, and can the existing tiny transformer
learn those decisions? Jason explicitly selected this direction and resumed the
research loop. This follows [[gus-success-probability-planning]]; another
auxiliary future-prediction objective is not the selected experiment.

The supplied *42 continuations v3* manuscript and technical notes distinguish
evaluation of a fixed contingent policy from optimization over lawful policies.
The former is linear in a root belief; the latter chooses among those policy
values. A future learner choice must be shared across hidden worlds producing
the same learner information. The prototype uses the full finite belief rather
than claiming to have computed a minimal closed representation.

The learning pattern is related to [Expert Iteration
(Anthony, Tian and Barber, 2017)](https://arxiv.org/abs/1705.08439):
planning supplies improved policies and a network learns to generalize their
decisions. This pilot is narrower than that algorithm. It uses exact bounded
information-cell planning against a frozen field, followed by one matched
distillation comparison; it does not yet feed a trained student back into an
improved planner or deploy search. The teacher's guarantee belongs to its named
finite model. Approximate neural fitting does not inherit that guarantee.

## Small enough to check, long enough to plan

The proposed root is immediately before play 20: the learner acts last in the
fifth trick, holding three tiles while each other seat holds two. Six hidden
remaining tiles admit at most `6!/(2!2!2!) = 90` allocations. Nine plays remain.
The learner can choose now, observe more play, then choose again with two tiles.
Its final tile is forced.

Eligibility requires a discretionary, publicly unsettled root and at least one
positive-mass, discretionary, unsettled later learner information cell. This
alone does not establish useful contingency. Before training, a worked training
witness must show that the chosen first action reaches two different observations
requiring different selected second actions, and that the complete teacher policy
has strictly greater exact success than the parent.

Root weights are uniform over legal-consistent hidden remainders. They omit
pre-root behavioral likelihoods; this is an explicit model approximation. After
the root, modeled field actions filter those same weights. Child beliefs are not
reset to uniform legality. The recursion sums observation branches and maximizes
one common learner action within each information cell. There is no per-world
maximization or opponent minimization.

## What reaches Gus

The unchanged actor input is its own original seven tiles, bid30, relative
bidder/declaration and public actor/tile history. Hidden worlds and continuation
values remain referee/teacher evidence. G0 remains the partner and pinned Zeb
occupies the other team. Each simulated seat receives only its own lawful
policy request. Exact request identities, fixed policy bytes and deterministic
responses are checked independently.

The proposed matched comparison starts both arms at the same 195,000-update
splat checkpoint's **tile-policy backbone**. One arm learns the exact teacher's
actions; the other learns the frozen parent's actions on the same prefixes.
Legal-masked cross-entropy trains the actor directly, without an auxiliary
forecasting head or original imitation-retention gate. Root-balanced sampling
prevents a large decision tree from counting as many independent deals.

Training alone chooses training labels. Development can check input aliases
and select a saved student snapshot; it cannot revise training labels. Test
teacher graphs remain unavailable until selection is frozen. A student's score
integrates its entire remaining policy over the exact root support, including
its own later decisions. Agreement on the first move is only a diagnostic.
Fresh full-hand play is a separate conditional stage and a separate strength claim.

## Current status and provenance

Preparation only: the information-cell solver, actual-game adapter, collector,
matched trainer and evaluator are being integrated and independently reviewed.
The synthetic solver checks observation-dependent choices against exhaustive
lawful-policy enumeration. The synthetic trainer checks exact recovery of
weights, optimizer and random state after interruption. These are instrument
tests, not measured Gus improvements.

A mechanical support fixture used during preparation is retained with its
receipt and excluded from study pools. All 607,306 earlier reserved deals,
that fixture, and 1,408 newly reserved source deals are kept separate. No
training or actual-policy measurement is authorized by this preparation page;
the reviewed frozen protocol, code hashes and bounded stage caps must first
be publicly registered and freshly verified.

Recovery and detailed preparation live at
`scratch/lem/gus-contingent-teacher-v1/CURRENT.md`. The exact one-choice
predecessor remains [[gus-fixed-field-curriculum-2026-10-18]]; its verifier
explicitly rejects a second discretionary learner choice and is not relabeled
as the new teacher. Lem, Book and Silas capability boundaries are recorded in
[[gus-book-silas-tools]].
