---
title: "Gus: exact endgames under the fixed G0/Zeb field"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and scope

Can exact short-endgame success feedback under the full-hand stage's fixed
G0/Zeb policies improve [[gus-tiny-gpt]] beyond matched prediction rehearsal,
and does any gain transfer to fresh full-hand outcomes? This follows the
completed update-scale comparison in [[gus-update-scale-2026-10-18]]. The
pretrained joint model, clean inputs and nonlearner policy weights/budgets stay
fixed. The earlier lowest/highest/uniform verifier keeps its separate identity.

The root belief is uniform over all legal-consistent remaining deals. It omits
pre-root behavior likelihood, so exact rational values certify this finite
approximation, not the true posterior or unknown players. Each root action is
held across every world; the learner's later move must be forced. Other players
respond to each continuation through their own information-limited policy.
Gus itself remains a direct neural policy. [[gus-success-probability-planning]]
records the expected-success objective and its textbook justification.

## Implemented construction and recovery

`lem.observer.rlvr.fixed_field` produces exact rational certificates plus every
terminal continuation, independently checked by the rules replay. Its identity
pins G0/Zeb weights, deterministic policy specification, source/native bytes,
Torch/platform and individual CPU float32 inference. Counterfactual hidden
hands stay in the witness, never the neural input.

`lem.observer.rlvr.fixed_data --plan PLAN --out DIR --seconds 45` constructs a
curriculum from predeclared, disjoint physical root pools. Histories use one
sampled pretrained learner, its fixed G0 partner and pinned Zeb opponents under
the existing bid30 auction. Each root supplies at most one unsettled two-legal
choice during the penultimate trick. Split quotas choose the first eligible
deals, without filtering by wins, mistakes or label gaps. Screening cost and
all potentially used source deals remain recorded. Collection batch size is
pinned; compute slicing does not change it.

The existing immutable `Extension` commits a complete collection/verification
unit. Signals stop between units; crashes redo only an uncommitted unit.
Completed records are reused, and `--recheck` recomputes their canonical bytes.
The loader checks clean prefixes, rational supports and every selected row
against its hashed witness record. Completed resumes preserve dataset bytes.
Source, weights, quotas or seating changes require a new plan/output identity.

The existing exact-success trainer now also supports `--mode rehearsal` as a
matched supervised-only control. It uses the same question/rehearsal streams
and optimizer counts, with zero policy-reward and reference-KL gradients.
Optimizer, all RNG, formation, initial/best/final and original-task retention
remain saved. Training resume pins the complete observer Python source tree.

## Experiment state

The generator, verifier and rehearsal comparator are implemented. This page
will record the preregistration and measured results as the bounded study runs.
No curriculum-trained player or general playing gain is claimed yet.

The series remains tracked by [issue #223](https://github.com/jasonyandell/mk5-main/issues/223).
Interpretation specimens follow `clean-neuron`; unit claims need separate causal
evidence. The `gus-training` skill carries the construction and recovery route.
