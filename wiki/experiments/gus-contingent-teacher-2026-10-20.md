---
title: "Gus: discover better contingent decisions, then train on them"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
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

The root is immediately before play 20: the learner acts last in the
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

The matched comparison started both arms at the same 195,000-update
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

## Result: better teacher, no transferred student gain

The pilot completed in ten normal segments: seven collection, one training,
one development and one test. The teacher found strict complete-policy gains
on 8/32 training roots and 15/32 untouched test roots. On test, the exact
teacher averages 32.3553% success versus the original parent's 25.2976%.
These are averages over finite root priors, not observed full-hand win rates.

Both training arms completed 64 updates. Sixteen development roots selected
teacher-arm initialization and parent-action control step 32. On 32 untouched
test roots the selected teacher arm therefore equals the original parent:
25.2976%, difference zero, paired interval [0, 0]. The control averages
23.9575%; its degradation cannot establish a learned improvement. The
registered requirement to beat both controls failed. No full-hand stage or
promotion followed, and no unselected checkpoint was tested.

One training witness makes the teacher gain concrete. In `train-0056`, both
policies first play 3-1. After left 0-0, partner 4-4, right 2-0, Gus chooses
4-1 with exact value 0; the teacher chooses 5-3 with value 1 in that cell.
The complete teacher policy improves root success from 1/18 to 4/18. A
different reached branch selects another action from a value tie, so this
is evidence of a useful later correction, not proof every branch requires
adaptation. Five training roots pass the registered contingency-witness gate.

## Why change the learning objective next?

The saved training table has 622 discretionary unsettled nodes. All legal
Q values tie on 556 rows, representing 87.6984% of root-balanced sampling
weight. Only 18 rows across nine roots strictly change the parent's action:
4.2379% of sampling weight. All 18 were sampled; the 2,048 draws contain 88
strict corrections and 1,824 all-action-tie examples. Most cross-entropy
pressure therefore reinforced parent choices or arbitrary value ties.
This is a measured imbalance, not proof of the cause of failed transfer.

The selected successor is a separately registered comparison of expected
exact action regret against ordinary teacher-label cross-entropy. Reuse only
these training roots, with fresh development/test confirmation. The actor
loss `sum_a pi(a|I) [max_b Q*(I,b) - Q*(I,a)]` gives flat-value rows zero
policy gradient and respects optimal ties. It is an offline cost-sensitive
expert-guidance step, related to [AggreVaTe](https://arxiv.org/abs/1406.5979),
not its full interactive algorithm or a borrowed no-regret guarantee.
Q uses optimal teacher continuation; the student still needs complete-policy
evaluation because its later actions can differ. No auxiliary head or new
actor input is selected.

## Verification and provenance

[Public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/7a87aaad030605e02dddb10a3dca30d898efe578/lem/gus-contingent-teacher-2026-10-20/registration)
was freshly verified before collection: 11 files, 245 archive members, 23
frozen protocol/instrument/input files. Independent source reviews covered
lawful seat inputs, information-cell recursion, training isolation and
resume/failure guards. Collection performed independent full-policy forward
checks and native-request replay. The later saved-evidence audit verified
80 selected root certificates, 30,300 listed cache hashes, 160 development
and 96 test records, selection and registered paired-bootstrap arithmetic.
It loaded no models or game replay; signal-forward arrays are retained only
as digests, a recorded limit of that later audit.

The first parent launch used system Python by resolving the venv symlink;
it failed during imports before any worker attempt or model call. Its command,
traceback and cost remain in `launch-failure-v1`. An explicitly inspected
interpreter-path correction changed no frozen science or budgets. The
scientific controller completed without failures or cap growth. Audit-script
construction errors are likewise disclosed without changing experiment data.

All 607,307 prior/fixture deals plus all 1,408 new potential source deals are
reserved: 608,715 exclusions for future fresh sampling. Old dev/test roots
remain unavailable for training. Recovery and receipts live in
`scratch/lem/gus-contingent-teacher-v1/CURRENT.md`; `READOUT.md` and the
independent saved-data reviews separate teacher, student and full-hand claims.
The prior one-choice verifier remains [[gus-fixed-field-curriculum-2026-10-18]];
capability boundaries are in [[gus-book-silas-tools]].

[Complete pilot evidence and independent saved-data audit](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/acfad317b78720517a61378a9c80641f0c38ba7c/lem/gus-contingent-teacher-2026-10-20/evidence)
was freshly downloaded and verified: 11 files and 56,578 archive members. It
retains the new root certificates, native caches, both training trajectories,
selection, test traces and the original launcher failure in one immutable batch.
The growing report has a [contingent-decisions chapter](https://gus-first-look.psychometrix.chatgpt.site/#contingent).
