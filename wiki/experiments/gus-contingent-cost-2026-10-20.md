---
title: "Gus: expected exact action regret versus teacher-label imitation"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->
# Gus: expected exact action regret versus teacher-label imitation

Status: independently reviewed preparation; public registration and launch receipts in `scratch/lem/gus-contingent-cost-v1/`. This is an adaptive successor to [the contingent teacher pilot](gus-contingent-teacher-2026-10-20.md), not an independent replication.

The prize remains: use continuation machinery to discover better contingent decisions, then train Gus on those decisions. The first teacher improved 15 of 32 unseen exact roots, but development selected the student at initialization. On the reused training distribution, 87.7% of root-balanced mass has all legal actions equally valued; only 4.24% supplies a strict changed teacher action. This motivates testing action costs; it does not prove the reason distillation failed.

## One comparison

Reuse only the immutable 32 TRAIN roots and 622 exact-Q rows. Both arms start from the same original splat parent, with identical clean actor inputs, fixed G0 partner/pinned Zeb opponents, minibatches, optimizer and 64 updates. The cost arm minimizes `sum_legal pi(a) * (max Q - Q(a))`; the control imitates the same teacher-selected action with legal cross entropy. All-action ties have zero cost actor gradient, although weight decay and momentum can still change weights. Raw success-probability cost scale is deliberately different from CE; no learning-rate tuning is bundled into the comparison.

These Q values assume optimal teacher continuation. The deployed student follows itself, so a lower loss is not a complete-policy improvement guarantee. This is a narrow offline cost-sensitive imitation step related to [AggreVaTe](https://arxiv.org/abs/1406.5979), not that full interactive aggregation algorithm or its guarantees.

## Fresh judgment and bounded execution

All 608715 previously reserved physical deals are excluded from fresh pools. Reserve 896 new deals: 256 development, 512 test and 128 conditional full hands. Select the first structurally eligible 16 development and 32 test late-game roots, without screening on outcome or teacher gain. No old development/test data or selected student checkpoint enters the successor. Development selects among steps 0/16/32/48/64 by exact complete-policy success, with earliest tie. Test opens only after selection is frozen.

Finite improvement requires at least 0.5 percentage points over both parent and CE, with both paired physical-root bootstrap lower bounds positive. Only a pass opens the registered 128 full hands; that separate gate requires 2 points over parent and 1 over CE, with positive lower bounds. No automatic promotion or test reselection.

The finite teacher retains the explicit uniform legal-consistent root approximation, carries masses through later observations, and shares each learner action across indistinguishable worlds. No search runs inside deployed Gus. Training roots are statically reused, never resimulated. Collection/training/development/test/full-hand caps remain 16/2/4/4/8 short resumable segments, 45 seconds cooperative and 90 seconds outer, through `scripts/cpu-slot`. Failures, unclosed attempts or exhausted caps require inspection.

## Review and provenance

The implementation's cost/tie/illegal-gradient, copied-Q provenance, matched optimizer/RNG resume and integration fixtures pass. Two independent readers cleared trainer and split/recovery boundaries without model calls. Torch version is recorded but not automatically enforced by trainer resume: use the pinned runtime and inspect any environment change. Final hashes, public verification, exact command and subsequent outcomes belong in the study receipts before any result claim.

All 896 potential new deals become future exclusions, including unused candidates: 609611 total with prior reservations. Full protocol: `scratch/lem/gus-contingent-cost-v1/PROTOCOL.md`. The [growing First Look report](https://gus-first-look.psychometrix.chatgpt.site/#contingent) separates teacher improvement, learned transfer and playing strength.
