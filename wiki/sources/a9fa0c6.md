---
title: "Source: a9fa0c6"
kind: source
commit: a9fa0c6
date: 2026-04-22
author: Jason Yandell
---

## Commit message

> feat(train): Q_head partial-depletion augmentation fine-tuner
>
> Freezes all trunk params (encoder, belief, v_head, pi_me), trains only
> q_head + world_encoder. With p=aug_prob, zeros k random assigned domino
> rows from world_assign before the forward pass. Target q_per_world unchanged
> — Q should be invariant to whether played dominos are still in assignment.
>
> Test of path (a): if Q_head learns OOD robustness, lamir1-qleaf with Bug 6
> enabled should approach or beat the 0.551 direct baseline.

## Files changed

| File | Change |
|---|---|
| `gus/train/train_q_aug.py` | New — Q_head augmentation fine-tuner |

## What this commit establishes

Path (a) implementation: augmentation fine-tuner that randomly zeros domino rows from
world_assign to teach Q_head robustness to depleted leaf states. Test of the hypothesis
that OOD robustness can be learned post-hoc.

## Links

[[experiments/gus-q-head-augmentation]] · [[topics/lamir1]]
