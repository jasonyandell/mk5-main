---
title: "Source: c04bda3"
kind: source
commit: c04bda3
date: 2026-04-20
author: Jason Yandell
first_seen: 2026-04-24
last_updated: 2026-04-24
---

## Commit message

> feat(gus): v0 student scaffolding — belief head + dataset + trainer + eval
>
> Per BUILD_PLAN.md. v0 = state encoder + belief-only head as a sanity
> check before the full five-head LAMIR-ready architecture.
>
> - gus/model/features.py: flat 183-dim state features (decl + player +
>   decision_idx + 6×28 domino masks: my_hand, played_total, played_by_seat×4).
>   Reconstructs prior plays from (game.hands, action_taken slots).
> - gus/model/dataset.py: JointWorldDecisionDataset — one sample per
>   decision, yields (features, belief_target, belief_mask).
> - gus/model/student.py: StateEncoder (MLP) + BeliefHead + StudentV0.
>   ~200K params. belief_loss + belief_accuracy helpers.
> - gus/train/train_v0_belief.py: AdamW, saves best-eval-acc checkpoint.
> - gus/eval/eval_belief.py: top-1 overall + bucketed by decision_idx
>   (belief should sharpen as the game progresses).
>
> Smoke-tested on the 1-game tire-kick corpus: trainer and eval run clean
> end-to-end, model overfits the tiny corpus to 100% in 3 epochs (as expected).
> Real evaluation awaits the 100-game training corpus currently generating.

Establishes the full v0 pipeline skeleton: features, dataset, model, trainer, and eval.
Smoke test on 1-game corpus confirms end-to-end plumbing. Real numbers await the 100-game
corpus generated in parallel (see [[experiments/gus-v0-v1-belief]]).

## Links

[[gus]] · [[experiments/gus-v0-v1-belief]]
