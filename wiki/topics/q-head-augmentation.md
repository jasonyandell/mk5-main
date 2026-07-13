---
title: Q_head Partial-Depletion Augmentation
kind: topic
first_seen: 2026-04-22
last_updated: 2026-04-22
status: retired
---

## What it is

An attempt to fix the [[lamir1-ceiling]] by making the Q_head robust to mid-rollout `world_assignment` tensors. During [[lamir1]] rollout, 1-3 opponent plays deplete the world hands, but Q_head was trained exclusively on initial-deal assignments (all 21 opponent dominos assigned). At rollout leaves, the assignment has depleted rows — an out-of-distribution input.

The augmentation fine-tuner (`gus/train/train_q_head_augmented.py`) freezes the entire trunk (encoder, belief, v_head, π_me) and trains only `q_head + world_encoder`. With probability `aug_prob`, it zeros `k` random assigned-domino rows from `world_assign` before the forward pass, where `k ~ Uniform(1, 3)`. The Q target (`q_per_world`) is left unchanged — the intent is that Q should be invariant to whether played dominos remain in the assignment (a9fa0c6).

This was "path (a)" in the [[lamir1-ceiling]] pivot options: if Q_head learns OOD robustness, then enabling Bug 6 (zeroing played rows at rollout leaves) should move lamir1-qleaf toward the 0.551 direct π_me baseline.

## Training

- **Frozen**: encoder, belief, v_head, π_me
- **Trainable**: q_head + world_encoder (160,007 params)
- **Epochs**: 15, lr=5e-5
- **Corpus**: 1000-game v2 diverse-seed train corpus
- **Best eval q_mae**: 8.277 → 8.169 (epoch 10)
- **Adapter**: `gus/adapters/q_head_aug.pt`

## Result

| mode | regret | bot-match |
|---|---:|---:|
| lamir1-qleaf + Bug 6 (baseline) | 2.156 | 63.2% |
| lamir1-qleaf + Bug 6 + aug Q_head | **2.216** | 63.8% |

Slightly worse. The small q_mae improvement (0.1 pts) did not translate to better rollout decisions (5f390fb).

## Why path (a) closed

The augmentation fixes a measurement artifact — q_mae on depleted inputs — but not the fundamental ordering problem. Scalar noise from distillation is large relative to the action-value gap at decision boundaries. The leaf evaluator would need to be trained end-to-end in rollout context (path b, a look-ahead-compatible V-head) to fix this. Augmenting the input distribution alone is insufficient (5f390fb, PRACTICALITIES §20).

Additionally, the Bug 6 "fix" (zeroing played domino rows) was itself reclassified: the training convention preserves the original `world_assignment` as `played_mask` advances, so both the augmentation and the zeroing were fighting a training invariant rather than correcting a real bug (b42669a).

## Links

[[lamir1]] [[lamir1-ceiling]] [[student-distillation]] [[gus]] [[expected-q-value]]
