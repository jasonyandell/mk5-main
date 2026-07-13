---
title: LoRA via Unsloth
kind: topic
first_seen: 2026-04-09
last_updated: 2026-04-17
status: superseded
---

## Overview

LoRA (Low-Rank Adaptation) via the Unsloth library is the fine-tuning method [[lem]] uses at every training stage. Unsloth provides a ready recipe for [[gemma-4-e2b]], making it the practical choice for the hardware available (lem/OVERVIEW.md @ a8bccfa).

## Hardware context

Stage 0 training ran on a [[modal]] L4 GPU (22GB VRAM). [[gemma-4-e2b]] is ~20GB in bf16, leaving minimal headroom for activations. LoRA is the only viable fine-tuning approach at this scale on this hardware (lem/OVERVIEW.md @ 24ae55a).

## Training recipe (Stage 0 v1)

The following steps are required; omitting any one of them causes silent failure or OOM. See [[stage-0-adapter]] and [[experiments/stage-0-v1-training]] (lem/OVERVIEW.md @ 24ae55a):

1. **Monkey-patch `Gemma4ClippableLinear`** to inherit from `nn.Linear` before loading. Without this fix, `grad_norm` stays 0 and the adapter never updates. Fix from huggingface/peft#3129.

2. **Use bf16, not fp16.** `GradScaler` fails under gradient checkpointing with Gemma 4; bf16 sidesteps the issue entirely.

3. **`gradient_checkpointing=True` with `use_reentrant=False`.** Required because the model is ~20GB in bf16 and the L4 has 22GB — no room for activations without checkpointing.

4. **Disable eval (`eval_strategy="no"`).** The eval forward pass OOMs on an L4. Training converges to 100% token accuracy by step 50; eval is redundant.

5. **1 epoch is enough.** Loss goes 32 → 15 → 2.8 → 0.001 in the first 40 steps (lem/OVERVIEW.md @ 24ae55a).

## Usage in LEM stages

- **Stage 0 ([[rules-adapter]])** — LoRA step on rules primer + Q&A corpus. Ten curriculum
  rounds ran on this recipe: v1 (3.5k Q&A, Gemma), Kerry (15k, Gemma), v3 (20k + trump
  drill, Gemma), v4 (31k game-context, Gemma, 67% comprehension), v5 (same corpus, pivoted
  to [[qwen3-1.7b]], 100% comprehension — see [[base-model-pivot-qwen]]), v7–v9
  (14-category expansion, 83% comprehension on 1.7B / 86% on [[qwen3-14b]]), and v10 /
  v10-maskfix (joint rationalization + completion-only loss, 86% comprehension, 55/100
  bot-match). LEM ended at v10-maskfix; see [[lem]].
- **Stage 1+ ([[star]])** — LoRA step after each round of kept traces and rationalizations,
  through 15 STaR iterations on the Gemma base (plateau ~40%, see [[k1-grading]]).

## Open questions — resolved

LoRA rank, learning rate, and epochs were an open hyperparameter question through the LEM
arc (Stage 0 used Unsloth's Gemma 4 defaults; Stage 1+ tuning was guided by wandb without a
settled recipe). The question was resolved later, during [[burl]]'s STaR work: conservative
hyperparameters — rank=8 (not 16), lr=3e-5 (not 1e-4), 1 epoch, with val-loss early-stopping
— are the settled recipe for small corpora. See [[star]] "Burl 2000-decision corpus ready"
section, point 3. LEM itself never converged on a rank/LR recipe before pivoting.

## Links

[[gemma-4-e2b]] [[qwen3-1.7b]] [[rules-adapter]] [[lem]] [[burl]] [[modal]] [[stage-0-adapter]] [[star]] [[k1-grading]] [[experiments/stage-0-v1-training]]
