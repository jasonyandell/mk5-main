---
title: LoRA via Unsloth
kind: topic
first_seen: a8bccfa
last_updated: 24ae55a
status: active
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

- **Stage 0 ([[rules-adapter]])** — LoRA step on rules primer + Q&A corpus. Stage 0 v1 complete.
- **Stage 1+ ([[star]])** — LoRA step after each round of kept traces and rationalizations.

## Open questions

- LoRA rank, learning rate, and epochs remain open hyperparameters for Stage 1+. Stage 0 used Unsloth's Gemma 4 defaults; tuning is guided by wandb (lem/OVERVIEW.md @ 24ae55a). (?)

## Links

[[gemma-4-e2b]] [[rules-adapter]] [[lem]] [[modal]] [[stage-0-adapter]] [[experiments/stage-0-v1-training]]
