---
title: Stage 0 v5 Adapter (Qwen 3 1.7B)
kind: entity
first_seen: 2026-04-16
last_updated: 2026-04-17
status: superseded
superseded_by: v9-adapter
---

## What it is

The Stage 0 v5 Adapter is the first LoRA fine-tune of [[qwen3-1.7b]] for Texas 42, trained
on the same [[topics/game-context-qa]] corpus used for [[v4-adapter]]. It is published at
HuggingFace as `jasonyandell/qwen3-1.7b-texas42-stage0-v5` and is the new Stage 0 base for
[[lem]] Stage 1 STaR iterations. (commit message @ 3465e29)

## Training provenance

- **Base model**: [[qwen3-1.7b]]
- **Training data**: same game-context Q&A corpus as [[v4-adapter]] (see [[topics/game-context-qa]])
- **Method**: [[lora-unsloth]] + xformers on [[modal]] B200
- **Throughput**: 36K tok/s, ~19 min/training run
- **HF repo**: `jasonyandell/qwen3-1.7b-texas42-stage0-v5`

(commit message @ 3465e29)

## Eval results

Comprehension eval v5: **100%** (vs [[v4-adapter]] on Gemma's 60% on the same eval).
This is the decisive factor in the [[decisions/base-model-pivot-qwen]] decision.

## Role in pipeline

Was the initial Stage 0 checkpoint on [[qwen3-1.7b]]. Superseded by [[v9-adapter]] (14
categories, 83% comprehension) and subsequently by [[v10-adapter]] (joint rationalization)
and v10-maskfix (completion-only loss, 86% comprehension). (commit messages @ b857299, be7efc4)
