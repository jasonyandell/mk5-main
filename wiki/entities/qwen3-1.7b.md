---
title: Qwen 3 1.7B
kind: entity
first_seen: 2026-04-16
last_updated: 2026-04-17
status: complete
---

## What it is

Qwen 3 1.7B is the base model for [[lem]] from commit 3465e29 (2026-04-16) onward. It
replaces [[gemma-4-e2b]], which was retired as the LEM base due to architectural constraints
and lower comprehension accuracy. (commit message @ 3465e29)

## Why it replaced Gemma 4 E2B

| Dimension | Gemma 4 E2B | Qwen 3 1.7B |
|---|---|---|
| Comprehension eval v5 | 60% | 100% |
| B200 throughput | — (underutilized) | 36K tok/s |
| Training run time | — | ~19 min on B200 |
| Flash-attention-2 | Not supported | Supported |
| KV-sharing quirks | Layers 15–34 (blocks LoRA-in-vLLM) | None |
| PLE (param-embedding layer) | Present | Not present |
| Fine-tune tooling | Unsloth (with monkey-patch) | Unsloth + xformers |

The B200 underutilization bead (t42-hv08) was resolved by switching models rather than
tuning GPU/kernel configuration. (commit message @ 3465e29; see [[base-model-pivot-qwen]])

## Properties

- 1.7B parameters
- Apache 2.0 license
- Supports flash-attention-2
- Works with Unsloth + xformers on [[modal]] B200

## Adapter lineage

| Adapter | Comprehension | Notes |
|---|---|---|
| [[v5-adapter]] | 100% (v5 eval) | Initial Qwen Stage 0; 5 categories |
| [[v9-adapter]] | 83% (14 cats) | Structured templates + verifier; rationalization ~68/100 |
| [[v10-adapter]] (original) | 83% | + joint rationalizations; 55/100 bot-match |
| [[v10-adapter]] (maskfix) | **86%** | = [[qwen3-14b]] v9 at 1/3 cost; bot-match still 55/100 |

Current best: `jasonyandell/qwen3-1.7b-texas42-stage0-v10-maskfix` — 86% comprehension.
See [[v10-maskfix-breakthrough]]. (commit messages @ b857299, 0c7392f, be7efc4)

## End state

LEM ended at v10-maskfix with no further Qwen 3 1.7B training. [[burl]] uses [[gemma-4-e2b]]
instead — Qwen 3 1.7B was never carried forward. See [[lem]] and [[lem-to-burl-handoff]].
