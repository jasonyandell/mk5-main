---
title: "Source digest: 8c5fbca — single-GPU STaR loop with vLLM batch inference"
kind: source
first_seen: 2026-04-10
last_updated: 2026-04-10
status: active
---

## Commit

- **SHA:** 8c5fbca7b764b121bee8327350bd6c5c29d76892
- **Date:** 2026-04-10
- **Author:** Jason Yandell

> feat(lem): single-GPU STaR loop with vLLM batch inference (t42-m84g)
>
> star_loop.py — one Modal function, one GPU, model loaded once:
>   1. vLLM batch inference on all prompts simultaneously
>   2. K1 grading (beat the bot)
>   3. vLLM batch rationalization of failures
>   4. LoRA training on winning traces + rationalizations
>   5. Push adapter to HF, repeat
>
> Target: ~4 min/iteration on B200, ~8 min on H100.
> Replaces separate inference + training scripts for iteration speed.
>
> Also: train_star.py (standalone STaR training), iterate.sh (manual loop),
> star_harness.py bumped to A10G.

## Files introduced / modified

| Path | Change |
|---|---|
| `lem/gemma_star/star_loop.py` | New, 417 LOC. One [[modal]] function, one GPU, continuous STaR loop (5 steps per iteration, described above) |
| `lem/gemma_star/train_star.py` | New, 238 LOC. Standalone LoRA training step for STaR traces (can be used outside the loop) |
| `lem/gemma_star/iterate.sh` | New, 64 LOC. Manual shell loop wrapper for orchestrating iterations outside Modal |
| `lem/gemma_star/star_harness.py` | Modified: GPU bumped from L4 to A10G |

## Key design

`star_loop.py` consolidates what the earlier [[star-harness]] did in separate steps into a single-GPU continuous loop. Loading the model once per [[modal]] function call (rather than once per step) is the primary throughput optimization. vLLM replaces single-sample inference for batch efficiency.

Throughput targets: ~4 min/iteration on B200, ~8 min on H100. These replace the per-step latency of the earlier harness.

The loop body per iteration:
1. vLLM batch inference (all prompts simultaneously)
2. [[k1-grading]] (beat the bot)
3. vLLM batch [[r1-rationalization]] of failures
4. LoRA training on winning traces + rationalizations
5. Push adapter to HuggingFace, repeat

Bead t42-m84g closed.

## Related pages

[[lem]] · [[star]] · [[star-harness]] · [[k1-grading]] · [[r1-rationalization]] · [[modal]] · [[gemma-4-e2b]] · [[lora-unsloth]] · [[7538016]] · [[6e71df9]]
