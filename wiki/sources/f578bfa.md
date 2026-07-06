---
title: "Source digest: f578bfa — local STaR runner with streaming output for interactive debugging"
kind: source
first_seen: f578bfa
last_updated: f578bfa
status: active
---

## Commit

- **SHA:** f578bfa5cad10ab6596bc06182612814977f081a
- **Date:** 2026-04-10
- **Author:** Jason Yandell

> feat(lem): local STaR runner with streaming output for interactive debugging
>
> Runs Gemma 4 E2B locally via llama.cpp (CPU, Q4_K_M GGUF). Streams the
> thinking channel in real time so you can watch the model reason about 42.
> Reuses parse_play/grade_k1 from the Modal harness.
>
> Baseline measurement (10 examples, base model, no adapter):
> - 60% pass rate (matched bot's argmax E[Q] play)
> - 30% fail (legal but suboptimal)
> - 10% illegal (hand tracking error)
> - 0% parse fail
>
> This is the number to beat with Stage 0 adapter + STaR iterations.

## Files introduced

| Path | LOC | Purpose |
|---|---|---|
| `lem/gemma_star/local_star.py` | 168 | Local CPU runner: llama.cpp, Q4_K_M GGUF quant, streaming thinking channel; reuses `parse_play`/`grade_k1` from [[star-harness]] |

## Key details

- Enables interactive inspection of the model's thinking channel without incurring [[modal]] GPU cost.
- Grading logic (`parse_play`, `grade_k1`) imported directly from `star_harness.py` — ensures local and Modal runs use identical grading.
- Baseline result on 10 examples, base [[gemma-4-e2b]] (no adapter): 60% K1 pass, 30% legal-suboptimal fail, 10% illegal, 0% parse-fail.
- The 60% baseline reflects the structural ceiling of [[k1-grading]] on a greedy-bot opponent (see [[experiments/base-model-k1-baseline]] for interpretation).

## Related pages

[[lem]] · [[star]] · [[star-harness]] · [[k1-grading]] · [[gemma-4-e2b]] · [[experiments/base-model-k1-baseline]] · [[sources/7538016]]
