---
title: "STaR Stage 1: 15 Iterations"
kind: experiment
first_seen: efad16e
last_updated: 908773a
status: active
---

## Summary

15 chained [[star]] iterations on [[gemma-4-e2b]], starting from [[stage-0-adapter]]. First sustained training run to show meaningful pass-rate improvement. Pass rate moves from 30% (iter 0) to a plateau of 36–42%, with best checkpoints at iter 5 and iter 7 (both 42%). Iterations 10–14 confirm the plateau; a ceiling hypothesis is proposed.

([lem/OVERVIEW.md @ efad16e](../sources/efad16e.md), updated at [[sources/908773a]])

## Setup

- **Platform:** [[modal]] B200 ($6.25/hr)
- **Initial adapter:** [[stage-0-adapter]] (`jasonyandell/gemma-4-e2b-texas42-stage0`)
- **Chain:** each iteration starts from the prior iteration's adapter
- **Data pool (iters 0–4):** 3148-example dataset, seeds 0–199; subset 200/iter
- **Data pool (iters 5–9):** 7409-example combined dataset, seeds 0–499 ([[sources/ff0d0d2]]); subset 300/iter
- **Subset sampling:** random draw per iteration (different examples each run)
- **Inference:** HF `model.generate()` with SDPA + `torch.compile` + left-padded batching; 120+ tok/s
- **Grading:** [[k1-grading]]; illegal/parse-fail discarded per [[decisions/discard-illegal-traces]]; [[r1-rationalization]] on legal failures
- **Cost:** total ~$25 on B200 for 15 iters (~$1.67/iter; the earlier ~$0.26 figure was a 5-example smoke test)
- **Wandb:** `jasonyandell-forge42/lem-star`

## Results

| Iter | Pass | Loss | Examples | Pool | Notes |
|------|------|------|----------|------|-------|
| 0 | 30% | 31.6 | 200 | 3148 | Stage 0 baseline |
| 1 | 34% | 17.5 | 200 | 3148 | |
| 2 | 33% | 21.2 | 200 | 3148 | |
| 3 | 36% | 15.5 | 200 | 3148 | |
| 4 | 35% | 19.7 | 200 | 3148 | |
| 5 | **42%** | 12.8 | 300 | 7409 | Bigger pool |
| 6 | 36% | 12.3 | 300 | 7409 | |
| 7 | **42%** | 12.9 | 300 | 7409 | |
| 8 | 38% | 11.4 | 300 | 7409 | |
| 9 | 36% | 11.8 | 300 | 7409 | |
| 10 | 39% | ~10 | 300 | 7409 | plateau continues |
| 11 | 41% | ~10 | 300 | 7409 | |
| 12 | 40% | ~10 | 300 | 7409 | |
| 13 | 39% | ~10 | 300 | 7409 | |
| 14 | 38% | ~10 | 300 | 7409 | |

## Adapters published

15 adapters on HuggingFace: `jasonyandell/gemma-4-e2b-texas42-star-iter0` through `-iter14`. Iter 5 and iter 7 are the best-performing checkpoints (both 42%).

## Key observations

> The model is learning — 30% to 42% is a real improvement. Plateau around 38% average suggests we need a different signal to push further. Bigger data pool (3148 → 7409) with different subsets per iteration helped. Chaining as separate Modal runs (1 iteration each) avoids container staleness.

— lem/OVERVIEW.md at [[sources/efad16e]]

Iter 5 (first iteration with the 7409-example pool) is also the first 42% checkpoint, suggesting data diversity contributed to the jump. The loss trend (31.6 → 11.8) is consistent across iterations.

## Plateau confirmed — ceiling hypothesis

Iters 10–14 (39%, 41%, 40%, 39%, 38%) show no improvement over iters 5–9. Loss stable at ~10. The plateau is confirmed.

> The plateau at ~40% likely reflects the ceiling of K1 grading without fact-verification. The model may be learning wrong game-facts that happen to produce correct plays ~40% of the time but can't go further because the reasoning is polluted. This was the original concern that motivated the scratchpad approach.

— lem/OVERVIEW.md at [[sources/908773a]]

## Infrastructure notes

- **LoRA adapter key mismatch resolved:** Gemma 4 E2B uses KV-sharing for layers 15–34 — those layers don't have `k_proj`/`v_proj` because they reuse KV states from layers 0–14. The adapter is complete and correctly covers all trainable layers.
- **Scratchpad validation attempted and deferred** during this period — see [[experiments/scratchpad-v2-iter0]] and [[scratchpad-validation]].
- **Iterations chained as separate Modal runs** to avoid container staleness (one-iteration-per-invocation pattern).

## Open directions — none taken

None of these three candidates (held-out E[Q]-delta eval, scratchpad-SFT bootstrap, another
Stage-0 round) happened. Five days later the project replaced the base model entirely
(Gemma → Qwen 3 1.7B, [[base-model-pivot-qwen]]) rather than pursuing any of them.

1. ~~Run held-out eval (seeds 900000–909999, see [[decisions/eval-seed-holdout]]) on iter 5 or iter 7 to get a proper [[expected-q-value]] delta measurement vs Stage 0 and vs base model.~~
2. ~~Bootstrap [[scratchpad-validation]] format via SFT — generate correct scratchpad examples from the engine, train one LoRA pass to teach the format, then resume validated STaR.~~
3. ~~11,672 narrations now available (seeds 0–799) for larger-subset iterations.~~ (What actually happened instead: [[experiments/stage-0-progression-star]] found the plateau was curriculum-bounded, not something more STaR iteration or scratchpad validation would fix.)

## Related pages

[[lem]] · [[star]] · [[k1-grading]] · [[r1-rationalization]] · [[learned-by-playing]] · [[scratchpad-validation]] · [[stage-0-adapter]] · [[gemma-4-e2b]] · [[modal]] · [[decisions/eval-seed-holdout]] · [[decisions/discard-illegal-traces]] · [[experiments/scratchpad-v2-iter0]] · [[sources/ff0d0d2]] · [[sources/efad16e]] · [[sources/908773a]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 1 correction applied in place and independently re-verified.

- HuggingFace adapters (`star-iter0`–`iter14`) and the `jasonyandell-forge42/lem-star` wandb project are external artifacts — names match repo docs but were not fetched, so the numbers trace only to lem/OVERVIEW.md.
- Page title says "15 Iterations" while the filename is `star-10-iterations`; harmless, but a redirect note could prevent confusion.
