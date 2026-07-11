---
title: STaR Stage 1 Iteration 0
kind: experiment
first_seen: 576b694
last_updated: 576b694
status: active
---

## Summary

First full end-to-end [[star]] iteration on [[gemma-4-e2b]] with [[stage-0-adapter]] as the starting point. Run on [[modal]] H100. Proves the loop produces training traces from an actual iteration; does not yet measure E[Q] delta on held-out eval seeds.

([lem/OVERVIEW.md @ 576b694](../sources/576b694.md))

## Setup

- **Platform:** [[modal]] H100
- **Initial adapter:** [[stage-0-adapter]] (`jasonyandell/gemma-4-e2b-texas42-stage0`)
- **Dataset:** `lem/data/narrations_train.jsonl`, seeds 0–199, 3148 examples (see [[narration]])
- **Batch size:** 10 examples
- **Inference:** sequential HF `model.generate()` (vLLM removed in [[sources/8724e93]])
- **Grading:** [[k1-grading]]; illegal/parse-fail traces discarded per [[decisions/discard-illegal-traces]]
- **Failure handling:** [[r1-rationalization]] on legal failures only

## Results

| Outcome | Count | % |
|---|---|---|
| K1 pass | 3 | 30% |
| Legal fail, rationalized | 3 | 30% |
| Illegal, discarded | 4 | 40% |
| Parse fail | 0 | 0% |

- Training traces collected: 6 (3 wins + 3 rationalizations)
- LoRA trained in 15s, loss = 0.11
- Adapter pushed: `jasonyandell/gemma-4-e2b-texas42-star-iter0`
- Wandb: `jasonyandell-forge42/lem-star`
- Total iteration time: ~15 min. Cost: ~$1.

## Significance

The 40% illegal rate confirms the [[decisions/discard-illegal-traces]] diagnostic hypothesis: a model trained only on Q&A still gets rules wrong in narration context. This is consistent with the [[learned-by-playing]] prediction that rules comprehension will improve through STaR iteration, not additional Q&A drilling. The illegal rate is the primary thing to watch decline across iterations.

The 30% K1 pass rate is lower than the 60% base-model baseline measured in [[experiments/base-model-k1-baseline]]. This is expected: adding [[stage-0-adapter]] improved state-tracking but did not improve strategic play, and the examples that benefit from the adapter's hand-tracking fixes may overlap more with the harder strategic decisions.

## Infrastructure note

vLLM was the original batch inference plan (see [[sources/8c5fbca]]). It was removed before this run due to version conflicts with Gemma 4's tokenizer format. Sequential HF `model.generate()` is used instead (~1 prompt/min on H100).

## Related pages

[[lem]] · [[star]] · [[star-harness]] · [[k1-grading]] · [[r1-rationalization]] · [[stage-0-adapter]] · [[gemma-4-e2b]] · [[modal]] · [[learned-by-playing]] · [[decisions/discard-illegal-traces]] · [[narration]] · [[experiments/base-model-k1-baseline]] · [[sources/576b694]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- HF adapters (`jasonyandell/gemma-4-e2b-texas42-star-iter0`) and wandb project `jasonyandell-forge42/lem-star` are external artifacts, not verifiable from the repo.
- Cheap next probe: n=10 cannot distinguish 30% vs 60% pass rates; a 50–100 example paired run (base vs [[stage-0-adapter]], same hardware and inference path) would settle whether the adapter regressed K1 pass or the delta is sampling plus llama.cpp-vs-HF confound.
