---
title: Base Model K1 Baseline (10 Examples, Local)
kind: experiment
first_seen: 2026-04-10
last_updated: 2026-04-10
status: active
---

## Summary

[[k1-grading]] baseline measurement for base [[gemma-4-e2b]] (no adapter) on 10 trick-6 decisions from the training narration corpus. Run locally via llama.cpp on CPU using the Q4_K_M GGUF quantization. Establishes the floor that [[star]] iterations must beat.

([lem/gemma_star/local_star.py @ f578bfa](../sources/f578bfa.md))

## Setup

- **Model:** base [[gemma-4-e2b]], Q4_K_M GGUF quantization, no adapter
- **Runtime:** llama.cpp, CPU, local machine (streaming thinking channel)
- **Examples:** 10 trick-6 decisions from `lem/data/narrations_train.jsonl`
- **Grading:** same `parse_play` / `grade_k1` functions as the [[modal]] [[star-harness]], ensuring identical grading logic
- **K1 criterion:** `E[Q][gemma] >= E[Q][bot]`

## Results

| Outcome | Count | % |
|---|---|---|
| K1 pass | 6 | 60% |
| Legal but suboptimal (K1 fail) | 3 | 30% |
| Illegal play | 1 | 10% |
| Parse fail | 0 | 0% |

## Interpretation

60% K1 pass on base [[gemma-4-e2b]] (no adapter, no STaR) reflects a structural property of the grading criterion rather than model competence: the bot is itself [[expected-q-value]]-greedy, so K1 reduces to "did Gemma pick an argmax-tied action?" A large fraction of trick-6 decisions have an obvious enough best move that even an uneducated 2B model stumbles into it.

Implication: [[star]] improvement headroom over base is smaller than Q&A-style accuracy gaps would suggest. The meaningful ceiling is not 100% but something closer to the fraction of trick-6 decisions where the argmax is non-obvious. See [[k1-grading]] for ceiling-awareness discussion.

This is the number to beat with [[stage-0-adapter]] + STaR iterations.

## Related pages

[[lem]] · [[gemma-4-e2b]] · [[star]] · [[star-harness]] · [[k1-grading]] · [[expected-q-value]] · [[decisions/eval-seed-holdout]] · [[experiments/star-harness-5ex-smoke]] · [[sources/f578bfa]]
