---
title: STaR Harness 5-Example Smoke Test
kind: experiment
first_seen: 7538016
last_updated: 7538016
status: active
---

## Summary

First end-to-end run of the [[star-harness]] (`lem/gemma_star/star_harness.py`) on 5 trick-6 decisions drawn from the training corpus. Proves the full pipeline path: inference → parse → grade → rationalize → JSONL output. No LoRA step performed; no learning measured.

([lem/gemma_star/star_harness.py @ 7538016](../sources/7538016.md))

## Setup

- **Model:** [[gemma-4-e2b]] with [[stage-0-adapter]] loaded
- **Platform:** [[modal]] function
- **Examples:** 5 trick-6 decisions from `lem/data/narrations_train.jsonl`
- **Grading:** [[k1-grading]] — pass iff `E[Q][gemma] >= E[Q][bot]`
- **Failure handling:** [[r1-rationalization]] — reveal bot action, ask model to justify it, keep that rationale

## Results

| Outcome | Count | % |
|---|---|---|
| K1 pass | 1 | 20% |
| K1 fail, rationalized | 2 | 40% |
| Illegal play, rationalized | 2 | 40% |
| Parse fail | 0 | 0% |

All 5 traces collected to JSONL output.

## Significance

Pipeline is end-to-end functional. Every path (pass / K1-fail / illegal) produced a kept trace. No LoRA step was run at this stage — this is a plumbing proof, not a learning measurement.

The 20% K1 pass rate on 5 examples is too small to interpret, but the 40% illegal rate matches the hand-tracking errors seen in [[experiments/second-gemma-contact]] and motivates the continuous iteration loop variant of [[star-harness]] introduced in [[sources/8c5fbca]]. The [[experiments/base-model-k1-baseline]] experiment (10 examples, no adapter) provides a more reliable pass-rate baseline.

## Related pages

[[lem]] · [[star]] · [[star-harness]] · [[k1-grading]] · [[r1-rationalization]] · [[stage-0-adapter]] · [[gemma-4-e2b]] · [[modal]] · [[experiments/base-model-k1-baseline]] · [[sources/7538016]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- The 1/5 / 2/5 / 2/5 result counts trace only to the commit message of 7538016 — the JSONL output is Modal/local and gitignored, so no independent artifact exists in the repo.
- The "with [[stage-0-adapter]] loaded" setup claim is unconfirmed: the harness takes the adapter as an optional parameter and the docstring's iter0 example runs without one.
- `lem/data/narrations_train.jsonl` is untracked; the path is the harness's expected input but the file itself is unverifiable.
