---
title: Second Gemma Contact
kind: experiment
first_seen: df73c8d
last_updated: df73c8d
status: active
---

## Summary

Second inference pass against [[gemma-4-e2b]], using the same prompt as [[experiments/first-gemma-contact]] (seed 42, fives trump, trick 6), but with [[stage-0-adapter]] loaded. Tests whether Stage 0 Q&A drilling transferred to the narration-context reasoning task.

([lem/gemma_star/modal_app.py @ df73c8d](../sources/df73c8d.md))

## Setup

- **Prompt:** identical to [[experiments/first-gemma-contact]] — seed 42, fives trump, truncated at trick 6 narrator turn, rules primer prepended
- **Adapter:** `jasonyandell/gemma-4-e2b-texas42-stage0` ([[stage-0-adapter]]) loaded via `PeftModel.from_pretrained`, merged and unloaded before inference
- **Infrastructure:** [[modal]] L4 GPU, bf16, thinking mode enabled

## Results

| Dimension | Base Gemma (first contact) | + Stage 0 adapter |
|---|---|---|
| Hand tracking | Confused initial/remaining hand; played 5-5 (already gone) | Correctly read "remaining: 6-2, 6-1" |
| Final answer | "Play 5-5" — illegal | "Sluff 6-2 or 6-1" — legal, correct |
| Trump membership | Called 6-4 a trump under fives | Still calls 4-4, 6-4 trumps (same error) |
| Reasoning | Good structure, wrong state | Better structure, correct state |

([lem/OVERVIEW.md @ 24ae55a](../sources/24ae55a.md))

## Interpretation

Hand tracking — the #1 error from first contact — is fixed. Factual state-reading (which dominoes remain) transferred from Q&A format to narration context.

Trump membership errors persist. 4-4 and 6-4 are not trumps when fives are trump, yet the model still calls them trumps. Q&A-format drilling on trump membership rules did not produce reliable compositional rule application in the narration context.

The key insight, per the OVERVIEW: trump rules are the kind of thing learned by playing and being corrected, not from flashcards. This motivates Stage 1 [[star]] over more Q&A drilling. See [[learned-by-playing]].

## Significance

Proves that [[rules-adapter]] Q&A training transfers for factual state-tracking but not for compositional rule application. The remaining gap is exactly what [[star]] K1 grading ([[k1-grading]]) will correct: trump errors cause bad plays, and [[r1-rationalization]] teaches the model why the bot's move was better.

## Related pages

[[lem]] · [[gemma-4-e2b]] · [[stage-0-adapter]] · [[rules-adapter]] · [[star]] · [[k1-grading]] · [[r1-rationalization]] · [[learned-by-playing]] · [[experiments/first-gemma-contact]] · [[experiments/stage-0-v1-training]] · [[modal]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 1 correction applied in place and independently re-verified.

- The raw inference transcript lives only in Modal/W&B logs, not in-repo; the results table traces to `lem/OVERVIEW.md` prose, not raw output.
