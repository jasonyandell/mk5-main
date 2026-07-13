---
title: "iter-5 E1: LoRA Rank Sweep with Truncation Fixed"
kind: experiment
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Summary

First real [[preserve-thoughts]] adapter trained with `max_seq_length=4096` (truncation fixed per [[decisions/sft-max-seq-length]]). Sweeps LoRA rank 16/64/128 on a 26-row corpus. Rank-16 is the sweet spot; higher rank degrades catastrophically on small corpora.

([burl/experiments/iter5_e1_capacity_eval_writeup.md @ ceca203](../sources/ceca203.md))

## Setup

- **Corpus:** 26-row `preserve_thoughts` corpus, `max_seq_length=4096`
- **Recipe:** 3 epochs, matched across all ranks, [[mlx-lm]] on M5 Max
- **Eval:** same 10-decision held-out set (rank-16 completed 10/10; base and rank-64 completed 9/10; rank-128 completed 0/10)

## Results

| Variant | Bot-match | Eq-delta | Notes |
|---|---|---|---|
| Base Gemma | 66.7% | −3.15 | Zero-shot baseline |
| rank-16 | **70.0%** | **−2.83** | First real preserve_thoughts adapter |
| rank-64 | 55.6% | −5.59 | 50% empty-tool-rollout |
| rank-128 | 0.0% | — | 10/10 retry-exhausted; 16K rambling chars/decision |

## Interpretation

Rank-16 is the sweet spot for a 26-row corpus. Higher rank over-fits/loses tool discipline — dose-response is catastrophic collapse above rank-16, not smooth degradation. MLX-LM lacks gradient clipping; the LR × rank × small-corpus interaction drives the instability. Verified: 147 rows at rank-64 still diverges at LR peak.

The rank-16 improvement (66.7% → 70.0%) is modest but real — the first evidence that preserve_thoughts actually transfers to behavior once thought tokens reach the loss. It does not match iter-3-rules (90%) — next lever is larger corpus at rank-16 (N=100+ rollouts), not more rank.

## Reframes

Retroactively reframes the iter-4 [[experiments/iter4-null-preserve-thoughts]] result: with `max_seq_length=1024`, both stripped and preserved rows were identical at token 1024. The fix was data-reaching-loss, not LoRA capacity.

Note: the E1 writeup and the ceca203 commit message say the old truncation ceiling was `max_seq_length=2048`; [[sft-max-seq-length]] and the iter-4 pages say TRL's default 1024. This page follows the 1024 account — one of the two source docs is wrong.

## Related pages

[[burl]] · [[preserve-thoughts]] · [[decisions/sft-max-seq-length]] · [[iter3-rules-adapter]] · [[mlx-lm]] · [[lora-unsloth]] · [[experiments/iter4-null-preserve-thoughts]] · [[sources/ceca203]]
