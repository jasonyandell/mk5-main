---
title: "SFT max_seq_length: Set Explicitly, Don't Trust Defaults"
kind: decision
first_seen: 2026-04-19
last_updated: 2026-07-13
status: complete
---

## Decision

Always pass `max_seq_length=4096` (or larger, matching corpus max) to `SFTConfig`. Never rely on TRL's default of 1024.

## Why

TRL's `SFTConfig` defaults to `max_seq_length=1024` when the parameter is unset. Burl's `preserve_thoughts` corpus has median 2054 tokens and max 4210 tokens per row — the thought-bearing regions the recipe is meant to train on were silently truncated at token 1024 in every Burl adapter trained before iter-5 (both Modal `star.py` and local `star_mlx.py` paths).

This is the very likely root cause of the [[iter4-null-preserve-thoughts]] "byte-identical" result. Both the stripped and preserved training rows were chopped at token 1024, making them nearly identical at training time. The adapter had nothing to distinguish — not LoRA capacity saturation as initially diagnosed.

## Parallel to LEM

[[sft-completion-only-loss]] documented TRL's first trap: `SFTConfig` defaults leave `assistant_only_loss=False`, wasting gradient on memorized prompt tokens. This decision documents the second known trap: `max_seq_length=1024` silently truncates rows longer than 1024 tokens.

TRL's `SFTConfig` has at least two traps whose defaults waste or destroy gradient. Both require explicit overrides.

## Generalizable principle

For any SFT recipe: log the actual token count of training rows AND assert the max against the configured `max_seq_length`. Silent truncation is the quietest failure mode in supervised fine-tuning — the trainer runs successfully, loss decreases, but the model never sees the tokens the recipe was designed to train on.

## Follow-up — resolved

[[iter5-e1-rank-sweep]] re-ran the [[preserve-thoughts]] A/B with
`max_seq_length=4096` and confirmed the result is non-null: rank-16 bot-match improved
66.7% → 70.0% over base Gemma. This retroactively vindicates the thought-gradient
hypothesis and upgrades [[iter4-null-preserve-thoughts]] from "null result" to
"truncation artifact."

## Related pages

[[preserve-thoughts]] · [[iter4-null-preserve-thoughts]] · [[sft-completion-only-loss]] · [[lora-unsloth]] · [[burl]] · [[edf86e9]]
