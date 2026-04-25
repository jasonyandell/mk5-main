---
title: Preserve Thoughts at SFT
kind: topic
first_seen: 20f4fa2
last_updated: edf86e9
status: re-opened
---

## Overview

Preserve-thoughts is an SFT training variant for [[burl]] that bypasses `apply_chat_template`'s `strip_thinking()` filter so assistant-turn `<|channel>thought...<channel|>` regions reach the loss computation. The default (`preserve_thoughts=False`) strips thought blocks before tokenization, meaning Gemma's winning reasoning chains never produce a gradient (20f4fa2).

## Reframe (edf86e9)

The iter-4-thoughts byte-identical A/B result (see "Result: null" below) was very likely a **truncation artifact**, not LoRA capacity saturation. TRL's `SFTConfig` defaults to `max_seq_length=1024` when the parameter is unset. The preserve-thoughts corpus has median 2,054 tokens and max 4,210 tokens per row — exactly the thought-bearing rows the recipe is meant to train on were being silently truncated to 1,024 tokens. Both paths had the same latent default: the Modal recipe (`star.py`) and the local MLX recipe (`star_mlx.py`) (edf86e9).

Fix: explicit `max_seq_length=4096` in `SFTConfig`. Local path was fixed first in 6fea6ab; Modal path fixed in edf86e9.

**Consequence:** no Burl adapter trained before iter-5 was trained on complete thought-to-tool-call traces. The thought tokens were present in the `formatting_func` output but were being clipped before they reached the loss (edf86e9).

This is a parallel to LEM's [[decisions/sft-completion-only-loss]] finding: TRL defaults silently trap gradient. The pattern — "we think we're training on X, but a default is preventing it" — has now appeared twice in this project (edf86e9).

**Follow-up:** iter-5 with `max_seq_length=4096` explicit should show whether `preserve_thoughts=True` actually changes the adapter when the thought data reaches the loss. [[experiments/iter4-null-preserve-thoughts]] updated accordingly.

## Implementation

A `formatting_func` path in `burl/train/star_iter0.py` is gated behind `preserve_thoughts: bool = False`. When enabled, assistant-turn content is appended verbatim — thought prose and all — instead of going through `apply_chat_template`. User-turn rendering stays canonical. `GEMMA4_TURN_TERMINATOR` is appended after assistant content to maintain format validity (20f4fa2).

## Empirical scale

On a real iter-3-rules corpus row: `apply_chat_template` path produces 1,714 tokens (thoughts stripped); `formatting_func` path produces 2,246 tokens (thoughts preserved) — +532 thought tokens per row. Across ~130 rows × 3 epochs: ~200K additional thought tokens that should reach the loss once `max_seq_length=4096` is set (20f4fa2, edf86e9).

## Result: null (suspect — see Reframe above)

A/B comparison against the stripped baseline (iter-4-thoughts vs iter-3-rules, identical recipe otherwise) produced **byte-identical adapter weights**. At the time this was interpreted as LoRA capacity saturation. The truncation discovery (edf86e9) makes this result uninformative about whether preserve-thoughts works — the thought tokens never reached the loss in either branch (dbadb5f, edf86e9).

## Links

[[burl]] [[iter3-rules-adapter]] [[experiments/iter4-null-preserve-thoughts]] [[decisions/sft-completion-only-loss]]
