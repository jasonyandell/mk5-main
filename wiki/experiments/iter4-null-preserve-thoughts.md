---
title: "iter-4 preserve_thoughts: Byte-Identical A/B (Null Result)"
kind: experiment
first_seen: 20f4fa2
last_updated: edf86e9
status: retired
---

## Summary

Same iter-3-rules corpus, same hyperparameters; only variable is `preserve_thoughts=True` — a `formatting_func` path that bypasses `apply_chat_template`'s `strip_thinking()` filter so thought tokens reach the loss. Result: byte-identical output to baseline. Null A/B.

([burl/train/star_iter4_thoughts.py @ 20f4fa2](../sources/20f4fa2.md))

## Setup

- **Variable:** `preserve_thoughts=True` vs baseline `False`
- **Token delta:** +532 thought tokens/row; across ~130 rows × 3 epochs = ~200K additional tokens reaching the gradient that previously produced none
- **All else:** identical corpus, rank 16, 3 epochs, lr 1e-4, B200

## Result

Byte-identical output on the held-out eval set. The thought-gradient produced no observable behavioral change.

## Interpretation

LoRA rank 16 may be capacity-saturated — the adapter had no room to learn from the extra thought gradient on top of the iter-3-rules knowledge already encoded. The thought tokens are reaching the loss, but the rank-16 adapter cannot absorb the additional signal.

**Follow-up hypothesis:** higher LoRA rank (32 or 64) may unblock thought-gradient learning. This is experiment E1 in `burl/ITER4_PLAN.md` — cheap to run locally on M5 Max.

## Reframe (edf86e9)

The byte-identical A/B was very likely a truncation artifact, not a LoRA capacity ceiling.

TRL's `SFTConfig` defaults to `max_seq_length=1024`. Burl's `preserve_thoughts` corpus has median 2054 tokens and max 4210 tokens per row. Both the stripped and the preserved training rows were silently chopped at token 1024 — making them nearly identical at training time. The adapter had nothing to distinguish between the two conditions.

No Burl adapter before iter-5 was trained on complete thought-to-tool-call traces. See [[decisions/sft-max-seq-length]].

**Follow-up:** iter-5 re-runs the A/B with `max_seq_length=4096`. If non-null, this experiment upgrades from "null/retired" to "artifact resolved."

## Related pages

[[preserve-thoughts]] · [[iter3-rules-adapter]] · [[lora-unsloth]] · [[burl]] · [[decisions/sft-max-seq-length]] · [[sources/20f4fa2]] · [[sources/dbadb5f]] · [[sources/edf86e9]]
