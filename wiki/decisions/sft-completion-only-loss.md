---
title: SFT Completion-Only Loss (mask fix)
kind: decision
first_seen: 2026-04-17
last_updated: 2026-07-13
status: complete
---

## Decision

For SFT training, use dataset format `prompt`/`completion` (not `messages`). TRL 1.2+ auto-enables `completion_only_loss=True` with this format, so loss is computed only over completion tokens.

## Why

TRL's `SFTConfig` default leaves `assistant_only_loss=False`, computing loss over the full sequence (prompt + answer). For LEM v10:
- Answer: ~50 tokens
- Template/prompt tokens: ~400 tokens (already memorized by the model)
- The answer's gradient was diluted ~9× by tokens contributing no learning signal.

This is not answer leakage — `build_rationalize_sft.py` already scrubs the "correct play is X" string from rationalizations. It is pure gradient waste on memorized prompt tokens.

## Why not use `{% generation %}` markers

Qwen 3 and Gemma 4 chat templates do not include `{% generation %}` markers, so `assistant_only_loss=True` via the `messages` format does not work reliably. The `prompt`/`completion` format is template-agnostic and works with both model families.

## Validation

`scratch/verify_sft_mask{,_fix}.py` confirms the loss mask is correct after the fix.

## Impact

1.7B v10-maskfix hits 86% comprehension overall — matching 14B v9 at ~1/3 the training compute on B200. See [[experiments/v10-maskfix-breakthrough]].

Three metrics that had been stuck (`intervention_check`, `partner_response`, `beaters_in_unseen`) moved +10–18pp. One metric (`transition bot-match`) did not — confirming its ceiling is capacity or STaR-iteration, not gradient allocation.

## Generalizable principle

When SFT stalls and the answer is a small fraction of the total sequence, inspect the loss mask. Default TRL behavior wastes gradient on memorized prompt tokens. The fix is to switch to `prompt`/`completion` format rather than trying to insert `{% generation %}` markers into chat templates.

## Scope at this frontier

Applied to the 1.7B trainer only. 14B and 5 other trainers still have the bug. (No
follow-up run ever applied the fix to 14B — LEM ended the next ingest and no
"t42-aoga path A" run exists anywhere in the wiki or git history.)

## Bead

t42-0ynt closed.

## Related pages

[[v10-adapter]] · [[experiments/v10-maskfix-breakthrough]] · [[lora-unsloth]] · [[sources/be7efc4]] · [[qwen3-1.7b]] · [[qwen3-14b]]
