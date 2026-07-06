---
title: "Source digest: be7efc4 — SFT completion-only loss — 1.7B-maskfix hits 86% comprehension"
kind: source
first_seen: be7efc4
last_updated: be7efc4
status: active
---

## Commit

- **SHA:** be7efc4917466b3f33c4c0401c912b14b14bad4f
- **Date:** 2026-04-17
- **Author:** Jason Yandell

> fix(lem): SFT completion-only loss — 1.7B-maskfix hits 86% comprehension (= 14B v9)
>
> TRL's `SFTConfig` defaults leave `assistant_only_loss=False`, so every one of
> our 6 SFT trainers was computing loss over the full sequence (prompt +
> answer). For v10 specifically, no answer leakage (build_rationalize_sft.py
> already scrubs the leaky "correct play is X" string) — but the ~50-token
> answer's gradient was diluted ~9× by ~400 template/prompt tokens the model
> had trivially memorized. Bead t42-0ynt.
>
> Fix (template-agnostic, works with Qwen 3 and Gemma 4 chat templates that
> lack {% generation %} markers): switch dataset format from `messages` →
> `prompt`/`completion`. TRL 1.2+ auto-enables completion_only_loss=True.
> Verified with scratch/verify_sft_mask{,_fix}.py.
>
> Validation on 1.7B v10 — same 31,307-example corpus, same hparams:
>
>   Comprehension:          83% → 86% overall (= 14B v9 at 1/3 cost)
>     intervention_check:   70% → 88% (+18, now beats 14B)
>     partner_response:     48% → 58% (+10)
>     beaters_in_unseen:    46% → 56% (+10)
>   Transition bot-match:   55/100 → 55/100 (literally unchanged)
>
> 3 of the 4 stuck metrics moved decisively; transition bot-match didn't,
> confirming it's not a gradient-allocation problem (capacity or STaR is the
> next lever, not more SFT signal).
>
> Adapter: jasonyandell/qwen3-1.7b-texas42-stage0-v10-maskfix.

## Files modified

| Path | Change |
|---|---|
| `lem/gemma_star/eval_comprehension_qwen.py` | Inline `grade_response()` removed; dispatch through `grade_offline.GRADERS` (single source of truth) |
| `lem/gemma_star/eval_comprehension_qwen_14b.py` | Same grader unification |
| `lem/gemma_star/train_comprehension_qwen.py` | Dataset format switched `messages` → `prompt`/`completion`; B200 pinned |
| `lem/OVERVIEW.md` | +73 lines: maskfix results, grader unification explanation |

## Key results

See [[experiments/v10-maskfix-breakthrough]] for the full comparison table. Headline: 1.7B v10-maskfix hits 86% comprehension overall = 14B v9 at ~1/3 B200 compute.

## Eval grader unification

Prior inline `grade_response()` in both eval scripts covered only 5 of 14 categories, scoring the remaining 9 as 0%. This caused [[modal]] eval to read 35% while `grade_offline.py` read 86% on identical raw responses. Both scripts now dispatch through `grade_offline.GRADERS`. Consistent with [[decisions/flexible-grader]] principle: single source of truth for grading logic.

## Bead

t42-0ynt closed (B200 gradient waste / SFT mask bug).

## Scope note

Loss mask fix applied to 1.7B trainer only. 14B and 5 other trainers still have `assistant_only_loss=False` at this frontier. 14B is next per t42-aoga path A.

## Related pages

[[decisions/sft-completion-only-loss]] · [[experiments/v10-maskfix-breakthrough]] · [[v10-adapter]] · [[decisions/flexible-grader]] · [[qwen3-1.7b]] · [[qwen3-14b]] · [[sources/0c7392f]]
