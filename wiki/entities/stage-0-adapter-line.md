---
title: Stage 0 Adapter Line (LEM)
kind: entity
first_seen: 2026-07-13
last_updated: 2026-07-13
status: complete
---

## What it was

The succession of Stage 0 LoRA adapters [[lem]] trained to give a small base model
Texas 42 comprehension before Stage 1 [[star]]. Seven shipped adapters across eight days
(2026-04-10 → 2026-04-17), spanning two base models and two curriculum formats. Each
adapter below is a receipt page with full provenance and eval detail.

## The chain

[[stage-0-adapter]] (v1) → [[kerry-adapter]] (v2) → [[v3-adapter]] → [[v4-adapter]] →
[[v5-adapter]] → [[v9-adapter]] → [[v10-adapter]] (two forms: v10 and v10-maskfix).

Numbering note: no Gemma v6 was ever built; v7 and v8 were Qwen category-expansion
intermediates with no pages of their own — [[v9-adapter]] absorbs them (its category
evolution table records what each added).

| Adapter | Base | Training data | Headline numbers | What it fixed / broke |
|---|---|---|---|---|
| [[stage-0-adapter]] (v1) | [[gemma-4-e2b]] | 3,500 flashcard Q&A, 7 categories | ~37% avg STaR pass, 42% peak, 33% illegal | Fixed hand tracking; trump membership broken (4-4, 6-4 called trump under fives) |
| [[kerry-adapter]] (v2) | [[gemma-4-e2b]] | 15,000 [[kerry-curriculum]] | ~43% avg, 46% peak, ~12% illegal | Fixed trump non-membership; cut illegal rate 33%→~12%; 6-4-under-fives persisted |
| [[v3-adapter]] | [[gemma-4-e2b]] | 20,000 (Kerry 15k + 5k [[trump-drilling]]) | ~44% avg, 48% peak (`star-iter2`), ~13% illegal | Last flashcard adapter; trump drilling did not kill the 6-4 error |
| [[v4-adapter]] | [[gemma-4-e2b]] | 31,830 [[game-context-qa]] examples | 67% comprehension; `is_trump` 100% | Game-context pivot resolved the 6-4 error and the Bridge hallucination; `what_beats` 15% |
| [[v5-adapter]] | [[qwen3-1.7b]] | same corpus as v4 | 100% comprehension | Base-model pivot ([[base-model-pivot-qwen]]): 100% vs v4's 67% on the same eval |
| [[v9-adapter]] | [[qwen3-1.7b]] | 14-category [[game-context-qa]] corpus | 83% comprehension; rationalization ~68/100 ceiling | Added [[rationalization-verifier]]; exposed `visibility_audit` 0% ([[single-fact-enumeration]]) |
| [[v10-adapter]] | [[qwen3-1.7b]] | v9 corpus + 331 verified rationalizations upweighted 10× (31,307 examples) | v10: 83% comprehension, 55/100 bot-match; v10-maskfix: **86%** comprehension, 55/100 bot-match | Completion-only loss ([[sft-completion-only-loss]]) recovered gradient-starved metrics; bot-match ceiling unmoved |

Cross-adapter STaR numbers (v1/Kerry/v3 rows) are from the progression table at
`8c1bb14`; each curriculum round raised the floor — the ingest-10 plateau at ~40% was a
Stage-0-quality ceiling, not a K1-grading ceiling. (commit message @ 8c1bb14)

## Supersession story

Four distinct levers moved the line, in order:

1. **Curriculum scale** (v1 → Kerry): 3.5k ad-hoc Q&A → 15k examples structured after
   Kerry Newberry's Learner's Guide.
2. **Targeted drilling** (Kerry → v3): +5k [[trump-drilling]] for the one stubborn
   trump-membership error. Marginal gain; the error survived.
3. **Format pivot** (v3 → v4): flashcards → [[game-context-qa]] drawn from
   engine-verified game records, ~170-token prompts vs ~2,400. This — not drilling —
   killed the 6-4-under-fives error.
4. **Base-model pivot** (v4 → v5): [[gemma-4-e2b]] → [[qwen3-1.7b]], 67% → 100% on the
   same eval — decisive for [[base-model-pivot-qwen]]. v9 then expanded categories
   5 → 14, and v10 added joint rationalization training.

The line closed with [[v10-maskfix-breakthrough]]: switching to completion-only SFT loss
took 1.7B comprehension from 83% to 86% — matching [[qwen3-14b]] v9 at roughly 1/3 the
training compute — and proved the stuck comprehension metrics had been gradient-starved,
not capacity-limited.

## How the line ended

v10-maskfix is the terminal adapter. The maskfix result left the 55/100 transition
bot-match ceiling untouched, and neither proposed next lever (14B capacity, STaR on v10)
was ever run — [[lem]] went dormant at `be7efc4` (2026-04-17) and the project pivoted
mechanisms to [[burl]] (see [[lem-to-burl-handoff]], and [[burl-adapter-line]] for that
line's adapters). As of jud v1, the project's play mechanism consumes no LoRA adapter at
all ([[champion]] runs "zero adapter"). The line ended because its consumer was
abandoned, not because a v11 lost a bake-off.

## Receipts

[[stage-0-adapter]] · [[kerry-adapter]] · [[v3-adapter]] · [[v4-adapter]] ·
[[v5-adapter]] · [[v9-adapter]] · [[v10-adapter]] · [[v10-maskfix-breakthrough]]
