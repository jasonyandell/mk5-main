---
title: "Stage 0 v7-v9: 14-Category Expansion"
kind: experiment
first_seen: b857299
last_updated: b857299
status: superseded
---

## Summary

Progressive curriculum expansion from 8 to 14 categories across Stage 0 versions v7, v8, and v9. All training on [[qwen3-1.7b]] via [[lora-unsloth]] on [[modal]] B200. Final v9 adapter achieves 83% overall comprehension across 14 categories.

([sources/b857299](../sources/b857299.md))

## Setup

- **Base model:** [[qwen3-1.7b]]
- **Training:** 3 epochs per version, [[modal]] B200, [[lora-unsloth]]
- **Eval:** [[decisions/flexible-grader]] (`grade_offline.GRADERS`), 14-category [[game-context-qa]]
- **v7:** added `conditional_beat` (first of 6 new categories toward 14 total)
- **v8:** added `beaters_in_unseen`, `partner_response`, `intervention_check`
- **v9:** added `visibility_audit`, `highest_unseen_in_suit`; full 3-epoch train on all 14

## Results (v9, 14 categories)

| Category | Accuracy |
|---|---|
| highest_unseen_in_suit (new) | 100% |
| is_trump, legal_moves, what_beats, rank_in_suit, suit_members, void_deduction, conditional_beat | 98-100% |
| where_is | 98% |
| count_status | 93% |
| intervention_check | 70% (dropped from 91% at v8) |
| partner_response | 48% (unchanged) |
| beaters_in_unseen | 46% (unchanged) |
| visibility_audit (new) | **0%** |
| **Overall** | **583/700 = 83%** |

Adapter: `jasonyandell/qwen3-1.7b-texas42-stage0-v9` ([[v9-adapter]]).

## Findings

**v7 — conditional_beat transfer:** the `conditional_beat` template propagates into rationalization context it was never explicitly trained on — real transfer across contexts.

**v8 — verbose-by-default shift:** adding 3 categories together shifted the response distribution to verbose-by-default. Concepts synthesize across categories even when not explicitly combined.

**v9 — rationalization plateau:** rationalization pass rate plateaus at ~68/100 across all v7/v8/v9 versions. Initially attributed to 1.7B capacity ceiling — later disproven by [[experiments/v10-maskfix-breakthrough]] (gradient allocation was the actual cause).

**visibility_audit at 0% — structural failure:** long-enumeration answers (list all visible and unseen dominoes) fail autoregressive truncation consistently. Single-fact supporting categories (`highest_unseen_in_suit` at 100%) are the reliable pattern. See [[single-fact-enumeration]].

**Composition doesn't auto-emerge:** adding atomic supporting categories (`highest_unseen_in_suit`) did not improve compound tasks (`beaters_in_unseen`, `partner_response`). Composition requires explicit training, not just prerequisite knowledge.

## Key new infrastructure

- `verify_rationalization.py` — [[rationalization-verifier]]: 6 engine-verified checks (domino validity, references-visible, hand claims, trump declaration, trump membership, action match). Filter that makes hallucinations irrelevant for training.
- `generate_decisions.py` — decision extractor with E[Q] data + structured state for the verifier.
- `scout_play_qwen.py` — runs rationalization experiments on [[qwen3-1.7b]].

## Related pages

[[v9-adapter]] · [[game-context-qa]] · [[rationalization-verifier]] · [[single-fact-enumeration]] · [[rules-adapter]] · [[qwen3-1.7b]] · [[lora-unsloth]] · [[modal]] · [[experiments/qwen-14b-capacity]] · [[sources/b857299]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 2 corrections applied in place and independently re-verified.

- The intervention_check 91% → 70% drop was later re-probed at stratified 50/category in the [[v10-maskfix-breakthrough]] eval: recovered to 88%, attributed to gradient allocation rather than noise (lem/OVERVIEW.md lines 834-856).
- Adapter `jasonyandell/qwen3-1.7b-texas42-stage0-v9` traces only to lem/OVERVIEW.md; not verified on HuggingFace itself.
- lem/OVERVIEW.md is internally inconsistent on the v8→v9 intervention_check comparison: line 673 says "a small sample (10 examples)" while the v9 overall (583/700) implies 50/category.
