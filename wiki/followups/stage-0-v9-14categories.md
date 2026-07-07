# stage-0-v9-14categories — audit 2026-07-07

## Corrections

- Page's v9 results table mixed in numbers from the earlier pre-curriculum baseline eval (where_is ~90%, legal_moves ~70%, count_status ~60%, what_beats 15%); the actual v9 eval is where_is 98%, count_status 93%, and is_trump/legal_moves/what_beats/rank_in_suit/suit_members/void_deduction/conditional_beat all at 98-100% (evidence: lem/OVERVIEW.md lines 639-649 vs the baseline table at ~line 373). Overall 83% (583/700) was correct.
- Clarified v7 line: v7 added `conditional_beat` alone (first of the 6 new categories), not 6 categories at once (evidence: commit b857299 message).

## Verified

- Commit b857299 exists and matches the narrative; scripts exist at lem/gemma_star/verify_rationalization.py, lem/gemma_star/scout_play_qwen.py, lem/rules/generate_decisions.py.
- Findings (v7 transfer, v8 verbose shift, visibility_audit 0% truncation failure, composition non-emergence, rationalization plateau 64-68/100) match lem/OVERVIEW.md.
- HF adapter `jasonyandell/qwen3-1.7b-texas42-stage0-v9` named in OVERVIEW.md; not verified on HuggingFace itself.

## Follow-ups

- (none — the one candidate, re-probing intervention_check's 91% → 70% drop with a larger held-out set, was already done: the v10-maskfix eval used stratified 50/category and intervention_check recovered to 88%, attributed to gradient allocation rather than noise; lem/OVERVIEW.md lines 834-856.)

## Review (second pass, 2026-07-07)

- Verified — both page corrections stand. The v9 table matches lem/OVERVIEW.md lines 639-649 exactly, and the original page's where_is ~90% / legal_moves ~70% / count_status ~60% / what_beats 15% rows do come from the earlier Gemma stage0-v4 baseline table (lem/OVERVIEW.md ~lines 373-381, overall 67%). The v7 = conditional_beat-only claim matches commit b857299's message ("conditional_beat (v7)"; v8 and v9 categories listed separately). Script paths, the 64-68/100 plateau range (OVERVIEW.md lines 686-690), and the adapter name (OVERVIEW.md line 888) also check out.
- Amended this followup only: dropped the intervention_check re-grading suggestion as already done in the repo (v10-maskfix re-eval at 50/category, intervention_check 88%; lem/OVERVIEW.md lines 834-856). The page needed no changes.
- Minor unamended note: OVERVIEW.md line 673 says the v8→v9 intervention_check comparison rested on "a small sample (10 examples)" while the v9 overall (583/700) implies 50/category — an internal inconsistency in OVERVIEW.md itself, not an error introduced by the page or the auditor.
