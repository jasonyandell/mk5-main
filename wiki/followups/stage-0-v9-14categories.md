# stage-0-v9-14categories — audit 2026-07-07

## Corrections

- Page's v9 results table mixed in numbers from the earlier pre-curriculum baseline eval (where_is ~90%, legal_moves ~70%, count_status ~60%, what_beats 15%); the actual v9 eval is where_is 98%, count_status 93%, and is_trump/legal_moves/what_beats/rank_in_suit/suit_members/void_deduction/conditional_beat all at 98-100% (evidence: lem/OVERVIEW.md lines 639-649 vs the baseline table at ~line 373). Overall 83% (583/700) was correct.
- Clarified v7 line: v7 added `conditional_beat` alone (first of the 6 new categories), not 6 categories at once (evidence: commit b857299 message).

## Verified

- Commit b857299 exists and matches the narrative; scripts exist at lem/gemma_star/verify_rationalization.py, lem/gemma_star/scout_play_qwen.py, lem/rules/generate_decisions.py.
- Findings (v7 transfer, v8 verbose shift, visibility_audit 0% truncation failure, composition non-emergence, rationalization plateau 64-68/100) match lem/OVERVIEW.md.
- HF adapter `jasonyandell/qwen3-1.7b-texas42-stage0-v9` named in OVERVIEW.md; not verified on HuggingFace itself.

## Follow-ups

- intervention_check's 91% → 70% drop was flagged in OVERVIEW.md as noisy (n=10 per category); a cheap probe would be re-grading that category with a larger held-out set.
