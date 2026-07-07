Reviewed against code on 2026-07-07 — no issues found.

All headline numbers (83%→86%, +18 intervention_check, +10 partner_response, +10 beaters_in_unseen, 55/100 bot-match unchanged), the mechanism (messages → prompt/completion triggering TRL 1.2+ completion_only_loss), the grader unification, and the B200 pin all match commit be7efc4 and the diffs to lem/gemma_star/{eval_comprehension_qwen,eval_comprehension_qwen_14b,train_comprehension_qwen}.py.

- Eval numbers live only in the commit message; no results JSON is in-repo, and the adapter (jasonyandell/qwen3-1.7b-texas42-stage0-v10-maskfix) is on HuggingFace — unverifiable locally.
- Cheap next probe (already flagged in commit): apply the same maskfix to the 14B and remaining 5 trainers and re-check whether bot-match is truly capacity-limited.
