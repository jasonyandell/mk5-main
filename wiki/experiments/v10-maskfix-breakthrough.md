---
title: "v10-maskfix: 86% on 1.7B (= 14B v9 at 1/3 cost)"
kind: experiment
first_seen: be7efc4
last_updated: be7efc4
status: active
---

## Summary

Same 31,307-example v10 corpus, same hyperparameters — but dataset format switched from `messages` to `prompt`/`completion`, which causes TRL 1.2+ to auto-enable `completion_only_loss=True`. Result: [[qwen3-1.7b]] reaches 86% comprehension overall, matching [[qwen3-14b]] v9 at roughly 1/3 the training compute.

([sources/be7efc4](../sources/be7efc4.md))

## The fix

See [[decisions/sft-completion-only-loss]]. TRL's `SFTConfig` default computes loss over the full sequence (prompt + answer). For v10, the ~50-token answer's gradient was being diluted ~9× by ~400 template/prompt tokens the model had already memorized. Switching to `prompt`/`completion` format re-focuses gradient entirely on the completion tokens.

## Results: v10 → v10-maskfix

| Dimension | v10 (before) | v10-maskfix (after) |
|---|---|---|
| Comprehension overall | 83% | **86%** |
| intervention_check | 70% | **88%** (+18, now beats 14B) |
| partner_response | 48% | **58%** (+10) |
| beaters_in_unseen | 46% | **56%** (+10) |
| Transition bot-match | 55/100 | **55/100** (unchanged) |

Adapter: `jasonyandell/qwen3-1.7b-texas42-stage0-v10-maskfix` ([[v10-adapter]]). The best LEM adapter at end-of-replay.

## Interpretation

3 of 4 previously stuck comprehension metrics moved decisively with the gradient reallocation. The metrics that moved (`intervention_check`, `partner_response`, `beaters_in_unseen`) had been gradient-starved, not capacity-limited.

**Transition bot-match unchanged at 55/100:** confirms the bot-match ceiling is not a gradient-allocation problem. It is a capacity or STaR-iteration problem — the next lever would have been [[star]] iterations on top of the maskfix adapter, not more SFT signal.

**Cost comparison:** 1.7B v10-maskfix reaches the same 86% comprehension as 14B v9 at roughly 1/3 the B200 training compute.

## Also in this commit

- Eval graders unified to `grade_offline.GRADERS` (single source of truth). The prior inline `grade_response()` in `eval_comprehension_qwen.py` covered only 5 of 14 categories, scoring the rest 0%. This is why [[modal]] eval read 35% while `grade_offline.py` read 86% on the same raw responses. Consistent with [[decisions/flexible-grader]].
- `train_comprehension_qwen.py` pinned to B200 (was H100).
- Scope: 1.7B trainer only. 14B and other trainers still have the loss-mask bug at this frontier.

## Related pages

[[v10-adapter]] · [[qwen3-1.7b]] · [[qwen3-14b]] · [[decisions/sft-completion-only-loss]] · [[decisions/flexible-grader]] · [[experiments/qwen-14b-capacity]] · [[star]] · [[modal]] · [[sources/be7efc4]]
