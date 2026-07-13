---
title: Qwen 3 14B Capacity Experiment
kind: experiment
first_seen: 2026-04-17
last_updated: 2026-04-17
status: active
---

## Summary

[[qwen3-14b]] base trained on the same v9 corpus (14 categories) with the same hyperparameters as the 1.7B run. Tests whether the ~68/100 rationalization ceiling and weaker list-enumeration scores in [[experiments/stage-0-v9-14categories]] are capacity-limited.

([sources/0c7392f](../sources/0c7392f.md))

## Setup

- **Base model:** [[qwen3-14b]]
- **Data:** same 14-category v9 comprehension corpus used for [[v9-adapter]]
- **Training:** same hyperparameters as 1.7B; [[lora-unsloth]] on [[modal]] B200
- **Adapter:** `jasonyandell/qwen3-14b-texas42-stage0-v9`

## Results

| Dimension | 1.7B v9 | 14B v9 |
|---|---|---|
| Comprehension overall | 83% | **86%** (+3pp) |
| partner_response | 48% | **75%** (+27pp) |
| beaters_in_unseen | 46% | **61%** (+15pp) |
| Rationalization clean | ~68/100 | **97/100** (+43pp) |
| Final loss | 0.205 | **0.125** |
| visibility_audit | 0% | **0%** (unchanged) |

## Interpretation

Capacity helps substantially for list-enumeration tasks (`partner_response`, `beaters_in_unseen`) and for multi-factor rationalization reasoning — the 14B model shows "lucid multi-factor reasoning" per the commit. The 43pp rationalization jump is the headline finding.

**visibility_audit at 0% on both:** confirms long-enumeration failure is structural (answer format), not a capacity problem. Both models fail equally. See [[single-fact-enumeration]].

**The capacity gap narrows later:** [[experiments/v10-maskfix-breakthrough]] shows that 1.7B with the gradient mask fix reaches 86% comprehension — matching 14B v9 at roughly 1/3 the training compute. This retroactively suggests the earlier 1.7B/14B comprehension gap was partly gradient allocation, not purely capacity.

**Open question at this frontier:** does 14B's 97/100 rationalization survive the mask fix? Not tested — but if 1.7B can close the comprehension gap this way, the rationalization gap may also narrow.

## Cross-commit finding

Parallel 1.7B v10 experiment (same commit): scout 500 decisions on v9 → 331 clean rationalizations (66% pass) → joint training → 55/100 bot-match (crosses 50%), 96/100 legal, 83% comprehension preserved. See [[sources/0c7392f]] for v10 1.7B details.

## Related pages

[[qwen3-14b]] · [[qwen3-1.7b]] · [[v9-adapter]] · [[v10-adapter]] · [[single-fact-enumeration]] · [[rationalization-verifier]] · [[r1-rationalization]] · [[experiments/stage-0-v9-14categories]] · [[experiments/v10-maskfix-breakthrough]] · [[sources/0c7392f]]
