---
title: Third Gemma Contact (Kerry Adapter)
kind: experiment
first_seen: 43009a4
last_updated: 43009a4
status: active
---

## Summary

Third inference pass against [[gemma-4-e2b]], using the same seed/trump/trick-6 prompt used for [[experiments/first-gemma-contact]] and [[experiments/second-gemma-contact]], now with [[kerry-adapter]] loaded. Tests whether the Kerry-structured Stage 0 v2 curriculum narrows the residual rules failure surface.

([lem/gemma_star/train_stage0.py @ 43009a4](../sources/43009a4.md))

## Setup

- **Prompt:** identical to prior contacts — seed 42, fives trump, truncated at trick 6 narrator turn, rules primer prepended
- **Adapter:** [[kerry-adapter]] (`jasonyandell/gemma-4-e2b-texas42-stage0-kerry`), trained on 15k-example [[kerry-curriculum]] corpus, 150 steps on B200
- **Adapter loading:** `PeftModel.from_pretrained`, merged and unloaded before inference

## Results

| Dimension | 1st contact (base) | 2nd contact (stage-0) | 3rd contact (kerry) |
|---|---|---|---|
| Hand tracking | wrong (played 5-5, already gone) | fixed | maintained |
| Led suit | correct | correct | correct |
| Void recognition | not tested explicitly | not tested | correct ("you hold no fours") |
| Trump non-membership | confused (6-2, 6-1 called not-trump but reasoning shaky) | confused | IMPROVED ("no fives, no trump" for 6-2, 6-1) |
| Trump membership | wrong (4-4 and 6-4 called trumps) | wrong (4-4 and 6-4) | only 6-4 still wrong |
| Strategic reasoning | shallow | medium | dramatically deeper — evaluates both legal options explicitly |
| Final answer | illegal (5-5) | legal, correct | legal, correct |

([lem/OVERVIEW.md and commit message @ 43009a4](../sources/43009a4.md))

## Significance

The Kerry curriculum successfully narrows the rules failure surface compared to v1. Specifically:

- Trump **non-membership** is now correct: the model correctly identifies 6-2 and 6-1 as not trump under fives.
- Strategic reasoning is dramatically deeper: the model evaluates both legal options before committing.
- The single persisting error — calling 6-4 trump under fives — is notable. The 6-4 is a count domino (10 points) that contains a 4-pip; under fives-trump, it is not trump. The model may be confusing count-domino salience or the presence of the 4-pip (4s are trump when fours are trump) with trump membership. This is a specific, targeted failure rather than broad rule confusion.

## Contact series progression

Three contacts at the same prompt provide a controlled comparison of what each adapter version teaches. The 6-4 error is the single remaining rules failure; all other dimensions are now correct or substantially improved.

## Related pages

[[lem]] · [[kerry-adapter]] · [[kerry-curriculum]] · [[rules-adapter]] · [[gemma-4-e2b]] · [[experiments/first-gemma-contact]] · [[experiments/second-gemma-contact]] · [[sources/43009a4]] · [[sources/f8cdbe7]]
