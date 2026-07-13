---
title: Stage 0 v3 Adapter (Kerry + trump drill)
kind: experiment
first_seen: 2026-04-11
last_updated: 2026-07-13
status: superseded
superseded_by: v4-adapter
---

Receipt in the [[stage-0-adapter-line]].

## What it is

The Stage 0 v3 Adapter is the LoRA fine-tune of [[gemma-4-e2b]] trained on 20,000 examples
(15k [[kerry-curriculum]] + 5k [[trump-drilling]]), 200 steps on [[modal]] B200. HF repo
naming was not recorded in the commit; see commit 601f622 for the target repo. It extends
[[kerry-adapter]] by adding targeted trump-membership Q&A to address the one remaining
stubborn error (6-4 called trump under fives). (commit messages @ 601f622, 8c1bb14)

## Training provenance

- **Base model**: [[gemma-4-e2b]] (`google/gemma-4-E2B-it`)
- **Training data**: 20,000 examples
  - 15,000 [[kerry-curriculum]] (sections A/B/C/D, section C at 40%)
  - 5,000 [[trump-drilling]]: targeted trump membership Q&A across five question types
    (`is_trump`, `list_trumps`, `which_trumps`, `trump_or_follow`, `count_trump`)
- **Method**: [[lora-unsloth]] LoRA fine-tune on [[modal]] B200, 200 steps

(commit message @ 601f622)

## STaR results (5 iterations)

| Iter | Pass rate |
|---|---|
| 0 | 44% |
| 1 | 42% |
| 2 | **48%** (new high water mark) |
| 3 | 47% |
| 4 | 38% |

Average ~44%. Illegal rate ~13% across iterations. Best adapter: `star-iter2` at 48%.
See [[stage-0-progression-star]]. (commit message @ 8c1bb14)

v3 was the last flashcard-format Stage 0 adapter — two days later [[v4-adapter]] pivoted
to [[game-context-qa]] and moved past this peak.

## Stage 0 progression

The cross-adapter progression table (v1 → Kerry → v3: avg pass, peak, illegal rate)
lives on [[stage-0-adapter-line]]. (commit message @ 8c1bb14)

v3 is superseded by [[v4-adapter]] (game-context Q&A, 4729dad) — flashcard curricula (v1,
Kerry, v3) were discarded in favor of game-grounded prompts.
