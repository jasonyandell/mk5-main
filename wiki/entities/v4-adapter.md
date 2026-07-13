---
title: Stage 0 v4 Adapter (game-context)
kind: entity
first_seen: 2026-04-13
last_updated: 2026-04-13
status: superseded
superseded_by: v5-adapter
---

## What it is

The Stage 0 v4 Adapter is the LoRA fine-tune of [[gemma-4-e2b]] that pivots from
flashcard Q&A to game-context Q&A. It is trained on 31,830 examples drawn from
engine-verified game records, with compact prompts (~170 tokens vs ~2,400 in [[v3-adapter]]).
Per-epoch HF checkpoints are published. (commit messages @ 4729dad, 3c33e86)

## Training provenance

- **Base model**: [[gemma-4-e2b]] (`google/gemma-4-E2B-it`)
- **Corpus**: 31,830 train + 7,725 eval, generated from game records via 5 question types
  (see [[topics/game-context-qa]])
- **Method**: [[lora-unsloth]] LoRA fine-tune on [[modal]] B200, 3 epochs
- **Hyperparameters**: `batch_size=16` (was 4 + gradient accumulation), no gradient
  checkpointing — 1.4× faster than v3 training
- **Final metrics**: loss 0.06, 97.4% token accuracy

(commit message @ 3c33e86)

## Q&A types

| Type | What it tests |
|---|---|
| `where_is` | Track a domino across tricks (who captured it?) |
| `count_status` | Which count domino was captured by which team |
| `is_trump` | Trump identification with pip-level reasoning |
| `what_beats` | Domino ranking in game context |
| `legal_moves` | Full derivation showing suit-check per domino in hand |

(commit message @ 4729dad)

## Key behavioral change: thinking mode disabled

Inference disables the model's thinking channel. With the adapter loaded but thinking
enabled, the base model's hallucination ("This is a game of Bridge") dominated the
reasoning trace. With thinking disabled, the adapter's learned behavior surfaces cleanly.
(commit message @ 3c33e86)

## Eval results (flexible grader, 100 held-out examples)

See [[experiments/stage-0-v4-comprehension-eval]] and [[decisions/flexible-grader]].

| Question type | Accuracy |
|---|---|
| `is_trump` | 100% |
| `where_is` | 90% |
| `legal_moves` | 70% |
| `count_status` | 60% |
| `what_beats` | 15% |
| **Overall** | **67%** |

Two eval bugs (EOS token, left-pad slicing) masked real model knowledge in earlier
iterations; see [[sources/1d3e1b7]] for the fix details. The flexible grader
(`grade_offline.py`) extracts facts from free-form responses rather than requiring rigid
format matching — this was necessary to see the true 100% `is_trump` result that rigid
grading reported as 0%. (commit message @ 3c33e86)

## Significance

The model no longer hallucinates playing Bridge. `is_trump` at 100% means the 6-4-under-fives
error that persisted through [[kerry-adapter]] and [[v3-adapter]] is resolved in the
game-context format. `what_beats` at 15% is the remaining weak spot and the likely target
for future curriculum work.

## Superseded

Three days later (3465e29) the base model pivoted from [[gemma-4-e2b]] to [[qwen3-1.7b]]
— see [[base-model-pivot-qwen]]. [[v5-adapter]] retrains the same corpus on Qwen
and reaches 100% comprehension vs v4's 67% on the same eval. No Gemma v6 was ever built;
v4 is the last Gemma-base Stage 0 adapter.
