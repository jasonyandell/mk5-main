---
title: Game-Context Q&A (Stage 0 v4)
kind: topic
first_seen: 2026-04-13
last_updated: 2026-04-17
status: superseded
---

## Overview

Game-context Q&A is the training approach for [[rules-adapter]] Stage 0 v4. Instead of context-free flashcard Q&A ("Is 6-4 trump when fives are trump?"), the model is given a compact game-state prompt (~170 tokens) and answers questions about *that specific game*. The corpus is engine-generated from actual game records; all answers are verified ground truth (4729dad).

## Philosophical shift

v1–v3 all used variants of the flashcard pattern: isolated question, isolated answer, no surrounding game context. The model learned to pattern-match on the Q&A surface form rather than ground knowledge in a game state. v4's hypothesis: LLMs ground knowledge in narratives with state, not in stripped-down fact tables. Asking "Is the 6-4 trump in *this game* where fives are trump?" is a different task from a flashcard, and closer to the narration-context reasoning [[star]] actually requires (4729dad).

## Five question types

| Type | What it tests |
|---|---|
| `where_is` | "Which trick did the 6-5 appear in?" — cross-trick domino tracking |
| `count_status` | "Who took the 5-5?" — count-domino bookkeeping across tricks |
| `is_trump` | "Is the 6-4 trump?" with required reasoning — rule application in context |
| `what_beats` | "Which of [6-5, 5-4, 4-4] would win this trick?" — ranking in a specific game state |
| `legal_moves` | "What can you legally play?" — suit-check logic against a concrete hand |

(4729dad)

## Corpus

31,830 training examples + 7,725 eval examples, generated from engine-verified game records. ~170 tokens per prompt (vs ~2,400 tokens in the v3 flashcard format — 14× reduction). No hallucination risk; engine is ground truth (4729dad).

## Training

31k examples × 3 epochs on B200, batch size 16 (no gradient checkpointing), per-epoch HF checkpoints. Final: loss 0.06, 97.4% token accuracy. See [[v4-adapter]] (3c33e86).

## Eval bugs discovered and fixed

Two bugs in the initial eval pipeline (`eval_comprehension.py`) were corrupting all previous results (1d3e1b7):

1. **EOS token**: `generate()` stopped only on `<eos>` (token 1) but Gemma 4 uses `<turn|>` (token 106) to end model turns. Missing this caused responses to run on into garbage.
2. **Left-pad slicing**: used `attention_mask.sum()` (real token count) instead of `input_ids.shape[1]` (padded input length) to separate generated tokens from input. With left padding, this included trailing input tokens in the "response", causing prompt-echo artifacts.

Both fixes required before responses were clean enough to grade. See [[1d3e1b7]].

## Flexible grader

A rigid pattern-matcher failed on free-form model responses (e.g., `is_trump 0%` even when the model answered correctly in prose). The flexible offline grader (`grade_offline.py`) extracts facts from free-form responses: finds legal moves in numbered lists, yes/no anywhere in response, trick numbers for domino tracking, domino overlap for `what_beats`. See [[flexible-grader]] (3c33e86).

## Eval results

Held-out seeds, 100 examples, flexible grader (3c33e86):

| Question type | Score |
|---|---|
| `is_trump` | **100%** |
| `where_is` | 90% |
| `legal_moves` | 70% |
| `count_status` | 60% |
| `what_beats` | 15% |
| **Overall** | **67%** |

"The model knows Texas 42. It doesn't say 'Bridge' anymore." (3c33e86) `what_beats` at 15% is the remaining weak spot — domino ranking in context is the hardest reasoning task in the set.

## 14-category expansion (v7-v9)

Six new question types were added across three curriculum rounds (b857299):

| Category | Version | What it tests |
|---|---|---|
| `conditional_beat` | v7 | "Can X beat Y when Z is led?" — multi-step reasoning with verifiable sub-steps |
| `beaters_in_unseen` | v8 | Reason over the unseen domino pool |
| `partner_response` | v8 | Reason over the unseen pool with partner's future plays |
| `intervention_check` | v8 | Play-order matters — partner can only beat opponents who play BEFORE partner |
| `visibility_audit` | v9 | Enumerate visible + unseen dominoes (FAILURE CASE — see [[single-fact-enumeration]]) |
| `highest_unseen_in_suit` | v9 | "What is the highest unseen domino in this suit?" — single-fact supporting question |

**Findings:**

- `conditional_beat` (v7) templates propagated into rationalization contexts the model was never trained on — real transfer (b857299).
- Adding 3 categories together in v8 shifted response distribution to verbose-by-default; concepts synthesize across categories (b857299).
- v9 full 3-epoch training: 83% overall on 14 categories (b857299).
- `visibility_audit` 0% on both 1.7B and 14B confirms a structural long-enumeration problem, not capacity. See [[single-fact-enumeration]] (b857299, 0c7392f).
- Compound tasks (`beaters_in_unseen`, `partner_response`) did not improve from atomic supporting categories — composition does not auto-emerge (b857299).

## Corpus portability

The game-context Q&A corpus is base-model-agnostic. The same 31,830 train / 7,725 eval examples were used to train both Gemma 4 E2B (v4, 67% overall) and Qwen 3 1.7B (v5, 100% overall). The corpus quality is validated — the 33-point gap is explained by the base model, not by the training data. See [[rules-adapter]] Stage 0 v5 and [[base-model-pivot-qwen]] (3465e29).

## Terminal LEM Stage-0 artifact

Game-context Q&A was the last Stage 0 curriculum format LEM used (v4 through v10-maskfix
all built on it) — but it is terminal to LEM, not carried forward. [[burl]] does not train
on this corpus or any adapter descended from it; only the *lesson* (game-grounded prompts
beat context-free flashcards) informed how Burl's tool responses are shaped. See
[[lem-to-burl-handoff]].

## Links

[[rules-adapter]] [[v4-adapter]] [[v5-adapter]] [[narration]] [[kerry-curriculum]] [[trump-drilling]] [[stage-0-v4-comprehension-eval]] [[flexible-grader]] [[base-model-pivot-qwen]] [[1d3e1b7]]
