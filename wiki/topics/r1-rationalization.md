---
title: R1 Rationalization
kind: topic
first_seen: a8bccfa
last_updated: be7efc4
status: active
---

## Overview

R1 rationalization is the failure branch of [[star]] in [[lem]]. When a model trace fails [[k1-grading]] with a legal-but-suboptimal action, the correct action (the bot's action) is revealed and the model is asked to produce a reasoning trace that arrives at that action. The resulting rationalization is kept as training data (lem/OVERVIEW.md @ a8bccfa).

## Rationale

The STaR paper established this approach: training on rationalizations of correct answers recovers signal that pure self-play would discard. It is simpler than DPO-style preference learning, which requires pairs of preferred and rejected outputs (lem/OVERVIEW.md @ a8bccfa).

## Scope narrowed (fb47ab3)

Prior to fb47ab3 (ingest 4 frontier, 8c5fbca), R1 was invoked on `illegal` and `parse_fail` traces as well as `fail` traces. That behavior was superseded.

The shift: "traces that arrive at impossible states (illegal moves, unparseable actions) are poison — the reasoning chain is corrupted even if intermediate steps looked reasonable. Don't rationalize, just discard." (fb47ab3)

At this frontier, only `fail` traces (legal action, E[Q] below the bot's threshold) are rationalized. `illegal` and `parse_fail` traces are discarded without training. This preserves R1's utility for strategy learning while preventing the model from internalizing rationalizations built on top of a corrupted reasoning chain.

The illegality rate (`illegal_rate` in wandb) is now a free diagnostic: high (~40%) signals Stage 0 needs more rules work; low (~5%) signals the model knows the rules and training can focus on strategy. See [[decisions/discard-illegal-traces]] (fb47ab3).

## Implementation

R1 is invoked inside the [[star-harness]] on traces graded `fail`. The single-GPU loop (`star_loop.py`) runs the rationalization pass as a second vLLM batch call on the same loaded model, keeping the GPU hot between steps (lem/gemma_star/star_loop.py @ fb47ab3).

## Rationalization-SFT bootstrap (v10)

v10 ([[v10-adapter]]) introduces an alternative to STaR-iteration rationalization: a one-shot SFT bootstrap (0c7392f).

**Procedure:**
1. Scout 500 trick-6 decisions using the v9 adapter.
2. Filter with the [[rationalization-verifier]] (6 engine checks) → 331 clean rationalizations (66% pass rate).
3. Joint-train in one SFT pass: v9 comprehension data + the 331 rationalizations upweighted 10×.

**Results:** 83% comprehension preserved. Transition test on open-ended prompts: 55/100 bot-match (crossing 50%), 96/100 legal moves, visible reasoning, commits to plays. A direct path from "knows the rules" to "plays with reasoning" without iterated STaR (0c7392f).

**Bug documented:** `rationalize-v1`, a variant trained without the v9 foundation (fresh LoRA on rationalizations only), transferred the reasoning habit but not domain knowledge — 0/100 bot-match, 97/100 no-play. Lesson: rationalization training must build on top of comprehension, not replace it. Joint training from the comprehension base with mixed data avoids this (0c7392f).

**Mask fix (be7efc4):** TRL's `SFTConfig` was computing loss over the full sequence (prompt + answer), diluting the answer gradient ~9× by memorized prompt tokens. Switching dataset format from `messages` → `prompt`/`completion` enables `completion_only_loss=True`. Transition bot-match unchanged at 55/100 after the fix — confirming bot-match is not a gradient-allocation problem; capacity or STaR is the next lever. See [[decisions/sft-completion-only-loss]] (be7efc4).

## Open questions

The original open question ("sufficient for Stage 3+, or does longer lookahead need
DPO?") is moot — Stage 3+ (the [[backwards-curriculum]] beyond trick 6) never happened;
LEM plateaued in Stage 1 and pivoted to [[burl]] before the question could be tested. R1
itself remains in active use: [[burl]] runs the same reveal-and-justify mechanism on tool-use
trajectories (see [[star]] "STaR on Burl's trajectories").

## Links

[[star]] [[k1-grading]] [[lem]] [[star-harness]] [[decisions/discard-illegal-traces]] [[rationalization-verifier]] [[v9-adapter]] [[v10-adapter]] [[decisions/sft-completion-only-loss]]
