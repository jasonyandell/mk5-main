---
title: From LEM to Burl
kind: trail
first_seen: 8d26e0d
last_updated: 8d26e0d
status: active
---

This trail traces the transition from [[lem]] to [[burl]] — the first major project-to-project handoff in the history replay. It connects the end state of LEM, the diagnosis that motivated a different approach, the pivot itself, and what carries forward into Burl.

## 1. The end state of LEM

At the close of the LEM replay (ingest 16, [[experiments/v10-maskfix-breakthrough]]), the project's best adapter is `jasonyandell/qwen3-1.7b-texas42-stage0-v10-maskfix` ([[v10-adapter]]):

- **86% comprehension overall** on the 14-category [[game-context-qa]] eval — matching [[qwen3-14b]] v9 at 1/3 the compute
- **55/100 bot-match** on open-ended play selection (crosses 50% on transition test)
- **96/100 legal moves**
- **Base:** [[qwen3-1.7b]]

The adapter produces lucid position-explanation traces and knows the rules of [[texas-42]] with high accuracy. The `is_trump` gap that ran through first/second/third contacts is closed at 100%.

## 2. The diagnosis

The 55/100 bot-match figure did not move with gradient reallocation ([[decisions/sft-completion-only-loss]]) or capacity scaling ([[experiments/qwen-14b-capacity]]). The [[experiments/v10-maskfix-breakthrough]] experiment showed that 3 of 4 stuck comprehension metrics moved decisively when gradients were redirected — but transition bot-match was literally unchanged.

The interpretation: open-ended play selection is a different skill than fact recall and comprehension. It is not clear that more SFT signal of the same shape (flashcards, rationalizations, structured templates) will move the number. The ceiling is not a data-quality or gradient-allocation problem — it may be a fundamental mismatch between the training shape and the target behavior.

## 3. The pivot

[[burl]] does not continue LEM's curriculum. It tries a different shape entirely:

- **Different mechanism:** tool orchestration instead of weight memorization. The [[engine]] answers rules questions; [[zeb]] answers belief questions; Burl reasons between tool calls and commits to a play.
- **Different base model:** [[gemma-4-e2b]], chosen for agentic function-calling strengths rather than comprehension accuracy (the rationale that drove the [[decisions/base-model-pivot-qwen]] for LEM is reversed — comprehension matters less when tools provide facts).
- **Different training data shape:** trajectories (tool calls + play + outcome) instead of flashcards.
- **Different eval:** legal rate + retry count + bot-match, not comprehension accuracy.
- **Different product slot:** "play against AI" instead of "explain this position."

The bet: small models are better at asking the right questions and synthesizing tool responses than at memorizing many facts. If true, the tool harness gives a 2B-class model competent play without the months of comprehension curriculum LEM required.

(burl/OVERVIEW.md @ [[sources/8d26e0d]])

## 4. What carries forward from LEM

**Rules infrastructure:** The [[rules-adapter]] curriculum work — primers, Q&A categories, engine-verified ground truth — informed what facts the engine needs to supply. Burl doesn't train on it, but the work sharpened understanding of where 2B-scale models fail on rules knowledge.

**Narration work:** The [[narration]] pipeline (second-person prose, public-state blocks per [[decisions/public-state-block]]) produced the decision datasets that grounded LEM's play experiments. Burl draws on the same game-state representations.

**Zeb:** Already resident in `forge/zeb/`, now promoted to a first-class Burl dependency. The 72% top-1 belief accuracy that Zeb achieved through forge's training pipeline becomes the engine behind Burl's `get_belief` tool.

**Discard-illegal policy:** [[decisions/discard-illegal-traces]] applies equally to Burl's STaR iterations — traces arriving at impossible states are still poison regardless of which model generates them.

**[[rationalization-verifier]] philosophy:** The verifier's principle — engine-verified facts make hallucinations irrelevant — is the direct ancestor of Burl's tool discipline. Burl doesn't rationalize facts; it asks for them.

**Vocabulary:** The LEM cleanup in [[sources/8d26e0d]] — solver / E[Q] framework / E[Q] bot — is the precise vocabulary both projects now share. Burl is graded against the E[Q] framework, not the perfect-info solver.

## 5. What changes

| Dimension | LEM | Burl |
|---|---|---|
| Training data shape | Flashcards: `(state, Q) → A` | Trajectories: tool calls + play |
| Eval primary metric | Comprehension accuracy | Bot-match + legal rate + retry count |
| Base model rationale | Comprehension accuracy (Qwen won) | Agentic tool-use strength (Gemma wins) |
| Rules knowledge source | In weights (trained in) | In tools (engine provides at inference) |
| Belief knowledge source | Not modeled explicitly | Zeb via `get_belief` tool |

## 6. What stays shared

Both projects run on [[forge]] infrastructure:
- **Engine** (`src/core/`) — rules, legality, state transitions
- **E[Q] framework** (`forge/eq/`) — expected-value computation over hidden-hand distributions
- **Solver** (`forge/oracle/`) — dev-time tool, never called by either model directly
- **Zeb** (`forge/zeb/`) — now a Burl tool dependency
- **Visualizer** — `eq_pdf_discs.html`, `eq_surface_3d.html`, `eq_game_journey.html`

Both projects use [[modal]] for GPU compute and HuggingFace for adapter storage.

## 7. Resolved since this trail was written

This trail reflects the LEM/Burl boundary as of 8d26e0d. The provisional question it left
open — whether the Gemma 4 base is agentic enough at 2B scale — is answered **yes**: see
[[burl-move3-base]] and [[burl-move4-native-spike]]. The model choice was not revisited;
Burl's subsequent STaR iterations and the 2000-decision harvest (see [[star]]) all build on
the Gemma 4 E2B base.

## Related pages

[[lem]] · [[burl]] · [[forge]] · [[gemma-4-e2b]] · [[qwen3-1.7b]] · [[tool-orchestration]] · [[zeb]] · [[engine]] · [[v10-adapter]] · [[experiments/v10-maskfix-breakthrough]] · [[decisions/base-model-pivot-qwen]] · [[decisions/discard-illegal-traces]] · [[rationalization-verifier]] · [[sources/8d26e0d]]
