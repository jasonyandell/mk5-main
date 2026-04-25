---
title: Tool Orchestration (Burl's core philosophy)
kind: topic
first_seen: 8d26e0d
last_updated: 0545342
status: active
---

## Overview

Tool orchestration is the central design bet of [[burl]]: instead of teaching a small model the game in its weights, teach it to ask the right questions and synthesize the answers. The engine answers facts about rules and visible state; [[zeb]] answers beliefs about hidden state; Burl reasons between calls and commits to a play. The [[engine]] validates the committed play; if illegal, Burl gets another turn (burl/OVERVIEW.md @ 8d26e0d).

## The bet

> "Small models are much better at asking the right questions and synthesizing tool responses than at memorizing many facts." (burl/OVERVIEW.md @ 8d26e0d)

If this holds, the tool-call harness lets a 2B-class model play competently without the comprehension curriculum [[lem]] spent months building — because tools replace memorization. [[lem]] reached 86% comprehension accuracy and 55% bot-match on open play after v1 → v10-maskfix. The mask-fix experiment showed capacity and gradient allocation didn't move bot-match, suggesting open-ended play selection requires something structurally different. Tool orchestration is that structural difference (burl/OVERVIEW.md @ 8d26e0d).

## Three authorities

Burl composes three distinct information sources at inference time:

**[[engine]]** (`src/core/`) — the authority on rules and visible state. Epistemic facts only:

| Tool | Returns |
|---|---|
| `is_legal(dom)` | Legality + reason if not |
| `is_trump(dom)` | Trump under current declaration |
| `unseen()` | Dominoes not in hand and not yet played |
| `void_audit(player, suit)` | Has this player been proven void? |
| `trump_declared()` | e.g. "blanks", "fives", "doubles" |

**[[zeb]]** (`forge/zeb/`) — **parked** after calibration eval (d9baf3b). Advertised 72% top-1 accuracy was inflated by already-played dominoes; hidden-only top-1 is 39%, Brier 0.224, ECE 0.067. Not useful enough to anchor Burl's belief reasoning. See [[decisions/zeb-parked-eq-primitive]] and [[experiments/zeb-calibration-eval]].

**E[Q] N=10 outcome PDF** — the belief primitive that replaced Zeb. `eq_outcome_distribution()` returns a distribution over game outcomes given the current visible state, computed by running the E[Q] framework over N=10 hidden-hand samples. Counterfactual shift validated on seed 900013 (Δmean +15, p_make 0.6→1.0). 290ms/play, 49.8 MB peak VRAM (d9baf3b).

**Burl itself** — the reasoning layer. Decides what to ask, synthesizes the responses, commits to a play.

## Higher-order tool: conditional_outcome (parked with Zeb)

`conditional_outcome(play, assume)` was originally designed to compose Zeb × E[Q]. With Zeb parked, this tool is also parked at this frontier. The belief primitive is now `eq_outcome_distribution` directly (d9baf3b).

## Tool surface (current frontier)

**Engine tools:**

| Tool | Purpose |
|---|---|
| `is_legal` | Legality check |
| `is_trump` | Trump membership |
| `unseen` | Unseen domino pool |
| `void_audit` | Suit-void status |
| `trump_declared` | Active declaration |

**Rules tools** (when `enable_rules_tools=True`; see [[rules-as-tools]]):

| Tool | Purpose |
|---|---|
| `count_dominoes_remaining` | Count values for remaining dominoes |
| `trick_winner_if` | Who wins a trick under a hypothetical |
| `what_beats_what` | Suit/trump ranking in current context |
| `contract_progress` | Bid, tricks taken, points needed |

**Outcome tools:**

| Tool | Purpose | Notes |
|---|---|---|
| `eq_outcome_distribution` | Outcome PDF; now includes candlewax fields | Returns `distribution_shape`, `modes`, `gap_between_modes`, `suggested_counterfactuals`, `spike_drivers`, `sampling_mode` (1efb9c5, 7321952, b0952a2) |
| `what_would_change_my_mind` | Top-K assumptions that most shift E[Q] of a play | Ranked by \|shift\|; first zero-shot touch after 0/145 `conditional_outcome` streak (7321952) |
| ~~`conditional_outcome`~~ | ~~Composed Zeb × E[Q]~~ | Zero-shot invisible across 145+ decisions on all models. See [[conditional-outcome-structural-nonuse]] |

**Commit tool:**

| Tool | Purpose |
|---|---|
| `commit_play` | Commit a play (native tool channel, not XML tag) |

Forbidden evaluative tools remain: `get_eq(dom)`, `best_move()`, `simulate_plan(actions)`. The rule — tools answer "**what IS the state?**", never "**what SHOULD you do?**" — is unchanged (burl/OVERVIEW.md @ 8d26e0d).

**`eq_outcome_distribution` candlewax fields** (1efb9c5, 7321952, b0952a2):
- `distribution_shape`: `"unimodal"` / `"bimodal"` / `"multimodal"` — bimodality legible at tool surface instead of buried in 85-bin PDF.
- `modes`: list of `{center, mass}` per mode.
- `gap_between_modes`: float quantifying mode separation.
- `suggested_counterfactuals`: top-2 counterfactuals from `conditional_outcome` on top-5 × 3 seats × both modes at N=5. Opt-out via `suggest_counterfactuals=False`.
- `spike_drivers`: for bimodal/multimodal, which (seat, domino) assignments are empirically over-represented in each mode. Repackages the same information in Gemma's bid-satisfaction vocabulary rather than raw PDF shape ("grain-recon").
- `sampling_mode`: `"sampled"` or `"enumerated"`. `enumerate="auto"` uses exact enumeration when pool ≤ 12 (~7 ms, deterministic); sampling otherwise.

See [[candlewax]] for the bimodal-distribution concept (0545342).

## Legality by construction

Legal-move compliance is a software invariant, not a trained behavior. The engine retry loop means Burl cannot commit an illegal play; if it tries, the engine rejects and Burl gets another turn with the rejection in context. This is the key structural difference from [[lem]]'s approach, where illegal rate was a diagnostic metric trained against (burl/OVERVIEW.md @ 8d26e0d).

## First results (Moves 3-4)

**Move 3 — XML format** (4b3ba3d): base Gemma 4 E2B, zero fine-tuning, 10 held-out decisions, $0.09. Result: 100% legal, 0 retries, 60% bot-match, 70% K1. The harness design works; XML parsing is clean.

But Move 3 also revealed two problems:
1. Base Gemma only calls `is_legal`. Never reaches for `eq_outcome_distribution` or `conditional_outcome` — the distribution tools that are Burl's whole point. Tool-use breadth: 0 distribution calls across 10 decisions.
2. 80% of trials hallucinate a fake `play` tool. The XML `<commit>` tag is off Gemma's post-training distribution (4b3ba3d).

**Move 4 R3 spike — native tool-use format** (3781dce): migrated to Gemma's post-trained native `<|tool_call>` format, added `commit_play` as a native tool. Result: 88.9% bot-match, 88.9% K1, `eq_outcome_distribution` called 15 times across 10 decisions, zero hallucinated tools. +28.9pp bot-match over XML path.

| Metric | Move 3 XML | Move 4 native |
|---|---|---|
| Legal rate | 100% | 100% |
| Bot-match | 60% | **88.9%** |
| K1 | 70% | **88.9%** |
| `eq_outcome_distribution` calls | 0 | 15 |
| Hallucinated tools | 8 (`play`) | 0 |
| Cost | $0.09 | $0.16 |

Key insight (3781dce):

> "Go with the model's grain; catch it doing right. Small models have their own instincts — Gemma reaches for a `play` verb even when our menu doesn't define one… If `play` is what Gemma wants, `play` is what we give it."

See [[decisions/native-tool-use-format]] (3781dce).

**Remaining limits at this frontier**: reasoning is still "generic card game" — never mentions partner seat, counts, offense/defense, target score. Expected to improve with STaR teacher rationalizations naming 42 concepts explicitly (3781dce).

## Contrast with LEM

| Dimension | LEM | Burl |
|---|---|---|
| Rules knowledge | In weights (flashcards, v1–v10) | In tools (engine is ground truth) |
| Legality | Trained + K1 grading | Engine retry loop (100% by construction) |
| Belief about hidden state | In weights (limited) | E[Q] outcome PDF (Zeb parked) |
| Bot-match | 55% (v10-maskfix) | 88.9% (Move 4 R3 spike, base model) |
| Comprehension eval | 86% (v10-maskfix) | Not the target metric |

Burl and LEM are siblings, not successor/predecessor. LEM ships as a "explain this position" companion artifact; Burl targets "play against AI" mode (burl/OVERVIEW.md @ 8d26e0d).

## Relationship to learned-by-playing

[[learned-by-playing]] — learning from play-and-correction — still applies during Burl's STaR iterations. Tool-orchestration trajectories (tool-call histories + committed plays) are filtered by K1 and used as SFT corpora. The "what the model needs to know" floor is lower because tools provide facts, but the play-graded feedback loop is the same mechanism (burl/OVERVIEW.md @ 8d26e0d).

## Relationship to scratchpad-validation

[[scratchpad-validation]] required verifying that the model's stated game-facts were engine-correct before training on a trace. Burl achieves the same guarantee structurally: every factual claim in Burl's reasoning is a tool response, and tool responses are engine-derived. Tool-mediated play is an orthogonal route to "reasoning with verified facts" that doesn't require a separate validation pass (burl/OVERVIEW.md @ 8d26e0d).

## Primer trade-off

Layer 1 (b8116b5) added a full rules primer (1,549 words, 2.7K tokens) and 42-aware framing block to the system prompt. The effect was a two-sided trade (b8116b5, fd6032b):

**Cost:** Bot-match dropped from spike v2's 88.9% to 70%. `eq_outcome_distribution` usage collapsed from 15 calls (on 10 decisions) to 2 calls (on 10 decisions, and only 8 on 50 decisions in Phase 2). Mean tokens per decision: ~16s warm time → ~55-65s. The model spends attention on rules text that the tools were designed to provide.

**Benefit:** 42-aware vocabulary in traces went from 0 mentions to 5-11 per trace: partner, team, offense/defense, count, bid. STaR needs that vocabulary present in the corpus for it to be distilled into the adapter.

**The trade:** "Primer buys vocabulary; costs tool-use breadth." For STaR corpus harvesting, the vocabulary is load-bearing — without it, the adapter has nothing 42-specific to learn. For pure K1 performance, the primer is a liability. See [[decisions/primer-tradeoff]] (b8116b5).

**Phase 4 consequence (789e14d):** iter-0 trained on Phase 2's 50-entry corpus reproduced Layer 1's pathology. The adapter learned "Layer-1 Gemma" baked into weights: `is_legal`-heavy, `eq_outcome_distribution`-shy, primer-contaminated. Bot-match fell to 60% (vs 70% Layer 1, 88.9% spike v2). Next step: trim or remove primer, keep the 42-aware framing block alone, re-harvest (789e14d).

## Open questions at this frontier

- Can Gemma 4 E2B tool-use reliably at 2B scale? **Resolved 3781dce**: 88.9% K1 on Move 4 spike. Premise confirmed.
- Is Zeb's 72% belief accuracy useful enough? **Resolved d9baf3b**: Zeb hidden-only is 39%, not 72%. Parked; E[Q] PDF is the belief primitive.
- Does tool-mediated reasoning transfer to tool-less inference if tool latency becomes prohibitive? Still open — measured via ablation eval. (?)
- Will STaR iterations teach domain-specific reasoning (partner seat, counts, offense/defense) or will generic card-game reasoning persist? **Partially answered (789e14d)**: iter-0 reproduced Layer-1 pathology; the primer-contaminated corpus prevented learning. With primer trimmed, this remains open. (?)
- Does trimming the primer (keeping only the 42-aware framing block) recover toward spike v2's 88.9% bot-match while retaining vocabulary? Open — planned iter-1. (?)

(d9baf3b, 4b3ba3d, 3781dce, b8116b5, 789e14d)

## Links

[[burl]] [[engine]] [[zeb]] [[lem]] [[forge]] [[star]] [[learned-by-playing]] [[rules-adapter]] [[scratchpad-validation]] [[expected-q-value]] [[candlewax]] [[rules-as-tools]] [[conditional-outcome-structural-nonuse]] [[decisions/native-tool-use-format]] [[decisions/zeb-parked-eq-primitive]] [[decisions/primer-tradeoff]] [[experiments/zeb-calibration-eval]]
