---
title: Burl Move 4 R3 Spike — Native Tool-Use 88.9% Bot-Match
kind: experiment
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Summary

Exploratory spike migrating Burl from XML to Gemma 4's native `<|tool_call>` tool-use format on the same 10 held-out decisions as [[experiments/burl-move3-base]]. Result: +28.9pp bot-match, +18.9pp K1, mean E[Q] delta halved. Tool-use breadth finally real.

([burl/eval/run_move4_spike.py @ 3781dce](../sources/3781dce.md))

## Setup

- **Model:** base [[gemma-4-e2b]], zero fine-tuning
- **Format:** Gemma 4 native `<|tool_call>` via `tools=[...]` in chat template (`burl/harness/agent_runner_native.py`)
- **Two variants:** v1 (no `commit_play` tool — commit expressed as XML, causing exhaustion); v2 (`commit_play(domino_id)` added as native tool)
- **Eval set:** same 10 decisions as Move 3
- **Cost:** $0.16 total spike

## Results

| | Move 3 XML | spike v1 native | spike v2 +commit_play |
|---|---|---|---|
| n_completed | 10/10 | 6/10 | 9/10 |
| legal_rate | 100% | 100%* | 100% |
| first_legal_rate | 100% | 60% | 90% |
| bot_match_rate | 60% | 66.7%* | **88.9%** |
| p_eq_geq_bot (K1) | 70% | 66.7%* | **88.9%** |
| mean_eq_delta | −4.65 | −4.05 | **−1.92** |
| eq_outcome_distr | 0 | 10 | **15** |
| trump_declared | 0 | 0 | **9** |
| hallucinated tools | play:8 | 0 | 0 |

*= "among completed"; v1 had 40% exhaustion so not directly comparable.

## Two blockers diagnosed and fixed

**Blocker 1 — XML commit off trained path:** Gemma expressed commits via `<|tool_call>` envelopes when the harness expected `<commit>INT</commit>` XML. v1 exhausted retries on 4/10 decisions because it couldn't complete the commit path. Fix: added `commit_play(domino_id)` as a native tool — letting Gemma use its own grammar.

**Blocker 2 — Retry budget insufficient for native format:** `max_retries=3` allows 4 total iterations. XML could pack multiple `<tool>` tags per completion. Native format emits exactly one `<|tool_call>` per turn. Move 4 production default: 7.

## Significance

The bet that "small models are better at asking questions than memorizing" survives — but only when the model is spoken to in its native grammar. The harness must adapt to the model's post-training, not the other way around. See [[decisions/native-tool-use-format]].

The OVERVIEW principle added at this commit: *"Go with the model's grain; catch it doing right... STaR trains the model's own best behavior back into itself — their words, their corrections, their self-checks."*

## Known limits at this frontier

- Gemma's reasoning is still "generic card game" — never mentions partner seat, counts, offense/defense, target score. Richer prompt framing (layer 1) and STaR rationalizations naming 42 concepts (layer 2) are the fixes.
- 1/10 decision chained 8 tool calls without committing — policy issue, expected to shorten under STaR SFT.

## Related pages

[[burl]] · [[gemma-4-e2b]] · [[tool-orchestration]] · [[k1-grading]] · [[decisions/native-tool-use-format]] · [[experiments/burl-move3-base]] · [[sources/3781dce]]
