---
title: Reference-Trace Distillation (Haiku 4.5 teacher)
kind: topic
first_seen: 1f13f92
last_updated: b5d05de
status: active
---

## Overview

Reference-trace distillation uses [[haiku-4-5]] (via the Anthropic Agent SDK) as a stronger teacher model to produce target tool-use traces, then distills the traces into [[burl]] (Gemma 4 base) via SFT. It is an alternative to purely self-generated [[star]] corpora, motivated by the observation that Burl's rationalizations showed a suspicious 100% convergence pattern — Gemma formatting hints rather than reasoning — suggesting the self-generated signal was weak (1f13f92).

## Motivation

iter-1's rationalization pass produced 23/23 convergences on first hinted re-prompt. A stronger model may produce traces with genuine multi-step reasoning worth learning from, rather than reformatted ground-truth answers (1f13f92).

## Implementation

8 in-process `@tool` handlers wrap `engine.py` and `eq_distribution.py`, exposed to the claude-agent-sdk via MCP. Traces are collected over held-out decisions (1f13f92, b5d05de).

**SDK pitfall documented:** the Agent SDK falls back to non-streaming `--print` mode when `prompt` is a plain string, silently breaking in-process MCP RPC. Fix: pass prompt as an async iterable of stream-json user messages (1f13f92).

## Haiku 4.5 baseline results (N=30)

30-decision run over the iter-0/iter-1 held-out eval union. Cost $0.78 (35% under cap). 29/30 complete (one 10-turn timeout). **72.4% bot-match** (1f13f92, b5d05de).

Haiku's characteristic reasoning pattern:
- Leads with prose, batches parallel engine-fact probes.
- Calls `eq_outcome_distribution` selectively on top 1-2 candidates — never sweeps.
- Self-checks `is_legal` before `commit_play`; zero retries needed.
- Uses 7 distinct tools per decision vs Gemma iter-1's 3.

## Key findings for iter-3 planning

1. **`conditional_outcome` has zero usage** across all 30 decisions — even at Haiku's ceiling, the counterfactual probe isn't reached zero-shot. If Burl is to use it, STaR must synthesize demos explicitly (b5d05de).

2. **4 big-gap misses all skipped `eq_outcome_distribution`**, relying on prose-only reasoning. A prompt nudge ("probe both candidates with E[Q] on close calls") would likely recover 2-4 of these (b5d05de).

3. **Haiku's "go with the grain" patterns** — 7 tools/decision, no retries, `is_legal` self-check — are worth imprinting via distillation if iter-3 takes that direction (b5d05de).

The 72.4% Haiku baseline sets the reference ceiling for [[burl]] iter-3+ comparisons. Traces live in `scratch/` (gitignored) (b5d05de).

## Links

[[burl]] [[haiku-4-5]] [[star]] [[r1-rationalization]] [[tool-orchestration]]
