---
title: Burl Adapter Line
kind: entity
first_seen: 2026-07-13
last_updated: 2026-07-13
status: complete
---

## What it was

The succession of LoRA adapters for [[burl]]'s tool-using play task: teach
[[gemma-4-e2b]] to reason about a Texas 42 decision through tool calls and commit a
play. Three shipped iterations, all on 2026-04-19, all SFT on STaR corpora over the same
base model. Each adapter below is a receipt page with full provenance and eval detail.

## The chain

[[burl-iter0-adapter]] → [[burl-iter1-adapter]] → [[iter3-rules-adapter]].

Numbering note: iter-2 shipped no page of its own; its training recipe (rank 16,
3 epochs, LR 1e-4) is recorded on [[iter3-rules-adapter]], which reused it.

All evals below are the same 10-decision held-out set; the untrained spike v2 baseline
(no framing, no primer) sits at 88.9% bot-match.

| Adapter | Base | Corpus | Headline eval | What it fixed / broke |
|---|---|---|---|---|
| [[burl-iter0-adapter]] | [[gemma-4-e2b]] | 50 entries (27 K1 wins + 23 hinted rationalizations), full primer | 60% bot-match, mean E[Q] Δ −3.33, 100% legal | Proved the training pipeline (clean descent, tool-call format intact); regressed judgment — baked in Layer 1's eq-shy pathology |
| [[burl-iter1-adapter]] | [[gemma-4-e2b]] | 30 entries, trimmed ~500-word primer | 80% bot-match / Δ −0.76 on the 5 completed; 5/10 retry-exhausted | Deeper reasoning; broke commit discipline (trim removed a load-bearing scaffold) |
| [[iter3-rules-adapter]] | [[gemma-4-e2b]] | Corpus harvested with `enable_rules_tools=True`, primer off | **90% bot-match, 0 retry-exhausted, 100% first-legal** | Validated [[rules-as-tools]] — `trick_winner_if` usage increased after SFT; measured under a tool-response confound (below) |

## Supersession story

- **iter-0 → iter-1**: the diagnosis was corpus shape, not training mechanics — iter-0
  learned "Layer-1 Gemma" including its failure to call distribution tools. iter-1
  re-harvested on a trimmed primer ([[primer-tradeoff]]), which traded commit discipline
  for reasoning depth.
- **iter-1 → iter-3-rules**: rules content moved from prompt text to callable tools
  ([[rules-as-tools]]). Best result on the line: 90% bot-match with zero
  retry-exhaustion and 100% first-legal.

## How the line ended

Burl dormancy. [[burl]] has been dormant since `dbadb5f` (2026-04-19); no iter-4 was
attempted. The winner's 90% was measured under the [[gemma-tool-response-shape]]
confound — Gemma 4's chat template silently dropped `role="tool"` messages, so every
rollout that built and evaluated these adapters ran with tool outputs invisible to the
model — and [[chat-template-fix-validation]]'s proposed re-test never ran (full caveat
on [[iter3-rules-adapter]]). As of jud v1, the project's play mechanism consumes no LoRA
adapter at all ([[champion]] runs "zero adapter"). Like the [[stage-0-adapter-line]],
this line ended because the mechanism it fed was abandoned, not because a successor beat
it.

## Receipts

[[burl-iter0-adapter]] · [[burl-iter1-adapter]] · [[iter3-rules-adapter]]
