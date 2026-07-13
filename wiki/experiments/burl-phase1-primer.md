---
title: "Phase 1: Primer + 42-Aware Framing"
kind: experiment
first_seen: 2026-04-19
last_updated: 2026-07-13
status: complete
---

## Summary

Layer 1 of Burl's STaR substrate. The rules primer + a 42-aware framing block are prepended to every Burl decision prompt. Bot-match regresses 18.9pp from the spike, but 42-specific vocabulary appears in traces for the first time — providing the domain substrate [[star]] needs to train on.

([SPIKE_REPORT.md @ b8116b5](../sources/b8116b5.md))

## Setup

- **Model:** base [[gemma-4-e2b]], native `<|tool_call>` format
- **Prompt additions:** full `lem/rules/primer.md` (1549 words, engine-verified tournament 42 rules) + "Current decision — 42-aware context" block: partner seat, left/right opponent seats, team role (offense/defense), bid contract, tricks-completed, score
- **Total prompt size:** ~11 KB / 2.7K tokens — fits within 8192 `max_model_len`
- **New tool:** `game_summary(game_state)` — one-shot structured view (not yet wired into native registry; available for follow-up iterations)
- **Eval set:** same 10 held-out decisions as [[burl-move3-base]] and [[burl-move4-native-spike]]

## Results

| Metric | spike v2 | Phase 1 |
|---|---|---|
| bot_match_rate | 88.9% | 70% |
| K1 (p_eq_geq_bot) | 88.9% | 70% |
| legal_rate | 100% | 100% |
| eq_outcome_distribution calls | 15/10 | 2/10 |
| 42 vocabulary mentions/trace | 0 | 5–11 |

## Tool-use shift

The primer shifts Gemma's attention toward lightweight rule-confirming tool calls (`is_legal`) and away from distribution tools (`eq_outcome_distribution`). The vocabulary in traces, however, is qualitatively new: traces now contain "partner", "team", "offense/defense", "count", "bid" where previously they had none. Trace excerpt: *"I am at seat 1 (Team 1)... Role: I am on DEFENSE. Team 0 bid 30 count. I need to SET them."*

## Verdict

This is the STaR substrate, not the final product. The vocabulary the corpus must carry into SFT is now present. The expectation at this frontier: iter-0 training should recover bot-match while keeping the vocabulary. That expectation was not met — see [[burl-iter0-eval]] and [[primer-tradeoff]].

## Related pages

[[burl]] · [[gemma-4-e2b]] · [[tool-orchestration]] · [[primer-tradeoff]] · [[burl-move4-native-spike]] · [[burl-phase2-starcorpus]] · [[b8116b5]]
