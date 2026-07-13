---
title: EQ-gate STaR (rejection sampling by eq_delta threshold)
kind: topic
first_seen: 2026-04-19
last_updated: 2026-04-19
status: retired
---

## Overview

EQ-gate is a stricter acceptance criterion for [[star]] traces in [[burl]], replacing the binary [[k1-grading]] pass/fail with a continuous threshold on `eq_delta` (the gap between the model's E[Q] and the bot's E[Q]). A trace is accepted only if `eq_delta >= threshold` (default 0.25), meaning the model's play must be within `threshold` points of the argmax — not merely at or above the bot's value (f164796, 761587c).

## Motivation

iter-0's regression traced to corpus quality: K1's binary pass/fail accepted any play that beat the bot, including marginal decisions barely above the threshold. These marginal traces dragged the adapter toward mediocrity. EQ-gate keeps only traces where the model's decision was genuinely competitive, not just technically passing (f164796).

## Mechanism

Implemented in `burl/harness/eq_gate.py` with three pure functions:

- `check_commit(eq_delta, threshold)` — returns accept/reject.
- `gate_feedback_prompt(variant)` — generates a non-leaking nudge prompt (three variants: `minimal`, `tool_nudge`, `social`). The nudge does not reveal `bot_play` or per-play E[Q] values — this is invariant-tested.
- `classify_rationalization(outcome)` — classifies gate outcomes: `converged_first_try`, `self_corrected`, `forced_flip`, `stubborn`, `exhausted`.

Wired into the STaR orchestrator in 761587c. Phase B fires the gate on legal-but-sub-optimal commits and nudges; Phase C composes SFT records only for `self_corrected` verdicts (tagged `source=eq_gate_self_correct`) (761587c).

## CLI flags

`--gate-variant {minimal,tool_nudge,social}` (default `tool_nudge`), `--max-gate-retries N` (default 1), `--eq-epsilon F` (default 0.25).

## Relationship to K1

K1 asks "did the model beat the bot?" EQ-gate asks "did the model come close enough to the argmax to be worth training on?" Both are acceptance criteria on the same E[Q] signal; EQ-gate is strictly stricter.

**Asserted, unverified.** EQ-gate was wired into the STaR orchestrator (761587c) but no
experiment page or commit ever reports a run that used it to gate a real training
corpus — the "produces a smaller but higher-quality corpus" claim below was the design
intent, not a measured result.

STaR with EQ-gate produces a smaller but higher-quality corpus than STaR with K1 alone (f164796).

## Links

[[burl]] [[star]] [[k1-grading]] [[expected-q-value]]
