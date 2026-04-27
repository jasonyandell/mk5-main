---
title: Perf Sprint — Playbook
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

The "let's friggin rock this perf problem" entry point.

## Kickoff

Paste into a fresh session at the project root:

> Read [[perf-sprint]]. Run a perf sprint targeting `<GOAL>`. e.g. *"drive Burl per-decision latency from current baseline toward ~2s on this M5 Max."*

## The contract

Make it faster. Use the available measurements (wall, K1, regret) to verify the faster version is equivalent to the baseline. **Don't give up.** Crashes are work, not a stop sign — read the traceback, apply the smallest fix, re-run. Stop only when the goal is hit or the user says stop.

That's it. The rest is detail.

## How to work

1. Write `scratch/PERF_GOAL.md` from [[perf-sprint-goal]] — sprint goal, anchor metric, equivalence bar.
2. Register the [[perf-sprint-loop]] message verbatim. The loop re-anchors on the goal at every fire and names idle as a bug.
3. Work the [[perf-sprint-levers]] ladder in order. Append wins, retire dead levers, evolve the ladder.
4. When the bench misbehaves, check [[perf-sprint-traps]] before re-deriving. Add new traps as you find them.
5. Append a post-mortem to [[perf-sprint-history]] when the sprint ends.

## Scope

Calibrated for Burl-style mlx-lm inference on Apple Silicon (Gemma 4 E2B, M5 Max). The lever ladder, trap recipes, and `decode_tok_s` thresholds assume that workload. Other perf sprints can borrow the contract and the loop discipline; fork the levers and traps.

## Links

[[perf-sprint-goal]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]]
