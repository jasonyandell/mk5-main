---
title: Perf Sprint — Session History
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Append-only log of perf sprints. Each entry: target, outcome, what worked, what didn't, what changed in the playbook. Future sprints read this so the wisdom of past sprints is already encoded.

## 2026-04-27 — Sprint 1 — Burl per-decision latency

**Target:** 26s/decision → 2s/decision on M5 Max (13×).

**Outcome:** Wall improvement on full-N never attributed (bench crashed at ~80% twice). 5-row directional evidence positive. -56% peak memory confirmed (Q4 PLE-safe Unsloth UD vs bf16). Sprint wrapped before the goal was hit and with levers still on the ladder.

**What worked.** Four parallel work-streams (measurement, cheap wins, big lever, aggressive) made the sprint productive in wall-clock terms. Research-mode work in parallel with someone else holding the GPU turned idle into structural-finding time. Honest in-session retractions caught two false wins (1.66× → 0.3% and 2.1× → tie) that source-reading exposed. Wiki-first culture — findings written as they were discovered — meaningfully enriched the wiki.

**What didn't.**
- **Cross-process GPU contention** corrupted hours of data before being recognized. Apple Silicon's unified-memory GPU is one resource and `ps aux` doesn't expose Metal contention. Codified in [[perf-sprint-traps]] (`decode_tok_s` < 100 detection signal, `bench.lock` convention).
- **The "3.4× noise floor" claim was a misdiagnosis.** Same-config wall variance of 3.4× across runs was attributed to small-subset noise; it was actually the contention above. Clean-GPU floor is ±4%. The misdiagnosis matters because "the bench is unreliable" became a felt-true reason to wrap, when it was really a fixable systems problem.
- **Bench crashed twice at ~80% of full-N runs.** Each crash had a small fix (try/except quarantine; `prefill_batch_size=2`). Both are now lever #1 and the first two trap recipes.
- **The sprint wrapped instead of fixing forward.** Two crashes plus the false noise floor produced a felt-true "ship the digest" instinct that read as discipline but was rationalization. Lever #1 was about an hour of work away from giving full-N attribution. This is the failure mode the new playbook's contract is shaped around: **don't give up — crashes are work, not a stop sign.**

**What changed in the playbook.** This session is the reason the playbook exists. The shape was over-specified at first (mood framing, wrap predicates, mandatory rules, scribe team shape) — clarified on 2026-04-27 to: clear goal, equivalence bar, don't-give-up clause, lever ladder, trap recipes, history. Trust the model to do the rest.

**Open levers when sprint ended:** see [[perf-sprint-levers]] (8 entries; #1 — bench resilience — would have unblocked full-N attribution).

**Worktrees + branches at session end:**
- `perf/bench` — Phase 0 bench harness
- `perf/cheap` — Phase 1 turn-aware budgets + Lever 2 empirical brief
- `perf/batch` — Phase 2 dispatcher + Lever 1 root-cause + cohort spec + mlx-lm internals
- `perf/aggressive` — Phase 3 quant audit + PLE-safe set + production recommendation

Suggested merge order back to `forge`: bench → cheap → batch → aggressive → forge. `perf/aggressive` carries the most-current wiki.

## Append a sprint reflection

When a sprint ends, add a section here. Durable institutional memory across garage weekends.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-traps]]
