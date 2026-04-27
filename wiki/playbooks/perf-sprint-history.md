---
title: Perf Sprint — Session History
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Append-only log of perf sprints. Each entry: date, target, outcome, what worked, what didn't, what changed in the playbook. Future sprints read this to start with the wisdom of past sprints already encoded.

## 2026-04-27 — Sprint 1 — Burl per-decision latency

**Target:** 26s/decision → 2s/decision on M5 Max (13×).

**Outcome:** Wall improvement on full-N never attributed (bench crashed at ~80% twice). 5-row directional evidence positive. -56% peak memory confirmed (Q4 PLE-safe Unsloth UD vs bf16). **Wrap was premature** — levers remained; orchestrator stopped instead of fixing the bench's two crash modes (5-line and one-flag patches respectively).

**What worked:**
- Four-scribe team shape: D (measurement) → B + A (parallel cheap + big) → C (aggressive). One worktree per scribe.
- Research-only mode for early scribes while GPU was busy. C in particular surfaced three structural dragons in 30 min before any GPU bench.
- Honest in-session retractions: [[burl-perf-phase1]] 1.66× → 0.3%; [[burl-perf-phase2]] 2.1× → tie. Source-reading caught both.
- Wiki-first culture — scribes wrote findings as they went; the wiki is meaningfully richer.

**What didn't:**
- Cross-scribe GPU contention discovered after hours of corrupted data. Now codified in [[mlx-cohort-bench-discipline]] and the `bench.lock` proposal.
- Bench at 5-row has 3.4× same-config noise floor on M5 Max. Sub-2× wall claims unattributable on that subset. Codified in `project_perf_subset_5_noise_floor` user-memory.
- Phase 4 crashed twice (different bugs) — orchestrator wrapped instead of fixing forward. The rationalization-as-discipline failure mode.
- Original loop message didn't restate the goal; orchestrator drifted under stress.

**What changed in the playbook:**
- This session is the reason the playbook exists. [[perf-sprint]] + [[perf-sprint-loop]] + [[perf-sprint-goal]] + [[perf-sprint-levers]] + [[perf-sprint-traps]] are all artifacts of "what we wished we had at minute 0."
- Wrap conditions tightened to three explicit predicates.
- Loop message restates goal at every fire + names idle as a bug.
- Pre-flight smoke test required.
- `bench.lock` mutex convention proposed (not yet implemented).
- Paired-baseline protocol elevated to mandatory.

**Open levers when sprint ended:** see [[perf-sprint-levers]] (8 entries; #1 — bench resilience — was 1 hour of work away from giving full-N attribution).

**Worktrees + branches at session end:**
- `perf/bench` — Phase 0 bench harness
- `perf/cheap` — Phase 1 turn-aware budgets + Lever 2 empirical brief
- `perf/batch` — Phase 2 dispatcher + Lever 1 root-cause + cohort spec + mlx-lm Internals
- `perf/aggressive` — Phase 3 quant audit + PLE-safe set + production recommendation

Suggested merge order back to `forge`: bench → cheap → batch → aggressive → forge. `perf/aggressive` carries the most-current wiki.

## Append a sprint reflection

When a sprint ends (success OR honest stop), add a section here with the structure above. The point is durable institutional memory across garage weekends.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-traps]]
