---
title: Perf Sprint — Loop Message (verbatim)
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

The `/loop` message to register at sprint start. Paste verbatim, replacing the bracketed goal token.

## Loop message

~~~
/loop 10m

GOAL: <restate the sprint goal here, e.g. "drive Burl per-decision
wall_s on perf_subset_5 from ~26s toward ~2s on M5 Max">.

CONTRACT METRIC: wall_s_per_decision on perf_subset_5, paired baseline
immediately before each variant. End-to-end (tokens + tools + harness).
EQUIVALENCE GATE: K1_match >= 60% AND regret_delta within ±10% on the
same paired run. If the gate fails, status = discard regardless of wall.

You are the orchestrator. /loop is SUPERVISION, not the driver.
The worker drives iterations: when a worker returns, you immediately
spawn the next one — you do NOT wait for /loop to fire. /loop only
matters while a worker is in flight (status pings, stuck-worker
recovery) or to re-anchor between iterations.

EACH FIRE:
  1. Re-read scratch/<sprint>/PERF_GOAL.md and the tail of results.tsv
     (last ~5 rows). Re-anchor.
  2. SendMessage the active worker for a one-line status. Slack-update
     with what it says ("variant X, running paired bench, ~60% done").
  3. If the worker has been silent for 2+ fires:
     - SendMessage one more time.
     - If no response: TaskStop, respawn fresh worker with same variant
       (template in [[perf-sprint]]).
     - If the same variant wedges twice: log a crash row to results.tsv,
       pick a different variant.
  4. If no worker is active (orchestrator missed spawning the next one
     after a return): pick the next variant, spawn the next worker.
  5. CLEANUP: if any prior worker is in returned/idle state, TaskStop it
     by name. Workers don't auto-release on return — completed sessions
     leak over a multi-hour sprint without explicit cleanup.

DON'T GIVE UP. "The bench is unreliable" is never a reason to stop —
spawn a worker to fix the bench. Idle is a bug. A silent /loop fire
with no slack-update is a bug.

Stop only when wall_s_per_decision hits the target OR the user types "stop".
~~~

## Why this shape

The orchestrator's loop is supervisory by design. Workers run in background (`run_in_background=true` in the spawn template) so the orchestrator never blocks. `/loop` fires periodically regardless of worker state and pings via `SendMessage` for status. If a worker hangs, `/loop` catches it; with synchronous `Agent` calls, a hung iteration would freeze the whole sprint.

The metric, gate, and keep/discard mechanic are stated together because they decide together. Every fire re-anchors on what advances the branch and what doesn't.

## Links

[[perf-sprint]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]]
