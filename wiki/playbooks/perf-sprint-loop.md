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

You are the orchestrator. You do not read bench output, source dumps,
or tracebacks. Each iteration is delegated to a fresh Agent — see the
spawn template in [[perf-sprint]].

LOOP FOREVER:
  1. Read scratch/<sprint>/PERF_GOAL.md and the tail of results.tsv
     (last ~5 rows is enough). Re-anchor.
  2. Pick the next variant. Read [[perf-sprint-levers]] for seed ideas;
     freelance is fine. A coherent stack of co-required changes counts
     as one variant.
  3. Spawn an iteration Agent (template in [[perf-sprint]]). Wait for
     the return: one TSV row + exactly 2 sentences.
  4. Append the returned row to scratch/<sprint>/results.tsv. The
     iteration already advanced or reset the branch — you don't.
  5. Slack update: current best wall_s, what was just tried, what's next.

DON'T GIVE UP. "The bench is unreliable" is never a reason to stop —
spawn an iteration to fix the bench. Idle is a bug. Heartbeat-without-
action is a bug.

Stop only when wall_s_per_decision hits the target OR the user types "stop".
~~~

## Why this shape

The orchestrator's loop is thin by design — pick variant, spawn, append, repeat. All the noisy work (modify code, run bench, parse output, handle crashes, decide keep/discard) happens inside the iteration Agent's context, which dies after returning. This is what keeps the orchestrator runnable for hours: it never accumulates the bench output, tracebacks, or source dumps that fill context fastest.

The metric, gate, and keep/discard mechanic are stated together because they decide together. Every fire re-anchors on what advances the branch and what doesn't.

## Links

[[perf-sprint]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]]
