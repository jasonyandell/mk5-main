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
latency from ~26s toward ~2s on M5 Max">. Speed is the goal; memory
wins are pleasant side-effects, not the destination.

DON'T GIVE UP. Crashes are work, not a stop sign.
  - Read the traceback, apply the smallest fix, re-run.
  - If 3 fixes on one lever still crash, switch to the next lever.
  - "The bench is unreliable" is never a reason to stop measuring
    speed. Find another way to measure speed.

Stop only when the goal is hit or the user types "stop".

Each fire:
  1. Re-read scratch/PERF_GOAL.md and [[perf-sprint-levers]] to re-anchor.
  2. Status: what's running? what crashed? where's the latest ledger row?
  3. If nothing is running and the goal isn't hit:
     START THE NEXT LEVER. Idle is a bug. Heartbeat-without-action is a bug.
  4. Slack update: what trying now, what tried last, current best wall.
~~~

## Why this shape

Context churns over hours. The loop message has to do two things at every fire:

1. **Restate the goal** so the orchestrator re-anchors on what they're here to do.
2. **Name idle as a bug.** Heartbeat without progress is the failure mode dressed up as patience.

The "don't give up" clause lives here and in [[perf-sprint-goal]] — two places, same voice. Nowhere else.

## Links

[[perf-sprint]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]]
