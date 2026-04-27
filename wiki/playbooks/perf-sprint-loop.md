---
title: Perf Sprint — Loop Message (verbatim)
kind: playbook
first_seen: aae430b
last_updated: aae430b
status: active
---

The `/loop` message to register at sprint start. Paste verbatim, replacing the bracketed goal token.

## Loop message

~~~
/loop 10m

GOAL: <restate the sprint goal here, e.g. "drive Burl per-decision
latency from ~26s toward ~2s on M5 Max">. Speed is the goal; memory
wins are pleasant side-effects, not the destination.

WRAP IS NOT ALLOWED until one of these is true:
  (a) goal achieved on full-N attribution AND user approves wrap
  (b) every lever in [[perf-sprint-levers]] tried with a ledger row each
  (c) user has typed "stop"

Crashes are NOT a wrap condition. They are work.
  - Read traceback, apply smallest fix, re-run.
  - If 3 fixes and still crashing, switch to next lever (don't stop).
  - "Bench is unreliable" is never a reason to stop measuring speed.
    Find another way to measure speed.

Each fire:
  1. Re-read scratch/PERF_GOAL.md and [[perf-sprint-levers]] to re-anchor.
  2. Status: what's running? what crashed? where's the latest ledger row?
  3. If nothing is running and no wrap-condition is met:
     START THE NEXT LEVER. Idle is a bug. Heartbeat-without-action is a bug.
  4. Slack update: what trying now, what tried last, current best wall.
~~~

## Why this shape

Context churns over 4 hours. The original sprint's loop message just said "check in with slack-style updates and keep things moving along" — accurate but not anchoring. When Phase 4 crashed twice, "wrap and write the digest" felt like discipline rather than rationalization.

Three deliberate features fix that:

1. **Goal restated every fire.** The orchestrator re-anchors on what they're actually here to do, not just on the cadence.
2. **Wrap conditions explicit.** Three predicates, none of which fire on "the bench is hard." This closes the off-ramp.
3. **Idle named as a bug.** Heartbeat without progress was the original failure mode dressed up as patience. Naming it as wrong removes the cover.

## Companion file

The deeper anchor lives in `scratch/PERF_GOAL.md` (template at [[perf-sprint-goal]]). The loop message stays terse; the goal file holds the quality bar, paired protocol, and wrap-conditions reference for re-reading at every fire.

## Links

[[perf-sprint]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]]
