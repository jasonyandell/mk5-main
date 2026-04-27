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

LOOP FOREVER:
  1. Pick the next variant. Read [[perf-sprint-levers]] for seed ideas;
     freelance is fine. Read [[perf-sprint-traps]] for known pitfalls.
  2. Modify the editable surface (see [[perf-sprint]]). git commit.
  3. Run paired baseline + variant on perf_subset_5.
  4. Read wall_s, k1_match, regret_delta, peak_gb from the run.
  5. Decide:
       gate failed OR wall_s didn't improve  → status = discard, git reset.
       gate passed AND wall_s improved       → status = keep, advance branch.
       run crashed                            → read traceback, smallest fix,
                                                 re-run. After 3 crashes on the
                                                 same idea, status = crash, log,
                                                 move to a new idea.
  6. Append row to scratch/<sprint>/results.tsv.
  7. Slack update: what tried, current best wall_s, what's next.

DON'T GIVE UP. "The bench is unreliable" is never a reason to stop
measuring speed — find another way. Idle is a bug. Heartbeat-without-
action is a bug.

Stop only when wall_s_per_decision hits the target OR the user types "stop".
~~~

## Why this shape

The loop is where the stop decision fires, so the rule lives here. The metric, the gate, and the keep/discard mechanic are stated together because they decide together. Every fire re-anchors on what advances the branch and what doesn't.

## Links

[[perf-sprint]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]]
