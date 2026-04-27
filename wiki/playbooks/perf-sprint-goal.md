---
title: Perf Sprint — Goal Anchor Template
kind: playbook
first_seen: aae430b
last_updated: aae430b
status: active
---

Template for `scratch/PERF_GOAL.md` — the sticky goal file the orchestrator writes at sprint kickoff and re-reads at every loop fire. Lives in `scratch/` because it's session-specific; the playbook lives in the wiki because it's reusable.

## Template

~~~
# Perf Sprint — Goal

TARGET: <baseline> → <target> on <hardware>. <ratio>×.
       (e.g. 26s/decision → 2s/decision on M5 Max. 13×.)
       Anything substantial is a win. The full ratio is a stretch.

ANCHOR METRIC: <e.g. wall_s_total on perf_subset_5; or wall_s on full-N>

QUALITY BAR:
  - K1 grade match >= 60% (the noise floor on small subsets)
  - regret_delta within ±10% on the comparison anchor
  - paired protocol required: fresh baseline IMMEDIATELY before each variant
  - never compare against ledger's "latest baseline" without verifying same flags

NEVER GIVE UP. The user is recharging. You are the team. The goal stays
front-of-mind every loop fire — that's what this file is for.

WRAP CONDITIONS (read [[perf-sprint]] for the full list):
  (a) goal achieved + user approves
  (b) every lever in [[perf-sprint-levers]] has a ledger row
  (c) user typed "stop"
~~~

## Why this exists separately from the loop message

The loop message restates the goal but is short — it has to fit in a brief slack-style ping. The goal file holds the deeper anchoring (quality bar, paired protocol, the wrap-conditions reference) that the orchestrator re-reads in 30 seconds at every fire to stay grounded.

Loop message = the bumper sticker. Goal file = the contract.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]]
