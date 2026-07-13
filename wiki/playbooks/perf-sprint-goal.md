---
title: Perf Sprint — Goal Anchor Template
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Template for `scratch/<sprint>/PERF_GOAL.md` — the sticky goal file written at sprint kickoff. Re-read at every loop fire.

## Template

~~~
# Perf Sprint — Goal

TARGET: wall_s_per_decision <baseline> → <target> on <hardware>.
       (e.g. 26s → 2s on M5 Max. 13×.)
       Anything substantial is a win. The full ratio is a stretch.

CONTRACT METRIC: wall_s_per_decision on perf_subset_5,
                paired baseline immediately before each variant.
                End-to-end (tokens + tools + harness).

EQUIVALENCE GATE (binary, must pass):
  - K1_match >= 60%
  - regret_delta within ±10%
  - measured on the same paired run as wall_s

LEDGER: scratch/<sprint>/results.tsv
        columns: commit  wall_s  k1_match  regret_delta  peak_gb  status  description
~~~

## Why a separate file

The loop message restates the goal at every fire but is short. The goal file is the contract: target, metric, gate, ledger location. Re-readable in 30 seconds to re-anchor.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]]
