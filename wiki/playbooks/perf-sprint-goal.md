---
title: Perf Sprint — Goal Anchor Template
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Template for `scratch/PERF_GOAL.md` — the sticky goal file written at sprint kickoff and re-read at every loop fire. Lives in `scratch/` because it's session-specific; the playbook lives in the wiki because it's reusable.

## Template

~~~
# Perf Sprint — Goal

TARGET: <baseline> → <target> on <hardware>. <ratio>×.
       (e.g. 26s/decision → 2s/decision on M5 Max. 13×.)
       Anything substantial is a win. The full ratio is a stretch.

ANCHOR METRIC: <e.g. wall_s on full-N; or wall_s_total on perf_subset_5
               with paired baseline immediately before each variant>

EQUIVALENCE BAR (the faster version must still be the same model):
  - K1 grade match >= 60%
  - regret_delta within ±10% on the comparison anchor
  - paired protocol: fresh baseline immediately before each variant

DON'T GIVE UP. Crashes are work, not a stop sign. Stop only when the
goal is hit or the user types "stop".
~~~

## Why a separate file

The loop message is short — a slack-style bumper sticker. The goal file is the contract: target, anchor, equivalence bar, the don't-give-up clause. The orchestrator re-reads it in 30 seconds at every fire to stay grounded.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]]
