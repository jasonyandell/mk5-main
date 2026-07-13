---
title: Blunder Detector (student-feature blunder classifier)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-07-13
status: retired
---

## Overview

The blunder detector is a small `GradientBoostingClassifier` that predicts, from
decision-level features, whether [[gus]]'s student will blunder (regret > threshold,
typically 8 Q-pts) on a given decision. It gates the [[detect-and-route]] fallback
policy: run the fast student everywhere, escalate flagged decisions to a stronger
evaluator (f90682c, 5373223).

## The concept in one line

The student's own uncertainty is a usable blunder signal — `pi_peak` (policy
confidence) is the top feature in the deployable version — so a detector needs no
oracle features at inference, only oracle labels at training time.

## Where the results live

- [[gus-blunder-detector]] — both versions with full numbers: v1 oracle-feature
  (ROC-AUC 0.926, the ceiling), v2 student-feature (ROC-AUC 0.839, the deployable
  candidate; 1.13 → 0.49 regret at 20% flag with oracle fallback), plus the
  ensembles-hurt/router-wins receipt.
- [[gus-router-pilot]] — end-to-end detect-and-route validation: oracle fallback works,
  every non-oracle fallback tested there hurts.
- [[gus-shine-analysis]] — the zero-inference pre-filter that removes ~80% of decisions
  from the detector's workload.
- [[gus-qmean-router]] — the later no-oracle router that works, using disagreement
  shape between direct π and belief-sampled Q-mean rather than pure confidence.

## Retired

Never wired into champion, arena, or forge (`grep -rl blunder_detector` across the repo
returns no hits outside `gus/eval/`). The [[detect-and-route]] architecture it was built
to gate was itself abandoned when the project pivoted to [[jud]] (self-play,
no CFR+) rather than a distilled-value look-ahead/fallback stack. Kept as a
source-backed record of the blunder-rate analysis, not a live component.

## Links

[[gus]] [[gus-line]] [[regret-eval]] [[detect-and-route]] [[v-pi-decoupling]] [[jud]]
