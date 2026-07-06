---
title: "Zeb Parked; E[Q] Distribution as Belief Primitive"
kind: decision
first_seen: d9baf3b
last_updated: d9baf3b
status: superseded
---

## Decision

[[zeb]] is parked as [[burl]]'s belief tool. E[Q] N=10 outcome PDF (from `burl/tools/eq_distribution.py`) takes its place as the belief primitive. Zeb remains in the repo and can be re-enabled behind a flag if a better-calibrated belief model is trained.

## Why

[[experiments/zeb-calibration-eval]] separated Zeb's prediction accuracy into two populations:

- **All 28 dominoes:** 72% top-1 — includes already-played dominoes whose locations are trivially determinable from the play record.
- **Hidden-only (genuinely uncertain dominoes):** ~39% top-1. Brier 0.224, ECE 0.067.

~39% is not reliable enough to anchor decisions. A belief tool that is wrong more than half the time on uncertain dominoes adds noise to Burl's reasoning, not signal.

## Replacement

E[Q] N=10 outcome distribution, counterfactually validated on seed 900013: a deliberate play shift produced Δmean +15 and p_make 0.6 → 1.0. Performance: 290ms per play, 49.8 MB peak VRAM.

This is a direct measure of expected outcome over the hidden-hand distribution — which is exactly what Burl needs to reason about.

## Not deleted — parked

`burl/tools/zeb.py` remains. Its default checkpoint was also fixed from the untrained `large-belief-bootstrap.pt` (std 0.036, essentially random) to `lb-v-eq-3740-bootstrap.pt` (the actual trained checkpoint). Both fixes are available when Zeb is re-enabled.

## Generalizable principle

When an ML artifact advertises a headline number, inspect what denominator the number uses. "Accuracy on all 28 dominoes" and "accuracy on the hidden set" can differ by 30+ points depending on how many are trivially known at the evaluated state. Always report accuracy on the uncertain set.

## Related pages

[[zeb]] · [[burl]] · [[forge]] · [[experiments/zeb-calibration-eval]] · [[tool-orchestration]] · [[sources/d9baf3b]]

## Status

Reconfirmed independently by [[w42-jud-v1|jud v1]]'s 2026-07-06 verdict (`afd4802`),
which names E[Q] n=10 the play champion "against every learned challenger since Zeb."
Superseded as Burl's belief primitive by [[belief-trajectory]] (Gus), and Zeb-as-belief
is itself now superseded architecturally by [[gus]]'s belief/champion engine.
