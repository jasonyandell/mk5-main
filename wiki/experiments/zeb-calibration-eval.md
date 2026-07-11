---
title: Zeb Calibration Eval
kind: experiment
first_seen: d9baf3b
last_updated: d9baf3b
status: active
---

## Summary

Calibration evaluation of [[zeb]]'s belief predictions using `burl/eval/belief_calibration.py`. Separates already-played dominoes (trivially known) from hidden dominoes (genuinely uncertain). Result: the advertised 72% top-1 accuracy is inflated; hidden-only top-1 is ~39%. See [[decisions/zeb-parked-eq-primitive]].

([burl/eval/belief_calibration.py @ d9baf3b](../sources/d9baf3b.md))

## Setup

- Evaluated [[zeb]]'s `P(opponent ∈ {L, partner, R} | visible state)` predictions over a held-out set
- Key distinction: already-played dominoes (location trivially known from the play record) vs hidden dominoes (location genuinely uncertain)

## Results

| Metric | Value | Scope |
|---|---|---|
| Top-1 accuracy (all 28 dominoes) | 72% | Inflated by trivially-known played dominoes |
| Top-1 accuracy (hidden-only) | ~39% | The number that matters for Burl |
| Brier score | 0.224 | |
| ECE | 0.067 | |

## Significance

The 72% figure has been the headline Zeb accuracy since the [[forge]] training pipeline. It was measured over all 28 dominoes at each state — including dominoes already on the table, whose locations are trivially determinable from the play record. Including those inflates accuracy substantially.

The number that matters for [[burl]]'s belief tool is the hidden-only figure: ~39%. At that accuracy, Zeb is not reliable enough to anchor decisions. Zeb is parked; E[Q] N=10 outcome distribution becomes the belief primitive. See [[decisions/zeb-parked-eq-primitive]].

## Generalizable lesson

When an ML artifact advertises a headline accuracy, inspect the denominator. "Accuracy on all 28 dominoes" and "accuracy on the hidden set" can differ by 30+ points depending on how many are trivially known at the evaluated state.

## Related pages

[[zeb]] · [[burl]] · [[forge]] · [[decisions/zeb-parked-eq-primitive]] · [[tool-orchestration]] · [[sources/d9baf3b]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- `burl/eval/belief_calibration.py` implements the hidden-vs-played separation as described (hidden_mask excludes own hand plus already-played; hidden-only top-1 computed per state).
- No raw JSON calibration output is checked into `burl/eval/results/` (only perf artifacts), so the 72%/39%/0.224/0.067 figures trace to the [[d9baf3b]] digest rather than being re-derivable in-repo.
- Cheap next probe: check in the calibration run's numeric output (or a one-line CSV) alongside the plots so those figures reproduce from the repo.
