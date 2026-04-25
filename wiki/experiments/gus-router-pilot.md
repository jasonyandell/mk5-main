---
title: Gus Router Pilot (detect-and-route inference)
kind: experiment
first_seen: eba5103
last_updated: a09ef43
status: active
---

## Summary

End-to-end validation of the detect-and-route inference wrapper on 560 held-out decisions.
Oracle routing works as projected; PIMC-Q and next-best-adapter fallbacks both hurt.
Receipt 14 in PRACTICALITIES.md is the money-shot. (commit messages @ eba5103, a09ef43)

## Setup

- **Baseline**: `v2_voids_3000g_big`, 560 held-out decisions, 1.39 Q-pt mean regret
- **Detector**: student-feature GBM (trained on 30% of train chunks 0-299)
- **Fallback policies tested**:
  1. Oracle argmax
  2. PIMC via student Q_head, K=50 sampled worlds
  3. Next-best-adapter (`v1_full_1000g`, smaller/weaker)

## Results (Receipt 14)

| flag% | oracle | pimc-q (K=50) | next-best-adapter |
|---|---|---|---|
| 5% | 1.15 | 1.39 | 1.48 |
| 20% | **0.56** | 1.47 | 1.55 |
| 25% | **0.49** | 1.48 | 1.69 |

## Finding 1 — Oracle routing works

0.49 regret at 25% flag matches the projection from [[experiments/gus-blunder-detector]].
Routing to oracle on flagged decisions is effective and reaches near-teacher-noise floor.

## Finding 2 — PIMC-Q-K50 hurts (receipt 14 money-shot)

PIMC-Q at 20% flag: regret 1.39 → 1.47 (WORSE). Fixes 7 blunders but introduces 6 new
errors on non-blunder decisions the detector incorrectly flags. Q_head was trained on one
random world per forward pass — its per-world Q estimates are too noisy to safely substitute
for oracle at inference. Increasing K to 50 does not fix the underlying signal quality.
(commit message @ eba5103)

## Finding 3 — Next-best-adapter is worst

Smaller adapters are wrong on the same sharp decisions where the primary adapter needs help.
Routing to a weaker adapter concentrates errors rather than diversifying them.
(commit message @ eba5103)

## Implication

Detect-and-route needs oracle calls to ship without quality regression. Without oracle
budget, the prerequisite experiment is Q_head multi-world variance regularization —
either during training (`train_v2_voids`) or by running K=50+ at inference with the
current architecture. (commit message @ a09ef43)

Router benefit concentrates on mid-game tricks (decisions 0-12, especially 4, 8, 10)
where primary regret is 2-6 Q-pts. End-game decisions (24-27) are correctly never flagged.

## Links

[[gus]] · [[topics/regret-eval]] · [[experiments/gus-blunder-detector]] · [[experiments/gus-scaling-ladder]]
