---
title: Detect-and-Route (blunder-gated inference wrapper)
kind: topic
first_seen: eba5103
last_updated: a09ef43
status: active
---

## Overview

Detect-and-route is [[gus]]'s proposed deployable inference architecture: the [[blunder-detector]] runs first on each decision; if the score exceeds a flag threshold, control is routed to a fallback policy; otherwise the primary student's π_me argmax is used (eba5103).

## Routing policies evaluated

Three fallback policies were tested on 560 held-out decisions (eba5103):

| Flag% | Oracle argmax | PIMC-Q-K50 | Next-best adapter |
|---|---|---|---|
| 5% | 1.15 | 1.39 | 1.48 |
| 20% | **0.56** | 1.47 ↑worse | 1.55 ↑worse |
| 25% | **0.49** | 1.48 ↑worse | 1.69 ↑worse |

Baseline (no routing): 1.39 regret.

## Findings

1. **Oracle routing works** — 0.49 regret at 25% flag rate, matching projection. Approaches the 0.5-1.0 Q-pt teacher-noise floor.

2. **PIMC-Q-K50 hurts** — fixes 7 blunders but introduces 6 new blunders on non-blunder decisions the detector incorrectly flags. Q_head trained on one random world per forward pass is too noisy to serve as a reliable fallback. See [[router-reality-check]] (eba5103).

3. **Next-best adapter is worst** — smaller/weaker adapters are also wrong on the same sharp decisions where the primary needs help. Adapter diversity doesn't help in the tail (eba5103).

## Emerging architecture

The deployable shape of Gus v1.0 is: detect → route → fallback. For oracle-budget inference this already works (0.49 regret). For no-oracle inference, the prerequisite is a Q_head that survives multi-world averaging — either via multi-world variance regularization during training or K=50+ worlds at inference (109f9e1, a09ef43).

Router benefit concentrates on mid-game tricks (decisions 0-12, especially 4, 8, 10) where primary regret is 2-6 Q-pts. End-game decisions (24-27) are correctly never flagged (eba5103).

## Relationship to ensembling

Naïve ensemble (majority vote, softmax average) boosts bot-match but hurts regret — it averages away the confident-right adapter on sharp decisions. Oracle-per-decision ceiling: 0.36 regret (74% reduction), with 58% of decisions showing adapter disagreement. A gating model (detect-and-route) is the right shape, not averaging (f90682c).

## Links

[[gus]] [[blunder-detector]] [[regret-eval]] [[router-reality-check]] [[pimc]] [[expected-q-value]]
