---
title: Detect-and-Route (blunder-gated inference wrapper)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-04-21
status: retired
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

## Proposed architecture (never shipped)

This was proposed as the deployable shape of "Gus v1.0": detect → route → fallback. No such
release exists — `git tag` and `git log --grep` show no "Gus v1.0" artifact. The oracle-budget
path reached 0.49 regret in eval scripts only; it was never wired into champion, arena, or
forge production. For no-oracle inference, the prerequisite would have been a Q_head that
survives multi-world averaging — either via multi-world variance regularization during
training or K=50+ worlds at inference (109f9e1, a09ef43) — but the whole line was abandoned
when the project pivoted to `jud`/[[champion]] instead of pursuing further LAMIR-era fixes.

Router benefit concentrates on mid-game tricks (decisions 0-12, especially 4, 8, 10) where primary regret is 2-6 Q-pts. End-game decisions (24-27) are correctly never flagged (eba5103).

## Relationship to ensembling

Naïve ensemble (majority vote, softmax average) boosts bot-match but hurts regret — it averages away the confident-right adapter on sharp decisions. Oracle-per-decision ceiling: 0.36 regret (74% reduction), with 58% of decisions showing adapter disagreement. A gating model (detect-and-route) is the right shape, not averaging (f90682c).

## Links

[[gus]] [[blunder-detector]] [[regret-eval]] [[router-reality-check]] [[pimc]] [[expected-q-value]] [[champion]]
