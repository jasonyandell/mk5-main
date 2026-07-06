---
title: V/π Head Decoupling
kind: topic
first_seen: 1a2f67f
last_updated: b007cf3
status: active
---

## Overview

V/π head decoupling is a failure mode in [[gus]]'s multi-head architecture: the V_head correctly predicts the oracle's best-legal E[Q] while π_me concentrates probability on a losing action. The two heads are not jointly optimizing — they learned separately and can give contradictory signals (1a2f67f).

## The observation

Blunder forensics from the arena pilot (1a2f67f): on a specific blunder decision, V_head outputs +26 (correctly predicting the best-legal E[Q]), while π_me assigns high probability to a play worth −0.4. The student "knows" the position is good but acts badly.

## Why it happens

In v0–v2, V_head and π_me are coupled only indirectly — through the shared state encoder. The training objectives do not directly enforce consistency between them. V can converge on an accurate position-value estimate while π_me independently converges on a policy that may not be consistent with that value (1a2f67f).

## Game-level consequence

The arena pilot (20 games, seeds 900020-900021, v2_voids_3000g_big):
- Student made 10/20 contracts (50%) vs all-bot baseline 16/20 (80%).
- avg bidder points: 21.2 (student) vs 29.6 (bot), −8.4/hand.

1.39 Q-point mean [[regret-eval]] compounds to a ~30 percentage-point contract-made drop at the game level. Blunders on individually rare decisions accumulate across 28 plays (1a2f67f).

## Remediation

[[consistency-regularizer]] in v3 adds a direct coupling loss that forces policy-expected-Q to match V_head's prediction, with V detached so gradient only reshapes π (b007cf3).

## Links

[[gus]] [[consistency-regularizer]] [[regret-eval]] [[experiments/gus-arena-pilot]] [[student-distillation]]
