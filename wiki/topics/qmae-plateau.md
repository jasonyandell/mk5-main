---
title: qMAE Plateau (Q-head scaling wall)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-04-22
status: active
---

## Overview

qMAE (Q-head mean absolute error) measures how accurately [[gus]]'s Q_head predicts per-world Q values from fused `(state, world)` input — the [[lamir1]]-critical piece. qMAE plateaus at scale: 3.3× more data improved qMAE only 7%, while [[regret-eval]] improved 59% and V-MAE improved 20% (PRACTICALITIES §18, 41fdb3c).

## The plateau numbers (3k → 10k)

| Metric | 3k games | 10k games | Delta |
|---|---|---|---|
| Regret | 1.39 | 0.551 (v3) | −60% |
| V-MAE | ~10 | ~8 | −20% |
| qMAE | ~12.3 | ~11.4 | −7% |

qMAE barely moved while the metrics that matter for play quality improved sharply. The Q_head is a weak link (41fdb3c).

## Structural cause

Q_head trains on one random world per forward pass — no cross-world consistency reward. Each training step shows the Q_head a `(state, world)` pair and asks it to predict Q for that world. The Q_head never sees how its predictions vary across worlds for the same state, so it cannot learn to be consistent across worlds even if it learns to be accurate on average (41fdb3c).

This is the same issue that caused the [[blunder-detector]]'s Q-spread feature to underperform as a blunder proxy — the per-world Q estimates are individually accurate but collectively noisy (5373223, 41fdb3c).

## Known fix paths — one tried and falsified, one never attempted

Two candidate remedies were proposed (41fdb3c):

1. **Multi-world variance regularization during Q_head training**: add a loss that penalizes high variance in Q_head predictions across worlds for the same state when the oracle says the Q values should be similar. Affects `train_v2_voids.py` and descendants. **Never attempted.**

2. **Joint co-training**: train Q_head simultaneously on multiple worlds per decision rather than one. Remedy #2's cousin was tried the same week as [[belief-co-train]] (`cf8ff79`) and **falsified on this setup**: "the hypothesis... is falsified on this setup" — joint co-training of belief + world_encoder + Q_head improved belief calibration but made downstream q-bootstrap regret slightly worse (0.685 → 0.718), because Q_head had been trained against the old belief head's output shape and joint retraining moved it off that sweet spot.

## LAMIR-1 dependency

LAMIR-1 requires the Q_head to produce reliable per-world Q estimates for belief-reweighted aggregation. The current Q_head's per-world noise degrades PIMC quality (see [[pimc]]) and will degrade LAMIR-1 look-ahead unless fixed. However, the fix is not blocking the LAMIR-1 track — [[lamir1]] can be explored with the current Q_head quality and improved iteratively (41fdb3c).

## Links

[[gus]] [[lamir1]] [[dense-q-supervision]] [[pimc]] [[blunder-detector]] [[consistency-regularizer]] [[regret-eval]] [[belief-co-train]]
