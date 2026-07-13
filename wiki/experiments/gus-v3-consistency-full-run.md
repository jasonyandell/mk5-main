---
title: Gus v3 Consistency Regularizer — Full 10k Run
kind: experiment
first_seen: 2026-04-21
last_updated: 2026-04-21
status: active
---

## Summary

First sub-1.0 regret result in Gus history. v3 consistency loss at 10k corpus beats v2
plain distillation decisively — gap widens from near-tied at 3k to −33% at 10k. Consistency
loss rides forward into LAMIR-1. (commit messages @ b4040c5, 31f0ec3)

## Setup

- **Corpus**: 10,000 games (lazy IterableDataset streaming; peak RSS 3.4 GB — see [[sources/f138069]])
- **Architecture**: v3 = v2 (shared transformer + belief + V + π_me + world_encoder + Q) +
  [[topics/consistency-regularizer]] loss (`w_consistency=0.3`, warmup 10 epochs)
- **Model**: 3.4M params, d=256, 6 layers (same as the v2-3k-big best)

## Results: v2 vs v3 at 10k

| Adapter | Corpus | bot-match | regret (Q-pts) |
|---|---|---|---|
| v2_voids_10k_big | 10000g | 73.21% | 0.818 |
| **v3_consistency_10k** | **10000g** | **76.07%** | **0.551** |

v3 wins clearly: −33% regret vs v2 at same corpus size. (commit message @ 31f0ec3)

## Decomposition of total regret reduction (v2-3k → v3-10k)

| Factor | Regret change |
|---|---|
| Baseline (v2-3k) | 1.391 |
| Data scaling alone (v2-3k → v2-10k) | −41% → 0.818 |
| Consistency loss on top (v2-10k → v3-10k) | additional −33% → 0.551 |
| **Total v2-3k → v3-10k** | **−60%** |

Consistency loss scales **better** than plain distillation — the gap between v2 and v3
widens as corpus grows. (commit message @ 31f0ec3)

## Scaling: first sub-1.0 regret

0.551 Q-pt mean regret is the first result below 1.0 in Gus history. On a ±42 scale:
~0.66% of Q-range lost per decision.

## qMAE plateau (§18 — 41fdb3c)

qMAE improved only 7% from 3k→10k while regret dropped 59% and V-MAE dropped 20%.
Structural cause: Q_head trains on one random world per forward pass, no cross-world
consistency reward. Two fix candidates:
1. Multi-world variance regularization during Q_head training
2. Joint co-training with belief distribution targets

Neither is blocking LAMIR-1 which doesn't depend on Q_head directly. (commit message @ 41fdb3c)

## Decision

Consistency loss rides forward into LAMIR-1. v3 is the new baseline. (commit message @ 31f0ec3)

## Links

[[gus]] · [[topics/consistency-regularizer]] · [[topics/v-pi-decoupling]] · [[topics/regret-eval]] · [[experiments/gus-scaling-ladder]]
