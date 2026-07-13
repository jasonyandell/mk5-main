---
title: Dense Q Supervision (3400× per-decision signal)
kind: topic
first_seen: 2026-04-20
last_updated: 2026-04-21
status: active
---

## Overview

Dense Q supervision is the key training-regime insight from [[gus]] v0→v1→v2 scaffolding. Adding a world-conditioned Q head to the [[student-distillation]] student produces ~3400× more gradient signal per decision than belief-only training, and this density acts as a strong regularizer on the shared state encoder (da21f52).

## The signal-density calculation

Per decision:
- ~100 sampled worlds × ~34 legal plays = ~3,400 Q labels from the [[joint-world-tensor]]
- vs 1 belief target (P(domino ∈ seat) per unseen domino — sparse and noisy at early game positions)

The Q head on fused `(state, world)` input — the [[lamir1]]-critical piece — generates this label density as a byproduct of the architecture. Each decision that costs one oracle call yields thousands of supervised Q targets (da21f52).

## Effect on training dynamics

With belief-only supervision (v1 transformer): train accuracy reaches 74%; eval 37.5%; per-decision progression shows the right shape (decision 0 at chance floor, late-game up to 75%) but overall eval is data-limited (8dbf7f3).

With 4-head supervision including Q (v1 student, 100g corpus): train and eval π_me track within 1-2 points. The overfit pathology that plagued belief-only training disappears (da21f52):

| Metric | 100g corpus, 4-head |
|---|---|
| π_me bot-match (held-out) | 57.9% |
| belief top-1 (held-out) | 34.5% |
| V MAE | 11.9 |
| Q MAE (legal) | 18.4 |

Per-decision π_me: 20% at decision 0 (first play, no info) → 60% at decision 13 (mid-game) → 100% at decisions 24-27 (end-game). Learning is concentrated where information exists (da21f52).

## Relationship to LAMIR-1

The Q head serves two roles simultaneously:
1. **Inference-time:** enables [[lamir1]] look-ahead without oracle calls — sample worlds from belief head, evaluate each via Q head, re-weight by belief posterior.
2. **Training-time:** regularizes the shared encoder via dense gradient signal, improving belief, V, and π_me heads as a side effect.

The multi-head architecture's value is therefore both inference-time composition AND training-time regularization (da21f52).

## Implication for Gus's training recipe

Never train a single-head model when multi-head oracle labels are available. Even if only π_me is needed at deployment, training with belief+V+Q supervises the encoder better. This is a generalizable principle for [[student-distillation]] from clean oracle labels (da21f52).

**Boundary (2026-07-13):** the principle did not transfer to the [[jud]]
architecture. [[jud-target-granularity]] added a dense per-legal-action E[Q]
auxiliary to the jud MLP at fixed capacity: the aux head itself learned a
ranking, but the trunk-shaping moved the primary head *nothing* — ranking
0.042 vs 0.046, greedy marks paired Δ `-0.086 [-0.254,+0.092]` — and the aux
head collapsed when the corpus tripled. Dense auxiliary supervision
regularizing the shared encoder is a Gus-transformer observation, not a law;
in a small MLP whose primary target is hand-level Monte Carlo, it bought
nothing measurable.

## V2: explicit void features (3c02d10)

V2 adds a `VoidsEncoder` that projects a [24]-dim (3 opponents × 8 suits) void indicator vector into the state embedding. The void signals are engine-computed (no inference), making them ground-truth features.

Empirical delta on 1,000g corpus (d=192, 4 layers, 40 epochs):

| Metric | v1 | v2 | Delta |
|---|---|---|---|
| π_me | 66.1% | 65.2% | ~flat |
| belief | 37.2% | 38.6% | +1.4pp |
| V MAE | 7.6 | 7.7 | flat |
| Q MAE | 12.3 | 12.2 | flat |

The transformer was already inferring voids attentionally from play tokens; explicit features close a small gap but are not a large unlock. The next lever is data scale and model capacity (3c02d10).

## Links

[[gus]] [[lamir1]] [[student-distillation]] [[joint-world-tensor]] [[expected-q-value]]
