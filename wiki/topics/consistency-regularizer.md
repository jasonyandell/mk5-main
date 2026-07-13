---
title: V/π Consistency Regularizer (v3)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-04-21
status: active
---

## Overview

The V/π consistency regularizer is a training objective added in [[gus]] v3 to close the [[v-pi-decoupling]] gap — forcing the policy head to act consistently with what the value head knows (b007cf3).

## Formula

```
L_consistency = (V_head.detach() - Σ_legal softmax(π_me) · e_q)²
```

The right-hand sum is the policy-expected Q: for each legal action, weight its Q-value by π_me's probability and sum. This is the Q the student would achieve if acting from its current policy.

V_head is **detached** — gradient flows only into π_me, not back into V_head. The effect: π concentrates on actions whose Q-value agrees with V's position estimate. V itself is not perturbed (b007cf3).

## Warmup

The consistency weight ramps from 0 to `w_consistency` (default 0.3) over the first 10 epochs. This lets V_head converge on its primary target (oracle best-legal E[Q]) before consistency pressure reshapes π_me toward it. Without warmup, V and π are trained simultaneously from a cold start, and a poor early V estimate could misguide π (b007cf3).

## Motivation

[[v-pi-decoupling]] (1a2f67f): blunder forensics showed V_head outputting +26 while π_me picks a −0.4 play. The blunder tail (6% of decisions, per f0139a3) drives nearly all of mean regret. A direct coupling loss is the targeted intervention (b007cf3).

## Training script

`gus/train/train_v3_consistency.py` — same arguments as `train_v2_voids` plus:
- `--w-consistency` (default 0.3)
- `--consistency-warmup-epochs` (default 10)

Smoke-tested (loss integrates cleanly, finite gradients) (b007cf3).

## Result at 10k scale: wins (31f0ec3)

v3 (consistency) vs v2 (plain distillation) at 10k games:

| Adapter | Regret | Bot-match |
|---|---|---|
| v2_voids_3000g (baseline) | 1.39 | 66.4% |
| v3_consistency_3000g | ~1.28 | — |
| v2_voids_10000g_big | 0.818 | 73.21% |
| **v3_consistency_10000g** | **0.551** | **76.07%** |

v3 is the first Gus adapter under 1.0 regret. Consistency loss **scales better than plain distillation** — the gap between v2 and v3 widens from roughly tied at 3k to decisive (−33%) at 10k.

Decomposition of total regret reduction from v2-3k to v3-10k (−60%):
- Data scaling alone: −41%
- Consistency loss on top: additional −33%

Decision: consistency loss rides forward into LAMIR-1 training (31f0ec3) — v3-10k became the
baseline adapter for every LAMIR-1 rollout mode evaluated. LAMIR-1 itself was later measured
as a dead end (every rollout mode lost to direct π_me; see [[lamir1-ceiling]]), so this
decision correctly carried the best base model forward — it just carried it into a track that
didn't pan out. The 0.551-regret / 76.07%-bot-match number remains the best-known single
adapter as of 2026-07-06, cited unchanged in [[champion]]'s asset map.

## Links

[[gus]] [[v-pi-decoupling]] [[dense-q-supervision]] [[regret-eval]] [[lamir1-ceiling]] [[champion]]
