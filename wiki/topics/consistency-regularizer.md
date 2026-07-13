---
title: V/π Consistency Regularizer (v3)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-07-13
status: complete
---

## Overview

The V/π consistency regularizer is a training objective added in [[gus]] v3 to close the
[[v-pi-decoupling]] gap — forcing the policy head to act consistently with what the value
head knows (b007cf3).

## Formula

```
L_consistency = (V_head.detach() - Σ_legal softmax(π_me) · e_q)²
```

The right-hand sum is the policy-expected Q: for each legal action, weight its Q-value by
π_me's probability and sum. This is the Q the student would achieve if acting from its
current policy.

V_head is **detached** — gradient flows only into π_me, not back into V_head. The effect:
π concentrates on actions whose Q-value agrees with V's position estimate. V itself is not
perturbed (b007cf3).

## Warmup

The consistency weight ramps from 0 to `w_consistency` (default 0.3) over the first 10
epochs. This lets V_head converge on its primary target (oracle best-legal E[Q]) before
consistency pressure reshapes π_me toward it. Without warmup, V and π are trained
simultaneously from a cold start, and a poor early V estimate could misguide π (b007cf3).

## Motivation

[[v-pi-decoupling]] (1a2f67f): blunder forensics showed V_head outputting +26 while π_me
picks a −0.4 play. The blunder tail (6% of decisions, per f0139a3) drives nearly all of
mean regret. A direct coupling loss is the targeted intervention (b007cf3).

Training script: `gus/train/train_v3_consistency.py` — same arguments as `train_v2_voids`
plus `--w-consistency` (default 0.3) and `--consistency-warmup-epochs` (default 10).

## Result: wins, and scales

Full numbers on [[gus-v3-consistency-full-run]] and [[gus-scaling-ladder]]. Headline:
v3 vs v2 is a wash at 3k games but decisive at 10k (0.551 vs 0.818 regret, −33%) —
**consistency loss scales better than plain distillation**, and v3-10k is the first Gus
adapter under 1.0 regret.

Decision: consistency loss rode forward into LAMIR-1 training (31f0ec3) — v3-10k became
the baseline adapter for every LAMIR-1 rollout mode evaluated. LAMIR-1 itself was later
measured as a dead end (every rollout mode lost to direct π_me; see [[lamir1-ceiling]]),
so this decision correctly carried the best base model forward — it just carried it into
a track that didn't pan out. The 0.551-regret / 76.07%-bot-match number remains the
best-known single adapter, cited unchanged in [[jud]]'s asset map.

## Links

[[gus]] [[gus-line]] [[v-pi-decoupling]] [[dense-q-supervision]] [[regret-eval]] [[gus-v3-consistency-full-run]] [[lamir1-ceiling]] [[jud]]
