---
title: Shine Analysis (where the student is already optimal)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-04-21
status: active
---

## Overview

Shine analysis is the mirror of [[blunder-detector]] forensics. It characterizes the 73% of decisions (per [[regret-eval]]'s bimodal distribution) where the [[gus]] student already achieves zero or near-zero regret. Understanding where the student shines tells you where capacity and data investment should NOT go (7a9c720).

## Key findings (v2_voids_3000g_big, 410 perfect decisions)

**Breakdown of zero-regret decisions by position difficulty:**

| Category | Fraction | Description |
|---|---|---|
| Dead-ties (spread < 0.5) | 59.0% | Any play works; no skill required |
| Moderate (0.5–5 spread) | 15.6% | Some skill; student gets it right |
| Sharp-and-perfect (spread ≥ 5) | **25.4%** | Real skill — 104 decisions where student chose optimally on high-stakes positions |

The sharp-and-perfect bucket spans all 10 declarations — wins are not concentrated in easy declaration types. The student is demonstrating real inference on high-spread positions (7a9c720).

**Distribution flip:**
- PERFECT owns late-game: decisions 24-27 = 100% perfect.
- BLUNDER owns early-mid leads: decisions 0, 4, 8 = 70% of blunders.

## Zero-inference routing heuristic

A cheap pre-filter before the [[blunder-detector]] (7a9c720):

> Trust student when `legal_count ≤ 2` OR `decision_idx ≥ 22`.

- Covers 450/560 decisions at ≤2% blunder rate.
- Only ~110 "wide-choice mid-game" decisions need the full detector.
- 4× reduction in detector workload.

This enables a **two-stage routing** architecture: cheap pre-filter → expensive GBM detector only where needed → fallback. Zero cost on ~80% of decisions.

## Implication for training investment

The dead-tie bucket (59% of perfects) requires no additional signal — the student already handles these trivially. Capacity and data improvements should concentrate on the early-mid blunder tail (decisions 0, 4, 8), which drives nearly all of mean regret. See [[regret-eval]] bimodal distribution and [[v-pi-decoupling]] for the failure mechanism (7a9c720).

## Links

[[gus]] [[regret-eval]] [[blunder-detector]] [[detect-and-route]] [[v-pi-decoupling]]
