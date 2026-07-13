---
title: Student Distillation (E[Q] oracle → neural multi-head)
kind: topic
first_seen: 2026-04-20
last_updated: 2026-04-20
status: active
---

## Overview

Student distillation is [[gus]]'s training paradigm. The [[forge]] E[Q] oracle produces per-decision labels — expected value averaged over a sampled hidden-state distribution — and a multi-head neural student is trained to reproduce them. This bypasses the reasoning channel entirely: Gus produces a decision at inference, not a rationale (42a7535).

## Contrast with LEM and Burl

| Paradigm | Training signal | Inference output | Reasoning channel |
|---|---|---|---|
| [[lem]] (STaR) | Self-generated traces, K1 grade | Play + rationale | Central |
| [[burl]] (tool orchestration) | Tool-trajectory STaR | Play (via tool calls) | Central |
| Gus (student distillation) | E[Q] oracle labels | Play decision | Skipped |

The kickoff doc planned LEM as the downstream commentary/explanation head once Gus's player worked — Gus provides the decision, LEM provides the explanation (42a7535). That pairing was never executed: no wiki page shows [[lem]] ever consuming a Gus decision, and LEM instead pursued its own independent STaR curriculum (see [[lem]]). The harl→lem narrator framing from the project's earliest planning conversations (pre-rename) carried the same plan forward and was likewise never built. Gus and LEM remain separate sibling projects, not a decision/narration pipeline.

## Why variance-free

The E[Q] framework computes expected value by running the full-information solver over a distribution of hidden hands and averaging the results. This produces a clean supervised label per decision: no reward variance, no bootstrapping from noisy game outcomes, no credit-assignment problem. The learning problem is supervised regression, not RL (31e10ef).

## Multi-head architecture

Five heads share a transformer backbone:

| Head | What it learns |
|---|---|
| Belief | P(domino ∈ seat) over unseen dominoes |
| V | Expected game value from current visible state |
| π_me | Policy — action probabilities for the player's seat |
| Q | World-conditioned Q-value: Q(state, world) — the LAMIR-critical piece |
| π_opp | Opponent policy (deferred in v0) |

The Q head on fused (state, world) input enables [[lamir1]] look-ahead at inference without oracle calls: sample worlds from the belief head, evaluate each via the Q head, re-weight by belief posterior (31e10ef).

## Literature grounding

2024-2025 consensus supporting gradient methods + neural functions over explicit CFR in imperfect-information games (from the Gus kickoff, 42a7535):

- PG beats CFR (ICLR 2025)
- [[lamir1]] (LAMIR; Oct 2025)
- Bridge AI (BMCS)
- PRR-TM teammate modeling

The regime: when the value signal is clean (oracle labels), supervised + neural approaches outperform explicit counterfactual-regret minimization (42a7535).

## Links

[[gus]] [[forge]] [[expected-q-value]] [[lamir1]] [[lem]] [[burl]] [[joint-world-tensor]]
