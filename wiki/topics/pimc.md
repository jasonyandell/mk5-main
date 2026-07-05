---
title: PIMC (Perfect-Information Monte Carlo) variants
kind: topic
first_seen: 5a4c9b9
last_updated: local-2026-07-05
status: active
---

## Overview

PIMC (Perfect-Information Monte Carlo) is a family of inference-time techniques that sample hidden-state worlds, evaluate each via a solver or surrogate Q function, and pick the play with the best averaged Q. [[gus]] evaluated three PIMC variants against direct π_me on 560 held-out decisions (5a4c9b9).

## Three variants evaluated

| Mode | Description | Bot-match |
|---|---|---|
| **direct (π_me argmax)** | Just run the policy head — no look-ahead | **65.4%** |
| **pimc-q** | Q_head on 1 oracle-sampled world | 62.1% |
| **pimc-belief** | Q_head on 50 student-belief-sampled worlds | 61.8% |

(5a4c9b9, 560 decisions, 1000g adapters)

## Finding: direct π_me beats both single-step PIMC variants

Direct π_me is itself the marginalized policy — it was trained on oracle argmax(E[Q]) which already averages over thousands of consistent worlds. Adding 1-50 student-sampled worlds at inference introduces variance without providing new marginalization. Single-step PIMC is redundant given how π_me was trained (5a4c9b9).

Exception: PIMC occasionally beats direct on specific decisions (e.g., dec 3, dec 14, dec 15). These show complementary signal an ensemble or search could exploit (5a4c9b9).

## LAMIR implication

This finding reframes [[lamir1]]'s value proposition. Single-step PIMC does not help because the policy head already has the marginalized answer. LAMIR's value comes from **multi-step look-ahead with mid-tree belief updates** — the model needs to reason about how its own actions change subsequent belief distributions, which requires the π_opp head (not yet trained at this frontier). Deferred to later sessions (5a4c9b9).

Single-step PIMC is therefore not the path to improving Gus's inference-time performance above the policy head ceiling. The next levers are data scale, model capacity, and eventually the π_opp-enabled multi-step variant (5a4c9b9).

## Where the flaw bites: rank vs price

Strategy-fusion optimism is asymmetric in its consequences: it barely disturbs
play (an argmax over siblings, where common-mode inflation cancels) and lands
whole in bidding (a cardinal tail-mass read against an external alternative).
[[rank-vs-price]] develops the mechanism; [[w42-champion-selfplay-fixed-point]]
is the measured over-bidder it explains; [[jud]] is the design that makes
prices honest.

## Links

[[gus]] [[lamir1]] [[joint-world-tensor]] [[expected-q-value]] [[regret-eval]] [[student-distillation]]
