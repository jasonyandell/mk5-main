---
title: PIMC (Perfect-Information Monte Carlo) variants
kind: topic
first_seen: 5a4c9b9
last_updated: local-2026-07-06
status: active
---

## Overview

PIMC (Perfect-Information Monte Carlo) is a family of inference-time techniques that sample hidden-state worlds, evaluate each via a solver or surrogate Q function, and pick the play with the best averaged Q. [[gus]] evaluated three PIMC variants against direct π_me on 560 held-out decisions (5a4c9b9).

PIMC's lineage in this project predates Gus by months: [[web-game]] ruled out AlphaZero, CFR, and neural-nets-from-scratch on deployment-constraint grounds in September 2025 and shipped PIMC-minimax as the working game AI in December 2025 (`3e063ff`) — see [[pre-ml-ai-attempts]]. PIMC-over-full-state, chosen there for refusing to abstract, is the substrate this page's variants run on top of.

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

This finding reframes [[lamir1]]'s value proposition. Single-step PIMC does not help because the policy head already has the marginalized answer. LAMIR's value was hypothesized to come from **multi-step look-ahead with mid-tree belief updates**, requiring the π_opp head (5a4c9b9).

π_opp was built and tried (the `lamir1-piopp` mode); it did not help — regret 2.268, worse
than `lamir1-qleaf`'s 2.006 (rotated-π_me opponent proxy). Single-step PIMC was not the path to
improving Gus's inference-time performance above the policy head ceiling, and neither was the
multi-step π_opp-enabled variant: every LAMIR-1 rollout mode lost to direct π_me. See
[[lamir1-ceiling]] for the full ladder and the pivot that followed (option 4: self-play,
no CFR+ — `jud`/[[champion]]).

## Where the flaw bites: rank vs price

Strategy-fusion optimism is asymmetric in its consequences: it barely disturbs
play (an argmax over siblings, where common-mode inflation cancels) and lands
whole in bidding (a cardinal tail-mass read against an external alternative).
[[rank-vs-price]] develops the mechanism; [[w42-champion-selfplay-fixed-point]]
is the measured over-bidder it explains; [[jud]] is the design that makes
prices honest. [[strategy-fusion]] is the formal result this optimism is named
after — E[max(score)] ≥ max(E[score]) — first named against this project's own
oracle in January 2026.

## Links

[[gus]] [[lamir1]] [[lamir1-ceiling]] [[joint-world-tensor]] [[expected-q-value]] [[regret-eval]] [[student-distillation]] [[web-game]] [[pre-ml-ai-attempts]] [[strategy-fusion]] [[champion]]
