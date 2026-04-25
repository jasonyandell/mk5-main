---
title: Router Reality-Check (detect-and-route PoC results)
kind: topic
first_seen: a09ef43
last_updated: a09ef43
status: active
---

## Overview

The router reality-check (PRACTICALITIES receipt 14, a09ef43) documents the honest outcome of deploying [[detect-and-route]] in end-to-end validation on 560 held-out decisions. It is not shelved — the oracle-fallback path works as projected — but it clarifies which routing strategies don't and why (a09ef43).

## What worked

Oracle-argmax fallback at 20-25% flag rate achieves 0.49-0.56 [[regret-eval]] (vs 1.39 baseline). This matches the projected gain from the blunder detector analysis (eba5103, a09ef43).

## What didn't work

**PIMC-Q-K50 fallback (regret → 1.47 at 20% flag):** Q_head trained on one random world per forward pass is too noisy to serve as a reliable mid-game alternative. It fixes some flagged blunders but introduces new blunders on non-blunder decisions the detector incorrectly flags (eba5103).

**Next-best-adapter fallback (regret → 1.55 at 20% flag):** smaller and weaker adapters are wrong on the same hard decisions where the primary fails. No complementary coverage at the tail (eba5103).

## Practical implication

To ship a no-oracle-inference student at 0.49 regret:

**Prerequisite**: a Q_head that survives multi-world averaging. Either:
1. Multi-world variance regularization during Q_head training (affects `train_v2_voids.py`), or
2. K=50+ worlds at inference (cheap, already validated in the blunder-detector K=20 path).

Until then, detect-and-route requires oracle calls. The oracle-budget version is already deployable (a09ef43).

## Numbers summary

| Routing strategy | Flag% | Regret | vs baseline |
|---|---|---|---|
| None (primary only) | — | 1.39 | — |
| Oracle argmax | 20% | 0.56 | −60% |
| Oracle argmax | 25% | 0.49 | −65% |
| PIMC-Q-K50 | 20% | 1.47 | +6% worse |
| Next-best adapter | 20% | 1.55 | +12% worse |

(a09ef43, 560 held-out decisions)

## Links

[[gus]] [[detect-and-route]] [[blunder-detector]] [[regret-eval]] [[pimc]]
