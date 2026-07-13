---
title: "LAMIR-primitive inference eval: direct vs PIMC-Q vs PIMC-belief"
kind: experiment
first_seen: 2026-04-21
last_updated: 2026-07-13
status: complete
---

## Summary

Three inference modes compared on 560 held-out decisions using the 1000g adapter.
Direct π_me wins; single-step PIMC adds variance without new information. (commit message @ 5a4c9b9)

## Setup

- **Corpus**: 1000g adapter, 560 held-out decisions
- **Modes tested**:
  - `direct`: π_me argmax (the trained policy head)
  - `pimc-q`: Q_head evaluated on 1 oracle-sampled world, argmax over result
  - `pimc-belief`: Q_head evaluated on 50 belief-sampled worlds, argmax over mean

## Results

| Mode | Bot-match |
|---|---|
| direct π_me | **65.4%** |
| pimc-q (K=1, oracle-sampled) | 62.1% |
| pimc-belief (K=50, belief-sampled) | 61.8% |

## Finding

Direct π_me beats both PIMC variants. π_me was trained on oracle's `argmax(E[Q])` which
already averages over thousands of consistent worlds — adding 1-50 student-sampled worlds
introduces variance without contributing new information at the single-step level.
(commit message @ 5a4c9b9)

## Per-decision hints of complementary signal

PIMC occasionally wins on specific decisions:

| Decision | direct | pimc-q | pimc-belief |
|---|---|---|---|
| dec 3 | 90% | 95% | 95% |
| dec 14 | 45% | 55% | 65% |
| dec 15 | 50% | 70% | 75% |

These hints suggest an ensemble or search could exploit complementary signal — but the
aggregate effect is net-negative at K=1 and K=50.

## LAMIR deferral

LAMIR's real value was expected to lie in **multi-step look-ahead with mid-tree belief
updates**, not single-step Q re-weighting. Multi-step required a π_opp head, deferred
here pending opponent-view oracle queries in corpus generation. (commit message @ 5a4c9b9)

The deferral resolved the following day: π_opp was trained ([[gus-pi-opp-training]],
68.57% oracle top-1) and multi-step LAMIR ran in this harness — and lost to direct π_me
in every mode ([[lamir1-ceiling]]).

## Links

[[gus]] · [[pimc]] · [[lamir1]]
