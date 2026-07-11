---
title: "LAMIR-primitive inference eval: direct vs PIMC-Q vs PIMC-belief"
kind: experiment
first_seen: 5a4c9b9
last_updated: 5a4c9b9
status: active
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

LAMIR's real value lies in **multi-step look-ahead with mid-tree belief updates**, not
single-step Q re-weighting. Multi-step requires a π_opp head (not yet trained; requires
opponent-view oracle queries in corpus generation). This eval script is the harness
that multi-step LAMIR will plug into when that head is available. (commit message @ 5a4c9b9)

## Links

[[gus]] · [[topics/pimc]] · [[topics/lamir1]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Headline numbers (65.4 / 62.1 / 61.8) and per-decision hints trace only to the commit message of 5a4c9b9 — no results JSON/log artifact exists in-repo.
- `gus/eval/eval_pimc.py` confirms the mechanism: pimc-q uses the single corpus-saved `world_assignment` per decision (K=1; `--k-corpus-cap 200` exists but is unused in the current path), pimc-belief samples K=50 worlds from belief_head and argmaxes mean Q.
- Cheap next probe: wire up the dangling `--k-corpus-cap` averaging to test whether pimc-q at K=50 oracle worlds closes the gap to direct — distinguishes sampler quality from single-step PIMC being inherently redundant.
