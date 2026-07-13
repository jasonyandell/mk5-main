---
title: Gus Scaling Ladder (100g → 10000g)
kind: experiment
first_seen: 2026-04-21
last_updated: 2026-07-13
status: complete
---

## Summary

Full adapter ladder across corpus sizes and model capacities, evaluated on 560 held-out
decisions using regret-based eval. Best adapter at this frontier: `v3_consistency_10k`
(3.4M params) at 0.551 Q-pt mean regret — first sub-1.0 result; 60% total reduction
from v2-3k baseline. Consistency loss scales better than plain distillation. (commit
messages @ 0472125, 5cdec8a, fdcd654, 286eb23, 31f0ec3)

## Full adapter ladder

| corpus | params | bot-match | regret (Q-pts) | near-ties |
|---|---|---|---|---|
| 100g | 0.4M | 59.3% | 2.48 | 68.9% |
| 1000g | 1.2M | 65.4% | 2.16 | 73.4% |
| 1000g | 3.4M | 62.7% | 2.10 | 70.5% |
| **2000g** | **3.4M** | **67.3%** | **1.60** | **75.4%** |
| 2000g | 7.4M | 65.7% | 1.65 | 75.2% |
| 2000g | 3.4M (120 ep) | — | 1.64 | — |
| **3000g** | **3.4M** | **67.9%** | **1.39** | **77.3%** |
| 10000g | 3.4M (v2) | 73.21% | 0.818 | — |
| **10000g** | **3.4M (v3)** | **76.07%** | **0.551** | — |

(560 held-out decisions; regret = oracle_best_eq − student_chosen_eq)

**Best**: `v3_consistency_10k` — 3.4M params, v3 consistency loss, 10k corpus. First
sub-1.0 regret. v3 beats v2 by −33% at 10k; gap widens vs near-tied at 3k. (commit
message @ 31f0ec3)

## Interpreting 1.60 Q-pt regret

Mean regret of 1.60 Q-pts on a ±42 scale = ~1.9% of Q-range lost per decision.
75% of "bot-mismatches" are near-tie alternative choices (within 0.1 Q-pts), not
strategic blunders. End-game (decisions 24-27): 100% match, 0 regret. (commit message @ 5cdec8a)

## Ceiling confirmation

120-epoch run of same 3.4M/2000g architecture yields 1.64 regret — essentially tied
with 60-epoch best (1.60) and 7.4M XL (1.65). All three 2000g runs converge on the
same regret floor. Architecture/data ceiling confirmed at 2000g × 3-7M params.
(commit message @ fdcd654)

## Scaling lessons

1. **Data dominates capacity.** 100g→1000g→2000g: clean regret reduction. 3.4M→7.4M
   on same 2000g data: flat.
2. **Explicit void features barely help once data + model compound.** The transformer
   infers voids attentionally from play tokens.
3. **Single-step PIMC underperforms direct π_me** (~62% vs ~66%) regardless of K.
   See [[gus-lamir-primitive-eval]].
4. **Decision hardness correlates with student regret.** Highest-regret decisions
   (0, 4, 8, 11, 12, 16) are the same decisions with highest oracle E[Q] spread.
   Student makes honest mistakes on strategically consequential positions. (commit message @ a50c9ef)

## Decision-hardness finding

Oracle E[Q] spread (max − min across legal actions) per decision_idx on held-out corpus:
- Decision 0 (first play): spread 13.2 Q-pts, student regret 4.0 Q-pts
- Uniform-random on dec 0 would give ~6.6 regret; student at 4.0 is doing real
  inference despite zero observable information

## Directions opened here — all subsequently closed

Three of the four directions ran and closed: the π_opp head was trained (68.57% oracle
top-1, [[gus-pi-opp-training]]), multi-step LAMIR was built and lost to direct π_me in
every rollout mode ([[lamir1-ceiling]]), and 10k-game scaling shipped (0.551 regret,
v3-10k — in the ladder above). Decision-difficulty-weighted training never ran; the
line pivoted to [[jud]]/[[champion]] instead.

## Links

[[gus]] · [[regret-eval]] · [[pimc]] · [[gus-v2-voids-1000g]]
