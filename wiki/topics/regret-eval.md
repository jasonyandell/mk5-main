---
title: Regret Eval (E[Q] lost vs oracle's best legal)
kind: topic
first_seen: 2a09050
last_updated: b007cf3
status: active
---

## Overview

Regret eval is the primary quality metric for [[gus]]. For each held-out decision: `regret = oracle_best_eq - student_chosen_eq`. Averaged over all held-out decisions, it measures how many E[Q] points the student loses relative to the oracle's best legal play (2a09050).

## Why better than bot-match

Texas 42 is full of near-tie positions where multiple actions have E[Q] within 0.1 Q-points of each other — the plays are strategically equivalent. Bot-match counts "picked the wrong one of a tied pair" as a miss; regret counts it as zero loss. Regret captures actual strategic cost, not nominal agreement with one particular tie-breaking choice (2a09050).

**Companion metric: near-ties rate** — fraction of bot-mismatches where the student's pick is within 0.5 Q-points of the oracle's best. In practice, 70–75% of mismatches are near-ties (2a09050):

| Adapter | Bot-match | Mean regret | Near-ties rate |
|---|---|---|---|
| v1_full_1000g | 65.4% | 2.16 Q-pt | 73.4% |
| v2_voids_1000g | 63.6% | 2.22 Q-pt | 72.0% |

2.16 Q-points of mean regret on a ±42 Q-point scale ≈ 2.5% of range lost per decision. The 35% "bot-mismatches" are overwhelmingly near-tie alternative choices, not strategic blunders (2a09050).

## Decision-hardness correlation

High-regret decisions correlate with high E[Q] spread (oracle max − oracle min across legal actions). The student's highest-regret decisions (0, 4, 8, 11, 12, 16) are the same decisions with highest strategic spread (a50c9ef):

| Decision | E[Q] spread | Student regret |
|---|---|---|
| 0 (first play) | 13.2 | 4.0 |
| 4 | high | 8.64 |
| 8 | high | 4.12 |

Uniform-random picking at decision 0 would give ~6.6 regret; the student achieves 4.0, doing real inference with zero observable information (a50c9ef).

End-game decisions (24-27): 0 regret, 100% bot-match. Perfect — forced play or near-forced (2a09050).

## Bimodal distribution of regret

Regret is NOT normally distributed (f0139a3, PRACTICALITIES receipt #5):

- **73% of decisions have exactly 0 regret** — the student matches the oracle's argmax.
- **6% blunder tail (regret > 5 Q-pts)** drives essentially all of the 1.39 mean regret.
- The remaining ~21% is spread across small non-zero regret (near-ties and modest misses).

Implication: mean regret is misleading as a scalar. The action is in the blunder tail — reducing the 6% blunder rate is worth more than shaving the middle distribution. The [[v-pi-decoupling]] finding explains where those blunders come from; the [[consistency-regularizer]] is the targeted fix (f0139a3, b007cf3).

Near-tie rate (70–75%) confirms: most "mismatches" are genuinely equivalent plays. The real ceiling is on the 6% blunder tail (f0139a3).

## Comparison to K1 grading

[[k1-grading]] in LEM/Burl asks "did the model beat the bot?" (binary). Regret asks "by how much did the model miss the oracle's best?" (continuous). Regret is more informative for a supervised-distillation student where every legal play can be scored by the oracle (2a09050).

## Links

[[gus]] [[expected-q-value]] [[k1-grading]] [[pimc]] [[v-pi-decoupling]] [[consistency-regularizer]]
