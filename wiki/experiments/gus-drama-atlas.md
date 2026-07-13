---
title: Gus Drama Atlas (outcome-variance / fragility / belief-sharpness join)
kind: experiment
first_seen: 2026-04-25
last_updated: 2026-07-11
status: complete
---

## Summary

Pure analytics over the existing `q_per_world [M, 7]` tensors — no new model training,
703 seconds of compute — tagging all 280,560 corpus decisions (10k train games + 560 eval)
with three derived quantities: outcome-variance, action-choice fragility, and belief
sharpness. This is the [[past-belief-future-direction]] analysis, run three days after it
was proposed (`76355ac`, `gus/analysis/drama_atlas_findings.md`).

## The three quantities

1. **Outcome variance** — `q_per_world[:, action_taken].std()` over M≈4000 worlds. Mean 13.1
   Q-pts, median 14.4, p99 29.1.
2. **Action fragility** — count of distinct oracle-best actions across M worlds. 1 (oracle
   agrees completely) = 40.3% of decisions; 2 = 21.3%; 3-7 (genuine fog-of-war) = 38.4%.
3. **Belief sharpness** — `1 − H(P_belief)/log(3)` averaged over unseen dominoes. Mean 0.064,
   median 0.020 — near-uniform through d_idx 20, spiking only in the final 2-3 tricks.

## Quadrant analysis

Foggy = belief_sharpness < 0.020 (train median); fragile = action_fragility ≥ 3.

| Quadrant | Description | Fraction |
|---|---|---:|
| easy-robust | oracle agrees, Gus has signal | 37.8% |
| foggy-forced | oracle agrees, Gus is guessing | 23.8% |
| easy-high-stakes | oracle disagrees, Gus has signal | 12.2% |
| **drama** | oracle disagrees AND Gus is in the dark | **26.2%** |

**Drama fraction: 26.2%** of all decisions are genuine fog-of-war calls. Late game (d_idx ≥
20): drama fraction is 0.0% — played dominoes collapse the belief space deterministically by
tricks 5-7. The game's craft is entirely front-loaded.

## Where drama concentrates: the opening lead

| Trick position | Drama fraction | Mean fragility |
|---|---:|---:|
| **Lead (1st play)** | **64.9%** | 3.97 |
| 2nd play | 14.6% | 2.18 |
| 3rd play | 12.7% | 2.17 |
| 4th play | 12.7% | 2.04 |

**62% of all drama decisions are lead decisions.** This quantitatively confirms the human
expert intuition that the opening lead is the heart of the game: the player choosing what to
play first faces maximum oracle disagreement and minimum belief signal. By game phase: early
(d_idx 0-11) 39.4% drama, mid (d_idx 12-19) 32.6%, late (d_idx 20-27) 0.0%.

## Gus vs. oracle mode

| Quadrant | Gus-mode agreement |
|---|---:|
| easy-robust | 86.5% |
| foggy-forced | 85.0% |
| easy-high-stakes | 46.5% |
| drama | 52.6% |
| Overall | 72.4% |

Gus is primarily a mode-player, degrading to near-coin-flip (52.6%) on drama decisions. This
is not evidence of learned meta-strategy — it is explained by ambiguous BCE training signal:
when oracle worlds split across 5-7 different best actions, the argmax label compresses a
near-uniform distribution into one target, and the trained output drifts accordingly.

## Methodology caveat: the refined drama filter

A later pass replaced the quadrant definition with an oracle-marginal filter for "real
drama": `marginal_eq_gap ≤ 1.0` (top1 − top2 E[Q] over legal actions) ∧ `count_unplayed ≥ 15`
(points still live), restricted to mid/endgame phases — plus a forced-move exclusion
`n_legal_actions ≥ 2`. Without the exclusion, 31,421 of 67,605 raw real-drama flags in
`drama_atlas_v2.parquet` are single-legal decisions where `marginal_eq_gap` is trivially 0;
the filtered set is 36,184 decisions (gus/analysis/real_drama_examples.md @ 233b7dc5;
exclusion counts from [[gus-qmean-router]]'s side finding). Any "real drama" view should
exclude forced moves.

## Concrete example

Seed 900016, d_idx=0 (opening lead), decl=6: belief sharpness 0.007 (pure prior), action
fragility 7 (all 7 legal actions are oracle-best in some world), outcome variance 27.9
Q-pts. Oracle world distribution: 33% / 24% / 16% / 13% / 11% / 3% / 0% across the seven
actions. Gus chose the mode (33%); the actual game play was 4.3 Q-pts suboptimal.

## Links

[[gus]] [[past-belief-future-direction]] [[belief-bayes-ceiling]] [[joint-world-tensor]] [[regret-eval]]
