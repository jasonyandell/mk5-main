---
title: "Source: 5cdec8a"
kind: source
commit: 5cdec8a
date: 2026-04-21
author: Jason Yandell
---

## Commit message

> docs(gus): MORNING3_STATUS.md — final overnight report
>
> Session summary. Full adapter ladder (held-out 560 decisions):
>
>   corpus  params   bot-match  regret(Q-pts)  near-ties
>   100g    0.4M       59.3%      2.48         68.9%
>   1000g   1.2M       65.4%      2.16         73.4%
>   1000g   3.4M       62.7%      2.10         70.5%
>   2000g   3.4M     **67.3%**  **1.60**     **75.4%**  ← winner
>   2000g   7.4M       65.7%      1.65         75.2%    ← 2x params, no help
>
> Best adapter: v2_voids_big_2000g (d=256/6L, 3.4M params, 60 epochs).
> Mean regret 1.60 Q-pts on ±42 scale = ~1.9% of Q-range lost per decision.
> 75% of "bot-mismatches" are near-tie alternative choices, not real
> strategic errors.
>
> Key scaling lessons:
> 1. Data dominates capacity at this scale. 100g→1000g→2000g: clean
>    regret reduction curve. 3.4M→7.4M on same data: flat.
> 2. Explicit void features barely help once data + model compound.
>    The transformer learns voids attentionally from play tokens.
> 3. Single-step PIMC via Q_head underperforms direct π_me (62% vs 66%)
>    regardless of K. LAMIR's true value requires multi-step look-ahead,
>    which requires a π_opp head we haven't trained.
> 4. Decision hardness correlates with student regret. The student
>    makes honest mistakes on strategically significant positions
>    (first play of game, mid-game pivots), not on easy near-ties.
>
> Directions open:
> - π_opp head (requires opponent-view oracle queries in corpus regen)
> - Multi-step LAMIR tree search (needs π_opp)
> - Further data scaling (5000g or 10000g)
> - Decision-difficulty-weighted training
>
> Ships: gus/eval/eval_regret.py, eval_pimc.py, analyze_hard_decisions.py;
> gus/train/train_v2_voids.py; gus/model/{voids.py, student.py with v2
> and XL architectures}.

## Links

[[experiments/gus-scaling-ladder]]
