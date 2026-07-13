---
title: "Source: fdcd654"
kind: source
commit: fdcd654
date: 2026-04-21
author: Jason Yandell
---

## Commit message

> docs(gus): MORNING3 update — 120-epoch run confirms ceiling
>
> Added v2_voids_big_long (same 3.4M/2000g arch but 120 epochs) to the
> adapter ladder. Result: 1.64 regret — essentially tied with 60-epoch
> best (1.60) and XL (1.65). All three 2000g runs converge on the same
> regret floor.
>
> Conclusion: at 2000 games of corpus data with ~3M-7M params, we've hit
> the architecture/data ceiling. Further improvement requires either
> more data or a fundamentally different training signal (π_opp head,
> multi-step LAMIR look-ahead).
>
> Best adapter: v2_voids_big_2000g (60 epochs) at 1.60 Q-pt mean regret.
> ~1.9% of Q-range lost per decision. 75% of mispicks are near-ties.
>
> Session final: 9 commits, full LAMIR-ready distillation pipeline,
> 2000-game oracle-labeled corpus, regret/hardness eval infrastructure.

Confirms architecture/data ceiling: 120 epochs (1.64 regret) matches 60 epochs (1.60)
and 7.4M XL (1.65). All 2000g variants converge to the same regret floor.

## Links

[[experiments/gus-scaling-ladder]]
