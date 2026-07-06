---
title: "Source: a50c9ef"
kind: source
commit: a50c9ef
date: 2026-04-21
author: Jason Yandell
---

## Commit message

> eval(gus): decision-hardness analyzer
>
> Computes oracle E[Q] spread (max - min across legal actions) per
> decision_idx on a held-out corpus. High spread = strategically
> consequential decisions; low spread = near-ties where any legal play is
> roughly equivalent.
>
> On the 2000g-big adapter, the student's highest-regret decisions
> (0, 4, 8, 11, 12, 16) are the SAME decisions with highest strategic
> spread. Honest mistakes on genuinely hard positions, not trivial slips.
>
> Decision 0 (first play): spread 13.2 Q-pts, student regret 4.0 Q-pts.
> Uniform-random picking would give ~6.6 regret, so the student is doing
> real inference even with zero observable info.

Confirms that student regret tracks genuine strategic difficulty: hardest decisions
by regret = highest oracle E[Q] spread. The student is not making random errors.

## Links

[[topics/regret-eval]] · [[experiments/gus-scaling-ladder]]
