---
title: Count-Fate Ledger
kind: topic
first_seen: 2026-07-14
last_updated: 2026-07-14
status: active
---

## The formulation

Jason's formulation, ratified 2026-07-14 (conversation 2026-07-13→14; tracked as
[issue #49](https://github.com/jasonyandell/mk5-main/issues/49)):

> The value of a hand is a belief-weighted ledger of count-fate scenarios, and a
> domino's importance is its role-weighted participation across that ledger.

Per the names doctrine this is **IDEATED** — conversation-designed, no repo
artifact. This page is the design record, not a build receipt.

## Three layers — why a corrected scalar is not the object

1. **Per-world evaluation is broken by [[strategy-fusion]].** Within each sampled
   world the oracle plays clairvoyantly, so insurance-shaped holdings (guards) and
   lottery-shaped holdings (walkers) price at zero: the clairvoyant never buys
   insurance, nor lottery tickets. Both exist only for one information-consistent
   strategy scored across many worlds.
2. **Even the true world's Q describes a game that will never be played.** Q assumes
   four clairvoyants; the actual table is four belief-players. E[Q] therefore
   averages, over worlds, numbers none of which describes any realizable game —
   including the true world's. Value lives on information states; the oracle is
   physics (what is reachable), not prediction. This sharpens [[candlewax]]: the
   distribution is mute partly because the flattening happened inside the oracle,
   along axes nobody chose.
3. **The deliverable is the ledger, not the flattened value.** Any decision
   eventually flattens to a preference; the failure is flattening first. Summing
   the ledger is legitimate, last, and lossy on purpose.

## Ledger structure

- **Tractability anchor**: 35 of 42 points ride on five tiles (5-5, 6-4, 5-0, 4-1,
  3-2). The scenario space is the biography of the count tiles — who captures each,
  by what mechanism (pulled, fed to partner, sloughed under duress, trumped), in
  what order. Small enough to enumerate, unlike trajectory space.
- **A row** = (count tile, fate, plausibility, swing). Row probability has three
  factors: plausible placement (belief), reachable cliff (oracle physics), and a
  navigable line (tables walk recognizable — largely book — lines). First-order
  placement-only attribution ([[w42-phase2-hidden-domino-threat-attribution]] and
  the branch atlases) computes only the first factor.
- **Rows carry preconditions (theirs) and vetoes (yours).** Cast roles:
  *protagonist* (the count tile), *guard* (junk kept to deny their rows — a spare
  2 shielding the 3-2 from a pulling lead), *walker* (junk that becomes a boss
  lead late and catches sloughed count; walkers per [[winning42-ch01-in-a-nutshell]]
  — distinct from era-5's "walker" codename in [[ideated-not-built]]), *enabler*.
- **Guards and walkers are one junk-retention economy with two signs.** A junk
  tile's retention value is the mass × swing sum over every row where it is cast.
  Walker status is mostly public (suit exhaustion); belief prices the stragglers
  and the timing. This makes the discard decision well-posed: which junk to keep
  now to deny or catch later.
- **Plans fall out as row operations** — fight for hoped rows, spend vetoes to
  delete feared rows, sequence so friendly rows come due first. Count fates are
  the destinations; good leads are the roads. Search destinations first
  (tractability), roads second (efficiency).

## Relation to existing mechanisms

- [[past-belief-future-direction]] — the mode/signal/hedge/gamble meta-strategies
  over the same per-world tensor; the ledger is the decomposition object those
  strategies would act on.
- [[belief-weighted-jud-mcts]] — the surviving search consumer; ledger rows are
  the interpretable coordinates of what such a search would fight over.
- [[gus-drama-atlas]] — drama decisions (oracle disagrees, belief blind) locate
  where rows have mass; the opening lead dominates.
- **Measurability**: joint-world artifacts already retain `q_per_world` +
  `world_hands`; worlds where candidate actions diverge sharply mark where rows
  glow, so directed sampling replaces uniform coverage. Pricing guards and vetoes
  additionally needs tied-strategy rollouts — one action across worlds until the
  player's own observations differ — which no current tool performs.
- **Consumer** (per the goals hierarchy on [[the-wall]]): the load-bearing
  candidates are [[jud]]'s auction lane (a bid prices context-conditional domino
  value under belief) and play. The teaching artifact — the post-game "you could
  have known they had X because they would have done Y" made rigorous and
  pre-hand — is a side benefit, never load-bearing.

## Open question

Whether tied-strategy rollouts can price the junk-retention economy at acceptable
cost, and whether a ledger-derived retention policy beats the current best player,
is filed in `questions/open.md` and tracked as
[issue #49](https://github.com/jasonyandell/mk5-main/issues/49).

## Links

[[the-wall]] [[strategy-fusion]] [[candlewax]] [[expected-q-value]]
[[past-belief-future-direction]] [[belief-weighted-jud-mcts]] [[gus-drama-atlas]]
[[w42-phase2-hidden-domino-threat-attribution]] [[winning42-ch01-in-a-nutshell]]
[[jud]] [[texas-42]]
