---
title: Count-Fate Ledger
kind: topic
first_seen: 2026-07-14
last_updated: 2026-07-15
status: active
---

## The formulation

Jason's formulation, ratified 2026-07-14 (conversation 2026-07-13→14; tracked as
[issue #49](https://github.com/jasonyandell/mk5-main/issues/49)):

> The value of a hand is a belief-weighted ledger of count-fate scenarios, and a
> domino's importance is its role-weighted participation across that ledger.

Per the names doctrine this began **IDEATED** — conversation-designed. The build
is [[otis]] (started 2026-07-14, overnight); registered predictions and grading
live at [[otis-v0]]. This page remains the design record, not a build receipt.

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

## Measured (2026-07-15, [[otis-v0]])

The formulation now has empirical receipts:

- **The 3-2's context-bimodality is pervasive**: 86.8% belief-weighted / 83.3%
  uniform of qualifying opening decisions show ≥2 context modes ≥10 points
  apart, each ≥20% mass (valid-world filtered per issue #52).
- **Fates are learnable from information states**: +0.4775 nats over base rate,
  top-1 +17pp ([[otis]] fate heads) — the ledger is a real predictive object,
  not only a post-hoc decomposition.
- **Fate correlations fatten both tails** (all 15 off-diagonals positive;
  X_3-2↔tricks +0.42): independence overprices the make below the mean points
  (crossover ≈33) and underprices the true high tail (+17.8pp at ≥41) — the
  flatten-last doctrine measured.
- **The junk-retention economy is real and priceable**: tied-strategy rollouts
  cost ~1.0 s/decision (M=50, MPS); best replicable cell prices keeping a guard
  at +2.75 points where the clairvoyant prices −0.24 (fusion gap +2.99) —
  [[strategy-fusion]]'s zeroing measured directly. 65.1% of count-carrying
  tricks are walker catches.

## The emergent-values intent (2026-07-15)

The ledger's roles — protagonist, guard, walker — are *analysis
vocabulary*, never implementation targets. Jason's binding statement
(full text on [[otis]], Design commitments): named contrasts like
[[otis-guard-premium]] are diagnostic instruments; the program is a
system that learns the shape of value-shifting contexts from lessons in
fair hindsight, where protection, guards, and voids **arise from
training** rather than being written as code. What the ledger contributes
is the *label space* that makes such lessons stateable and gradeable —
not a feature set to hand the student.

## The argument's referees (2026-07-15)

Ratified in conversation (Jason, 2026-07-15); filed on
[issue #55](https://github.com/jasonyandell/mk5-main/issues/55) as Phase 2
amendments, binding on the lesson extractor. Doctrine underneath all of it:
the era-8 sharpening of [[strategy-fusion]] ("eq is not 42," full statement
there) — the oracle bootstraps and referees physics (what is reachable) but
may never grade a lesson; graders are behaviorally information-honest
rollouts only (tied; every seat a function of its own info-state).

- **Two hindsight leaks, both filtered.** The *deal leak* (a lesson conditions
  on belief-at-k, never the deal — already in #55) and the *outcome leak*:
  grading an alternative by replaying it on the realized hand. Sometimes the
  better move loses this particular hand; grading by the draw ("resulting")
  punishes correct insurance every time the threat fails to materialize —
  which is most of the time, which is why insurance is cheap. Registered form:
  **would it have worked *more often*** — tied rollouts across worlds sampled
  from the belief at decision k, alternative vs actual, belief-weighted; the
  realized world enters at its posterior weight and no more.
- **Triage, not blanket action.** Most sloughs are "meh — play the weakest
  thing"; a few really matter. Lesson harvesting and any consumer's action
  surface gate on the belief-weighted dispersion of fate-conditional prices
  across candidates ([[otis-phase-1]]'s override acted on every trigger —
  noise on the meh body was part of the price). High-dispersion sloughs and
  leads are the lesson sites ([[gus-drama-atlas]]: the opening lead dominates).
- **Three referees, three organs.** *Policy lessons* ("should I have kept the
  2?") are graded belief-averaged, per the outcome-leak filter. *Belief
  lessons* ("was my 20%-he's-void right?") are graded against realized deals
  in aggregate (calibration) — non-negotiable, because belief-averaged grading
  alone can never discover the belief is wrong; it launders a miscalibrated
  posterior into policy lessons. *The loop* is refereed by aggregate realized
  marks on reserved blocks; its tracked metric is the **claim-vs-cash gap**
  ([[otis-phase-1]] round 0: claimed +0.75 pts/override, table paid −0.13
  marks/game). Convergence = claims cash. This is the binding reading of
  Phase 1's exported constraint "realized outcomes must referee" — per organ,
  never per-lesson outcome-grading.
- **Field-relativity.** "Works more often" always means *against the current
  field*, whose oracle-distilled dispositions under-protect and under-attack;
  protection lessons and predation lessons must co-evolve through the loop.
  The fate-belief is decision-universal — consulted at bid, defense, slough,
  and lead alike ([[jud]]'s bidding-and-play-one-act, landing on the ledger);
  the auction is where its marks-signal is least masked, not where it lives.

## Open question

The cost half of the filed question is answered: tied-strategy rollouts price
retention at trivial cost ([[otis-v0]] P7). What remains — whether a
ledger-derived retention/discard policy beats the current best player in paired
marks — is [issue #53](https://github.com/jasonyandell/mk5-main/issues/53);
the original thread is
[issue #49](https://github.com/jasonyandell/mk5-main/issues/49). The
program-to-the-end bridge (clean corpus → lesson consumer → lesson extractor →
argument loop → promotion or graded negative), with the operating doctrine for
future implementers, is
[issue #55](https://github.com/jasonyandell/mk5-main/issues/55).

## Links

[[otis]] [[otis-v0]]
[[the-wall]] [[strategy-fusion]] [[candlewax]] [[expected-q-value]]
[[past-belief-future-direction]] [[belief-weighted-jud-mcts]] [[gus-drama-atlas]]
[[w42-phase2-hidden-domino-threat-attribution]] [[winning42-ch01-in-a-nutshell]]
[[jud]] [[texas-42]] [[belief-policy-value-algebra]]
