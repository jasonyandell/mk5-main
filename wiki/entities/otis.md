---
title: Otis — the fate-ledger-native player
kind: entity
first_seen: 2026-07-14
last_updated: 2026-07-14
status: active
phase: 2026-07-15 — v0 BUILT and graded overnight ([[otis-v0]], all seven predictions graded). The player is an incumbent-parity bidder; the decomposition wins at matched arms (+0.29 marks/game pooled, CI excludes zero, round-0) and passes calibration 3× over band; the instrument suite (fate parser with three exact referees, world-bank clustering, tied-rollout retention pricing at ~1 s/decision) is live. No promotion over margin:wp(r8) — parity, both blocks. Corpus contamination found and filed (#52); follow-ups #51, #53; the program-to-the-end bridge is #55 (clean deck on Vast→HF, lesson consumer, lesson extractor, argument loop, promotion-or-graded-negative).
---

## What it is

Otis is the [[count-fate-ledger]] made into an organism: a player whose value is
computed from an auditable ledger of count-tile fates rather than learned as an
opaque scalar. Five organs:

1. **The fate parser** — replays a finished hand and writes each count tile's
   biography: who captured it, by what mechanism, on which trick. Machine-checked
   by an exact identity (captured count + tricks = final points) cross-refereed
   against the [[engine]].
2. **The net** — from an information-state (own hand, auction, public history —
   never hidden cards), a shared trunk feeds a 43-bin total-points pricing head
   (the [[w42-jud-v0]] `V_realized` shape, the only organ licensed to price) plus
   five per-tile fate heads and a trick head, tied by a consistency penalty.
   Fates are offline labels only, per the leakage boundary
   ([[w42-phase2-hidden-domino-threat-attribution]]).
3. **The loop** — regenerate on-policy self-play, re-parse fates, retrain, re-gate
   in the paired [[arena]]; [[w42-jud-v0]]'s proven machinery with decomposed
   targets.
4. **The analysis suite** — context-clustered value over the joint-world bank
   ([[joint-world-tensor]]): ledger cards, interaction structure, the 3-2
   bimodality that belief-averaging destroys (#25).
5. **The tied-rollout tool** — one action held across belief-consistent worlds
   until the actor's own observations diverge; prices the guard/walker
   junk-retention economy that clairvoyant per-world evaluation zeroes via
   [[strategy-fusion]] — the open question filed on [[count-fate-ledger]].

## Design commitments (from the 2026-07-14 red-team, binding)

- **The collapse stays distributional.** Per-tile fate marginals get the mean of
  total points right and the tails wrong (fates are positively correlated; sweeps
  put all five tiles on one side), and bids consume tails ([[rank-vs-price]]). So
  the pricing path is the joint 43-bin head; the fate heads are constrained
  decomposition, graded on calibration. In v0 the ledger is an instrument; it must
  earn load-bearing status through a registered arm.
- **Training is on-policy with real auctions.** The 10k-game bid-30 eq corpus in
  `gus/data/` is world-bank and shakedown material, never the bidder's training
  data (policy-conditional pricing law, [[jud]]).
- **Mechanism taxonomy is the trajectory-decidable subset** — (capture side) ×
  (led / followed / trumped-in / sloughed). "Under duress" and "pulled" need
  counterfactuals and are explicitly deferred.

## Relation to jud

Otis is not a fork of [[jud]]; it is jud v2's target-granularity question
([[jud-target-granularity]]) asked with a different object: not per-move values
(graded null at v1 capacity) but per-hand fate structure, consumed at the auction
— the only lane where marks have ever moved. Control-vs-treatment arms at matched
capacity make the comparison honest; [[otis-v0]] carries the bands, including the
tie and the decomposition-tax negative.

## The program to the end

[Issue #55](https://github.com/jasonyandell/mk5-main/issues/55) is the bridge
from v0 to the end of the program. Its first two phases execute overnight
2026-07-15→16: [[otis-phase-r]] (the clean deck — corpus regen on Vast with
write-time validity, recorded posterior weights, the decl-8 purge) and
[[otis-phase-1]] (the first lesson consumer — the retention override on
`lens:ev`, tied-rollout and fate-head variants).

## Links

[[count-fate-ledger]] · [[otis-v0]] · [[otis-phase-r]] · [[otis-phase-1]] ·
[[jud]] · [[w42-jud-v0]] · [[rank-vs-price]] · [[strategy-fusion]] ·
[[joint-world-tensor]] · [[arena]] · [[the-wall]]
