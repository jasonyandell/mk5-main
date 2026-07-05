---
title: Rank vs price — why PIMC's flaw bites the auction, not the play
kind: topic
first_seen: local-2026-07-05
last_updated: local-2026-07-05
status: active
---

## The mechanism

[[pimc]]'s strategy-fusion optimism — per-world double-dummy values assume the
player will *know the world* when acting later — is a distribution-shape error:
it fattens the upper tail of every action's outcome blob. The stack's two
decision types consume that blob differently:

- **Play consumes rankings.** A play decision is an argmax over sibling actions
  that share an information state and most of the future information structure.
  An optimistic transform applied similarly across siblings largely cancels in
  the ordering; argmax is first-order insensitive to it. This is why measured
  play is near-oracle (0.49–0.55 regret, [[gus]]) and why every play-side lever
  nulls — [[w42-champion-selfplay-fixed-point|#25]] belief reweighting and #27
  score risk change the *values* without reordering the top action.
- **Bids consume prices.** A bid decision reads tail mass — P(outcome ≥
  threshold) — and compares it against an external alternative (pass, the
  [[champion]] `race_wp` utility). Nothing cancels: the inflated tail is
  consumed as a cardinal number, and the optimism lands directly in the
  decision. That is the #26 over-bidder ([[w42-champion-selfplay-fixed-point]]),
  mechanically.

So "it's all about bidding" — the least-recovered fragment flagged at
[[belief-conditioned-self-play]] — has a mechanism beneath the reasons already
on record (game-tree root, marginal-value peak, public channel for
conventions): **the auction is where the stack's one systematic error term is
read cardinally rather than ordinally.** PIMC ranks actions well but prices
contracts wrong; play consumes rankings, bids consume prices.

## Status: interpretation, with three measured legs

The mechanism is an interpretation laid over measurements, not itself a
measurement. Its legs:

1. Fusion inflation grows with the threshold — measured:
   `champion/optimism_gap.json` has the realized best-declaration make-rate
   falling 0.52 (bid 30) → ~0.19 (bid 41) while the oracle sits at 0.64 @ 30;
   thresholds bind exactly where the inflation is largest.
2. Shape-not-order play levers null — measured three ways (#25 ×3 configs, #27).
3. The prediction it makes: an evaluator calibrated in absolute terms fixes
   bidding *without touching play*. `net:wp` already half-demonstrates this —
   frozen realized-play calibration, strongest bidder on the board
   ([[champion]] rung #22). The full test is [[jud]] v0's registered
   prediction 1.

## Corollary — the calibration law for evaluators

Any evaluator whose output is consumed as a price (bid thresholds, pass/play
choices at the auction, contract selection) must have tail masses redeemable by
the policy that will actually play the hand. The perfect-information oracle
structurally cannot satisfy this ([[jud]], the brick wall); a value trained on
realized outcomes satisfies it by construction. Evaluators consumed only as
rankings (play-phase argmax) tolerate optimism that price-consumers cannot.

## Links

- [[pimc]] — the flaw whose bite this localizes
- [[jud]] — the design that makes prices honest (value-native pricing path)
- [[w42-champion-selfplay-fixed-point]] — the over-bidder this explains
- [[belief-conditioned-self-play]] — the "all about bidding" fragment this resolves
- [[arena]] · [[champion]] — where the measurements live
