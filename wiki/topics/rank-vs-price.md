---
title: Rank vs price — why PIMC's flaw bites the auction, not the play
kind: topic
first_seen: 2026-07-05
last_updated: 2026-07-06
status: active
---

## The mechanism

[[pimc]]'s [[strategy-fusion]] optimism — per-world double-dummy values assume the
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
   falling 0.52 (bid 30) → ~0.19 (bid 41), N=604. The oracle curve falls too
   (0.64 @ 30 → 0.37 @ 39, N=50) but slower: the absolute gap holds at
   +0.12–0.17 while the realized/oracle ratio worsens 0.81 → 0.64 over bids
   30–39 — relative inflation is largest exactly where thresholds bind.
2. Shape-not-order play levers null — measured three ways (#25 ×3 configs, #27).
3. The prediction it makes: an evaluator calibrated in absolute terms fixes
   bidding *without touching play*. `net:wp` already half-demonstrates this —
   frozen realized-play calibration, strongest bidder on the board
   ([[champion]] rung #22). **The full test ran, confirmed it, and then beat the
   baseline** ([[w42-jud-v0]] → [[w42-plateau-probe]], 2026-07-06): a value trained
   on realized outcomes (`V_realized`), consumed only as a bid-side price with play
   left on `lens:ev`, dissolved the #26 over-bidder. The v0 loop reached parity with
   `net:wp` (−0.07 [−0.66, +0.49]); scaling on-policy data 3× per round then carried
   it **past** `net:wp` (+0.38 [+0.09, +0.67] / +0.42 [+0.12, +0.72]) — the pricing
   mechanism validated and then *dominant*, without ever touching the near-ceiling
   play argmax. The round-0 miss added a rider the mechanism did not name: optimism
   re-enters the price not only through the evaluator (double-dummy) but through
   *selection* (the winner's curse — a calibrated head still over-bids the hands it
   happens to over-rate), and on-policy data volume is what closes that second
   channel (the plateau probe showed it was data-limited, not structural).

## The other half — play consumes rankings, and the ranking gap survives

[[w42-jud-v1]] tested the play half of the same law by replacing `lens:ev`'s
oracle-per-move ranking with a value-native player over the same `V_realized`
head. The result confirms the framing from the losing side: **greedy 1-ply value
play is a bad *ranker* even when the head is a fine *evaluator*.** The head prices
positions honestly (its MAE sharpens root→terminal), but reading it greedily
argmaxes over post-*move* values that share one hand-level Monte-Carlo label, so the
ranking is noisy — the full stack loses −4.37 at round 0 and the self-play loop moves
play quality **zero** (unlike the bidder it dissolved). What *does* recover the play
channel is turning the sharp post-*trick* leaf into a ranking via search: `judsearch`
(belief-lift worlds, current-trick rollout, `V_realized` leaves, **no oracle
anywhere**) plays −1.16 where greedy scored −3.44 — two-thirds of the gap, without any
oracle. But it stops short of `lens:ev`: **the oracle's rankings are still unbeaten**,
and neither more worlds nor a better-calibrated head closes the rest. So the law now
reads with both halves measured: prices went value-native and *won* (the bidder beats
the hand-tuned baseline), while a realized-outcome leaf inside search recovered *most,
not all* of the ranking gap — the per-move discrimination a perfect-information oracle
gives is the residual play edge value-native pricing does not touch.

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
- [[w42-jud-v0]] — the experiment that validated the mechanism at parity
- [[w42-plateau-probe]] — the follow-on that carried the value-native price past `net:wp`
- [[w42-jud-v1]] — the play half: greedy value play is a bad ranker; search recovers most (not all) of the gap oracle-free; the oracle's rankings stay unbeaten
- [[w42-champion-selfplay-fixed-point]] — the over-bidder this explains
- [[belief-conditioned-self-play]] — the "all about bidding" fragment this resolves
- [[arena]] · [[champion]] — where the measurements live
