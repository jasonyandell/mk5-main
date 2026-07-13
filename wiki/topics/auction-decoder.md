---
title: Auction Decoder
kind: topic
first_seen: b28fb55a
last_updated: b28fb55a
status: active
---

Research position: **SELECTED LANE** ([[research-lane-selection]] Lane A).
Implementation status: design registered; build begins after
[[partnership-wall-research]] Stage 0 closes.

The auction decoder models the probability of each public bid or pass
conditioned on the actor's possible hand, seat, role, prior bids, who raised
whom, match score, declaration, and policy population:

`π_θ(bid | hand, seat, role, auction prefix, score, policy type)`

It is evaluated **first as an inference instrument** — a likelihood that
reweights consistent worlds by what the auction reveals — and only afterward
connected to predeclared consumers. This is the
[[search-literature-transfer|policy-based-inference transfer]] applied to the
one surface where information has already produced marks.

## Why the auction is the strongest near-term surface

The evidence lines up unusually well ([[partnership-wall-research]] Stage 2):

- realized-outcome bidding is the champion's only demonstrated learned marks
  gain (+0.38/+0.42 marks/game, [[w42-plateau-probe]], [[champion]]);
- auction conditioning improved held-out belief accuracy +2.59pp
  ([[w42-champion-auction-belief]]);
- [[w42-book-second-pass]] uncovered a large bid-to-hand, role/order, score,
  and partner/opponent semantics corpus the first extraction missed;
- the self-play fixed point showed double-dummy P(make) causes systematic
  over-bidding ([[w42-champion-selfplay-fixed-point]]) — so the decoder's
  consumers must price realized outcomes, a constraint this design carries
  from the start rather than discovering late.

## Instrument evaluation (before any consumer claim)

Registered diagnostics, per [[partnership-research-gates]] (auction-decoding
and voluntary-action-likelihood rows):

- held-out negative log-likelihood and calibration;
- true-world rank and posterior mass on held-out deals;
- effective sample size and support collapse under repeated reweighting;
- role/order generalization beyond one seat assignment;
- the book's concrete bid-semantics hypotheses as fixtures: `31 ⇒ ≥1 double`,
  `35 ⇒ two offs / one five-count`, shuffler-last `30/31` weak evidence, and
  the `{30,31,35,36}` bid lattice (open question at
  [[w42-book-second-pass]] §1).

## The policy-type mixture

One universal decoder is the wrong prior: a tight book-like bidder, `head_8`,
an aggressive bidder, and a weak/noisy bidder assign different meanings to the
same 35. The decoder is a mixture over a latent policy type `z`:

`P(bid | world) = Σ_z P(bid | world, z) · P(z | public history)`

The posterior infers hidden hand and policy type jointly, and retains a
nonzero unmodeled component so a deceptive or out-of-distribution bidder
cannot collapse the particle set. [[convention-aware-blueprint-search]]
registers the same construction for reading opponent book-likeness during
play; the decoder is its auction-side instance.

## Predeclared consumers

Per the consumption law ([[candlewax]], [[consumption-ledger]]): every
inference artifact names its consumer before it is built, or it becomes
another head sitting beside the policy. In order of directness:

1. **A bidder conditioned on posterior features** — still trained against
   realized make/set outcomes, never double-dummy P(make)
   ([[w42-champion-selfplay-fixed-point]] makes this load-bearing).
2. **Defender/setter policy features** derived from what the auction makes
   probable about the bidder's hand.
3. **A matched-auction counterfactual** — does the same hand act differently
   after partner versus opponent bidding? This is the cheapest causal probe of
   auction information value.

Downstream, the decoder is also the calibrated action-likelihood component
that [[belief-weighted-jud-mcts]] J3 requires for mid-tree belief updates;
J3 is not eligible before an instrument of this kind passes held-out
evaluation ([[research-lane-selection]]).

## What would deselect this lane

- Held-out NLL no better than a hand-independent bid model (the auction
  carries no decodable hand information at this population);
- posterior gains that disappear under masked-auction comparison or fail
  role/order generalization;
- a consumer gain that vanishes when posterior features are replaced by raw
  auction tokens (the decoder added nothing beyond what the consumer could
  learn directly).

## Links

[[research-lane-selection]] [[search-literature-transfer]]
[[partnership-wall-research]] [[partnership-research-gates]]
[[w42-book-second-pass]] [[w42-champion-auction-belief]]
[[w42-champion-selfplay-fixed-point]] [[champion]] [[jud]]
[[belief-weighted-jud-mcts]] [[convention-aware-blueprint-search]]
[[pi-opp-head]] [[candlewax]]
