---
title: Auction Decoder v0 — instrument validation
kind: experiment
first_seen: local-2026-07-13
last_updated: local-2026-07-13
status: active
---

Can a decoder recover hand information from one-round arena auctions — and
does the gain appear only where the bidder's action actually depends on its
hand? First build of [[auction-decoder]] (Lane A of
[[research-lane-selection]]), run as an inference instrument: NLL, accuracy,
calibration. No marks or policy claims.

Predictions were registered before training (`scratch/lane-a/PREDICTIONS.md`,
reproduced in the run artifacts); code at `champion/bid_decoder.py` with
leakage tests in `champion/test_bid_decoder.py`.

## Setup

- Corpus: 40,774 hand snapshots (main-checkout `scratch/jud-v1/corpus/`),
  deduped to unique hands; three logged bidder populations —
  `margin:wp(r8)` (9,911 hands), `net:wp` (5,900), `random` (uniform over
  legal). One-round auction, four bid decisions per hand, 15 action classes
  (pass + observed bid levels). 90/5/5 deal-hash split; test = 4,328
  decisions.
- Features: actor hand (63-dim), seat position in bidding order, bid prefix
  (strictly-prior bids only — later seats' bids and the winner's declaration
  are excluded by construction, with a test).
- Variants: marginal-frequency baseline; hand-independent (seat+prefix);
  pooled (hand+seat+prefix); population-conditioned (+population one-hot).

## Results (held-out NLL per decision / top-1 / ECE)

| model | NLL | acc | ECE | NLL margin | NLL net | NLL random |
|---|---|---|---|---|---|---|
| marginal baseline | 1.3789 | 0.581 | 0.003 | 1.288 | 0.986 | 1.988 |
| hand-independent | 0.9807 | 0.599 | 0.012 | 0.735 | 0.683 | 1.770 |
| pooled (hand) | 0.7660 | 0.771 | 0.011 | 0.375 | 0.402 | 1.902 |
| population-conditioned | 0.5224 | 0.846 | 0.019 | 0.199 | 0.164 | 1.525 |

Registered-prediction grades:

1. Pooled beats marginal baseline — **PASS** (Δ 0.61 nats).
2. Hand features help only where the bidder is hand-dependent — **PASS**, the
   experiment's causal signature: adding the hand improves `margin:wp` by
   `+0.361` and `net:wp` by `+0.281` nats, and *worsens* `random` by `0.131`
   (a hand-independent policy's actions carry no hand information; the
   negative is small-sample overfit). The auction decodes the hand exactly
   when the bidding policy consults the hand.
3. Population-conditioning beats pooled — **PASS** (Δ 0.24 nats): the same
   (hand, seat, prefix) means different things under different policies — the
   [[auction-decoder]] policy-type-mixture premise in its crudest
   (label-given) form.
4. Bid-thinness — **PASS for the real populations, MISS for random**:
   `margin:wp` mean winning bid 31.01, `net:wp` 30.39, pass/30/31 dominate;
   but `random` is flat-and-high (mean 37.79, spike at 42), not thin. Only 6
   `84` bids exist in ~10k margin hands.

## What this does and does not establish

Established: arena auctions carry decodable hand information for real
bidders; a population-conditioned likelihood is measurably sharper; the
decoder scaffold (leakage-disciplined prefix features) works. This clears the
instrument half of Lane A's first gate on this corpus.

Not established: any consumer value (bidding, defense, belief) — the
predeclared consumers of [[auction-decoder]] remain unbuilt. Not testable
here: the book's bid-semantics fixtures (`31 ⇒ double`, `35 ⇒ shape`) — the
real populations almost never bid above 32 (six 84s total). Testing those
requires a corpus deliberately seeded with 35+/double bidders, which no
current arena bidder produces (`HeuristicBidder` bids minimum legal raises).
The latent (inferred) policy-type mixture also remains unbuilt — tonight's
population input is an observed label, not an inferred posterior.

Follow-ups filed on [[auction-decoder]]: enriched-bid corpus generation;
true-world-rank/ESS evaluation against particle filtering; the
matched-auction counterfactual consumer.

## Links

[[auction-decoder]] [[research-lane-selection]] [[search-literature-transfer]]
[[partnership-research-gates]] [[w42-book-second-pass]] [[champion]]
