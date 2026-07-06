# jud v1 — registered predictions (2026-07-06 ~02:35, pre-build-completion, pre-any-data)

Build in progress: ONE net (hand + canonical auction + play history → 43-bin margin
distribution), two consumers (ValueBidder at auction root; `judplay` 1-ply argmax-EV
play). Goal (Jason): beat `net:wp+lens:ev` n=10 with the same method (realized-value
pricing + self-play loop). Stated before the round-0 corpus, training, or A/B exist:

## JP1 — round-0 full stack loses, legibly

judbid+judplay (trained on lens:ev-played corpus — judplay's own play is off-policy at
round 0) vs net:wp+lens:ev(n=10), 512 games: **loses, mark margin in [−4, −1]**.
Mechanism prediction: play errors, not bidding — the 1-ply value play at round 0 is a
student of games it never played. Diagnostic: point margin per hand ALSO negative
(unlike v0's over-bidder which won points while losing marks).

## JP2 — the bid side survives the unification

judbid alone (judbid+lens:ev vs net:wp+lens:ev, 256+ games): within ±0.4 of the
margin-head level (i.e., mark margin in [−0.2, +0.8]). The unified featurization with
empty play history should price at least as well as margin_net given the same corpus
scale; a big regression means the unified encoding broke something.

## JP3 — the loop closes most of the play gap; parity with E[Q] n=10 uncertain

After ≥4 self-play loop rounds at ≥1000 games/round (judplay generating its own
on-policy play data): the full-stack margin improves by ≥ half the round-0 deficit.
Honest prior on reaching parity-or-better with E[Q] n=10: ~35%. The bidding arc
(P1 miss → loop → beats) is the reason for hope; "play is near-ceiling" (rank-vs-price,
LAMIR null) is the reason for doubt. Falsifier for the hopeful half: no improvement
round-over-round (then 1-ply value play is mechanism-limited, not data-limited, and
v2/search is the cue).

Grading protocol: same as v0/probe — per-round A/Bs at per-round seeds, definitive
512-game measurements at reserved seeds 7000000 + 9000000, evidence committed, #33.

---

# GRADES (2026-07-06 ~03:30) + the search rung registered

- **JP1: MISS by a hair, shape CONFIRMED** — round-0 full stack −4.37 [−4.55, −4.19]
  (predicted [−4, −1]); point margin −6.10/hand negative as predicted (play channel,
  not bidding).
- **JP2: MISS, wrong bar** — bid-only −1.09 [−1.50, −0.69] vs registered [−0.2, +0.8].
  The bar compared a round-0 head to the loop-matured head_8; the honest apples-to-apples
  is v0's round 0 (−1.44), which the unified head BEATS at its own round 0. Encoding not
  broken; loop-maturity was the missing ingredient in the prediction.
- **JP3: FALSIFIER FIRED** — 4 loop rounds: −4.37 → −4.38 → −4.16 → −3.72 → −4.04
  (needed ≥ half the deficit closed, i.e. ≥ −2.2). The loop that dissolved the bidding
  gap does NOT close the play gap at this scale: 1-ply greedy value play over a
  hand-level-MC-trained head is **mechanism-limited** (28 decisions share one label;
  the opponent evaluates 10 sampled perfect-information worlds with a 97% oracle).
  Offense share fell 64% → 49% (the bidder adapts to its own play), made-rate 32.5% →
  ~40%: bidding-side learning works; play-side signal is too weak.

## JS1 — search above the leaves (registered pre-build)

Build (amended pre-build, 03:35): `judsearch` — the v1 spec's literal shape. For each
legal move: sample N=10 consistent worlds (eq's lift, same as lens), roll the CURRENT
TRICK to resolution inside each world with the jud head playing every seat (info-honest
featurization within the sampled world), evaluate the post-trick info-state with the same
head from the searcher's POV, average across worlds, argmax EV (defenders minimize).
No oracle anywhere — the leaf is V_realized. (First draft claimed no world sampling;
that was wrong — opponents' in-trick replies come from hidden hands, so honest lookahead
needs the belief lift. Caught before building.) Prediction: **search improves the play channel by
≥ +1.0 marks/game over greedy judplay** (same head, play-only A/B, bid fixed), because
greedy 1-ply cannot see trick resolution (who wins the count) while the leaf after trick
resolution is exactly where the head is sharpest (MAE falls root→terminal). Honest prior
on judsearch reaching lens:ev parity: ~15% — E[Q] n=10 sees distributional world
information a single determinized rollout cannot.
