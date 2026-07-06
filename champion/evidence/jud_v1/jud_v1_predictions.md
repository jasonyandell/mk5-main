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

## JS1 GRADE (04:20): PASS, +2.28 — search recovers 2/3 of the play gap

Play-only, same head (r4), same deals (seed 7200000): greedy judplay −3.44 [−3.75,−3.14]
→ judsearch:n10 **−1.16 [−1.55,−0.77]** (made 46.1% → 62.7%). Registered bar ≥ +1.0:
CLEARED at +2.28. The leaf was fine; the greedy consumer was the bottleneck. Still short
of lens:ev (CI excludes 0). No oracle anywhere in the jud side.

## JS2 — worlds sweep (registered pre-run): judsearch:n20, same seed/head.
Prediction: mild gain, +0.2 to +0.6 (world-average noise shrinks but the leaf's bias is
shared); falsifier ≥ +1.0 (then worlds were the binding constraint, push n harder).

## JS3 — search-in-the-loop (registered pre-run): one loop round where SELF-PLAY uses
judsearch:n10 both sides (~600 games), retrain cumulative, re-measure play-only + full
stack. Prediction: the policy-conditional pricing insight cuts both ways — the head
retrained on search-quality games improves BOTH its prices and its leaf, worth ≥ +0.4
on the play channel beyond JS2's config, and the full jud stack (jud bid + judsearch)
lands within [−1.0, +0.2] of net:wp+lens:ev. Honest prior on full parity tonight: ~25%.

## JS2 GRADE (04:35): BELOW BAND — n20 = −1.05 [−1.44,−0.64], gain +0.11 (predicted
+0.2–0.6). Worlds are not the constraint; the shared leaf bias is. Don't push n.

## JS3 GRADE (05:25): FALSIFIER FIRED — search-in-the-loop does not improve the leaf

One round of judsearch self-play (600 games, 6612 hands) + cumulative retrain (r5, test
CE 2.09 — best head yet on paper): play-only −1.41 [−1.79,−1.02] vs r4's −1.16 (gain
ZERO, registered ≥ +0.4); full stack jud:wp(r5)+judsearch(r5) at reserved seed 7000000:
**−1.43 [−1.68,−1.16]** (registered band [−1.0,+0.2]). Better paper calibration did not
buy better play. Interpretation: the leaf's per-move discrimination is the wall — a 470k
MLP trained on hand-level Monte-Carlo labels cannot match a 97%-accurate 3.3M
perfect-information oracle evaluated per move over sampled worlds. Capacity/signal, not
data or loop rounds. v2's cue: bigger leaf + per-move targets (e.g. distill E[Q] as an
auxiliary policy/value signal — the solve/oracle as bootstrap, per jud.md), and
opponents-in-rollout.

# NIGHT VERDICT — the best player I know how to make at this time

**margin:wp(head_8) + lens:ev** — the value-native bidder (v0 mechanism, loop-matured,
data-scaled) over E[Q] n=10 play. Beats the previous champion net:wp+lens:ev by
**+0.38 [+0.09,+0.67] / +0.42 [+0.12,+0.72]** (512 games × 2 reserved seeds). The first
learned bidder to beat the hand-tuned one. jud v1's one-organ stack reached −1.43 from
−4.37 in one night of registered rungs (organ → loop → search → search-in-loop) with
every rung graded; its bidding validates the unification, its play names the next wall.
