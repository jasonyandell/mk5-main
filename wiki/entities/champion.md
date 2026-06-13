---
title: Champion — unified belief-state player
kind: entity
first_seen: local-2026-06-09
last_updated: local-2026-06-12
status: active
phase: rungs #20-#23, #27 v2, #25 landed (2026-06-12); play-risk (wrong lever) + belief-weighting (null, pending stronger belief) measured; #24 auction-conditioned belief is the next unlock
---

## What it is

The champion is the unification target for the project's player work: one
belief-state player that bids and plays full games to 7 marks, built from
pieces that already exist as separate artifacts. The reframe (2026-06-09): the
[[forge]] oracle, [[gus]], the pre-wiki bidding evaluators, the mark-utility
machinery, and the [[w42]] concept vocabulary are stapled together today; the
champion is the architecture that makes them one thing — and the same object,
read from the other side, is the teacher.

Action ladder lives in GitHub issues, milestone **Champion**
(beads retired 2026-06; historical beads remain readable in
`.beads/issues.jsonl`).

## Decision loop

At every decision, bid or play:

1. Maintain a posterior over the 21 hidden tiles conditioned on **all**
   evidence — auction history, every play, every failure to follow suit.
2. Sample worlds from that posterior, not uniformly over consistent worlds.
3. Evaluate each world with the perfect-info oracle ([[expected-q-value]]) or
   its student ([[gus]]); marginalize.
4. Choose under marks-to-7 win probability conditioned on the score, not raw
   points and not fixed p_make.

This is the belief-conditioned-search architecture behind the strongest
bridge, Skat, and poker programs. Texas 42 is small for the class (7 tricks,
~4×10⁸ worlds at deal collapsing rapidly with voids), so near-equilibrium play
is a realistic target, not a romantic one. The residual [[pimc]] flaw
(strategy fusion, information value) is mitigated by self-play consistency and
— optionally, the summit — depth-limited subgame re-solving on late tricks,
where 42's endgames are small enough to solve exactly at the information-set
level.

## Asset map

| Organ | Status | Where |
|---|---|---|
| Exact perfect-info value | **done** | [[forge]] oracle; [[expected-q-value]] |
| Fast student | **done** | [[gus]] v3-10k, 0.551 regret; 0.49 with routing ([[blunder-detector]]) |
| One-step utility ceiling | **done** | Lens(ev); [[w42-lens-v1-utility-head-to-head]] |
| Belief posterior | wired (null) | [[gus]] belief head; play-evidence only; [[belief-bayes-ceiling]]. **Now wired into world sampling** (rung #25, `champion/belief.py`+`play.py`): importance-weights MRV worlds by the posterior, ESS-active (~5–8/10) but measured **null** in the arena (−0.13 marks/game, CI includes zero) — the play-evidence belief is too weak. **not** auction-conditioned — that's the #24 unlock |
| Mark utility | **v2** | `champion/utility.py`: `race_wp` Pascal WP table + `MarksToSeven` (now with an optional equilibrium-aware pass model `pass_q_opp`/`pass_make_rate`, default off = v1) + `score_to_utility` for play risk. Score-conditioned **play** risk measured **negative** (rung #27 v2): `ScoreConditionedLensPlay` loses to `lens:ev` −1.20 marks/game, CI [−1.69,−0.70] ([[arena]]) — risk-shaped lenses sacrifice contracts (make-rate 48.9% vs 60.2%); confirms play-risk is the wrong lever. Pass model shifts 16.7% of sampled bids toward fighting for the auction (win-rate impact unmeasured) |
| Bid-strength net | **done** (rung #22) | `champion/bid_net.py`: Gus-backed bidding corpus (`forge/cli/bidding_continuous.py` rewired off the retired 817k policy) distilled to a hand → (9 decl × 13 bid) p_make MLP. Test MAE 0.053, ECE 0.007, 0.012 ms/call — the <1ms replacement for the live Gus sim the `GusBidder` pays per hand |
| Contract evaluator | done twice | `forge/bidding/` (2026-01) and `gus/bidding/` (2026-04); see inventory below |
| Auction policy | v0 (two tiers) | static risk-budget `HeuristicBidder` (`arena/bidders.py`), beats bid30 58.9% under identical play ([[arena]]); **Gus-backed** `champion.GusBidder` (2026-06-12, rung #21) — min positive-utility bid over a simulated P(make) table, pluggable `MarkEV`/`MarksToSeven` utility, static prefilter, one Gus eval per hand cached. **Beats the static heuristic 84/128 (65.6%), +1.09 marks/game, 95% CI [+0.54, +1.62]** under identical oracle play — wins on make-rate (65.8% vs 55.8%) and doubles-trump access, not auction volume ([[arena]]) |
| Full-game arena | **done** | [[arena]] (2026-06-12); 192 games ≈ 150 s; `BidContext` now carries game score for score-conditioned bidding; `gus[:samples[,wp]]` CLI bidder |
| Self-play consistency | **missing** | — |

## Bidding inventory (pre-wiki work, promoted 2026-06-09)

Contract evaluation predates the wiki and was never promoted until now:

- `forge/bidding/` (2026-01): Monte Carlo contract evaluator — hand →
  P(make) over 9 declarations × 13 bids, Wilson CIs, convergence analysis
  (N=500 ≈ ±0.04), PDF poster, multi-GPU support, and a continuous corpus
  generator with a finished 365-column parquet schema
  (`forge/bidding/schema.py`). The corpus generator **never ran**; no
  `data/bidding-results/` exists.
- `gus/bidding/` (2026-04): second-generation evaluator with Gus playing all
  four seats; batched all-trumps × N games; `find_best_bid` returns
  (trump, bid, mark swing). Seconds per hand.
- `w42/bidding_risk_budget_claim_validation/hand_eval()`: static
  [[at-risk-points]] metrics (`unique_exposed_points`, `bid_ceiling_proxy`,
  trump counts) reused across the w42 bidding claim tests.
- TS-side `BeginnerAIStrategy` (`src/game/ai/`): 5-sim Monte Carlo per bid,
  fixed 0.50 threshold; elementary but wired into the playable game.
- ~~Known landmine ([[gen-fleet]] priority 1): `bid_value` not plumbed
  through generation action selection.~~ **Fixed 2026-06-12 (rung #23):**
  `forge/cli/generate_eq_continuous.py` now threads per-seed `bid_values`
  into `generate_eq_games_gpu` (and records them in each `.pt`), with a
  `--bid-value 30|42|84|seed` flag. Proof: regenerating the same 8 seeds at
  bid=42 vs bid=30 changes 18–24 of 28 decisions per game. The same commit
  fixed the `estimator.py` 84-threshold bug (P(make 84) was measured against
  84 points, not all-42) and the `cefb617` import breakage that had left the
  continuous generator unrunnable.

Every piece answers "P(make) if I play contract (decl, B)" — the hand in a
vacuum. None answers the live auction question: pass vs bid given partner and
opponent bids, the defensive value of the pass alternative, and the score.
Forcing bid=30 in all evals dodges all three at once.

## Why the auction dominates

Once card play is near-double-dummy, remaining edge in trick-taking games
concentrates in auction accuracy and belief quality (the bridge lesson). The
stack's card play is already near-oracle (0.49–0.55 regret); no auction
exists. Marginal-value ranking for the champion:

**auction ≫ belief-weighted worlds > score-conditioned utility ≫ card-play polish.**

## Self-consistency

A bid is information only if the policy that produces bids is the policy the
belief model is trained on. The champion reaches that fixed point by self-play
iteration: arena games with the current bidder+player → retrain the belief
model on those games → re-derive the policy via belief-weighted oracle search
→ repeat. Bidding conventions emerge as equilibrium artifacts rather than
authored rules. Existing evidence for the wiring step: [[belief-co-train]]'s
q-bootstrap-belief result — belief-sampled worlds beat corpus worlds.

## Build ladder

1. **Arena** — full games: auction + play, marks to 7; paired-seed team
   rotation like `w42/lens_v1/parallel_match.py`. The measuring stick; "best
   player" is not a measurable sentence without it. **Done 2026-06-12**
   ([[arena]]): first physics — a static risk-budget bidder beats
   always-bid-30 by +0.78 marks/game under identical oracle play.
2. **Auction v0** — Roberson risk-budget policy over `gus/bidding`
   (anchor: ch02 bid-only-enough, `supported` at wave 2.B.2). **Done
   2026-06-12** (`champion/bidder.py`): `GusBidder` takes the minimum
   positive-utility legal bid over a Gus-simulated P(make) table, declares
   the trump maximizing P(make) at the contract threshold, and prefilters
   statically hopeless hands before paying for simulation. Wired into the
   arena CLI as `gus[:samples[,wp]]`; beats the static heuristic 84/128
   (65.6%, +1.09 marks/game, CI excludes zero) under identical oracle play.
3. **Bid-strength net** — **done 2026-06-12** (`champion/bid_net.py`): the
   2026-01 corpus generator, rewired off the retired 817k policy onto the Gus
   simulator (`forge/cli/bidding_continuous.py`), ran a scaled Gus-backed
   corpus; a small MLP distills hand → (9 decl × 13 bid) p_make at 0.012 ms/call
   (test MAE 0.053, ECE 0.007). The <1ms replacement for the live Gus sim the
   bidder pays per hand. The 2026-01 `estimator.py` 84-threshold fix landed with
   rung #23.
4. **Belief v2** — condition the belief head on auction + play history
   (training data free from arena self-play); revisit [[belief-bayes-ceiling]]
   with auction evidence.
5. **Belief-weighted world sampling** — **mechanism landed 2026-06-12**
   (`champion/play.py BeliefLensPlay`): importance-weights the MRV worlds by the
   Gus belief posterior (`champion/belief.py`), changing only the marginalization
   (`compute_eq_pdf` already took weights; `compute_eq_weighted_mean` added; the
   belief↔world seat rows align exactly). Measured **null** with the weak
   play-evidence belief (−0.13 marks/game, CI [−0.76, +0.48]; ESS ~5–8/10 confirms
   the weights are active) — the highest-leverage *slot* is wired and validated,
   and the win awaits the stronger auction-conditioned belief of rung #24
   ([[arena]]). One change will then improve bidding, play, and defense together.
6. **Self-play fixed point** — iterate policy ↔ belief until conventions
   stabilize.
7. **Marks-to-7 utility** — score-conditioned bidding and play risk (ICM
   analogue; absorbs the Lens v2 design). **v2 done 2026-06-12**
   (`champion/utility.py`, `champion/play_risk.py`): `race_wp` is the WP table
   (Pascal's recursion under a neutral one-mark-per-hand race); `MarksToSeven`
   scores a contract as Δ win-probability, now with an optional equilibrium-aware
   pass baseline (`pass_q_opp`, default off). The play-risk hook
   (`score_to_utility` + `ScoreConditionedLensPlay` + the `upside_10` lens) is
   built and measured — and it **loses** to plain EV play (−1.20 marks/game, CI
   excludes zero), a clean confirmation that play-risk is the wrong lever
   ([[arena]]). The lever is the auction and belief; the full equilibrium pass
   baseline stays rung #26.
8. **Summit (optional)** — depth-limited subgame re-solving on late tricks;
   exact information-set endgame solving.

## Teaching half

A maximally strong player is mute; the [[w42]] campaign built the concept
vocabulary that makes it legible. Run the champion through the detector
battery: where the book is right, where it is wrong, and what to do instead —
with receipts ("Roberson says double ahead of your off; the champion agrees
84% of the time, and the 16% has a pattern"). [[burl]] narrates in the
Roberson idiom ([[at-risk-points]], [[post-commit-q-and-a]]). The distillation
chain **oracle → champion → gus → burl → lem** is also the pedagogy chain:
each level explains the one above to the one below. Target artifact: a
data-validated strategy guide — *Winning 42, second edition*.

## Links

- [[forge]] · [[gus]] · [[burl]] · [[lem]] · [[w42]] — the organs
- [[w42-lens-v1-utility-head-to-head]] — EV ceiling for fixed one-step utilities
- [[w42-bookval-v1-wave2-bid-aware-atlas]] — bid-aware mark machinery + power analysis
- [[belief-bayes-ceiling]] · [[belief-co-train]] — belief evidence base
- [[pimc]] — the flaw the self-play loop and the summit address
- [[book-strategy-player]] — plan algebra; natural fit is contract plans at the auction, not play-phase overlay
- [[w42-book-claim-synthesis-and-ai-directions]] — the single-decision blind spot that started this thread
