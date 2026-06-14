---
title: Champion — unified belief-state player
kind: entity
first_seen: local-2026-06-09
last_updated: local-2026-06-14
status: active
phase: auction tier won (#21 +1.09 / #22 +1.29 marks-game) + #24 auction belief MEASURED WIN (+2.59pp acc); two play-side levers measured DEAD — score-conditioned play (#27, negative) and belief-weighted play sampling (#25, decisive null across both belief models + bidder regimes, closed 2026-06-14). Belief value routes to bidding/defense via self-play (#26 = live frontier). Tracker reconciled 2026-06-14: #21/#23/#24/#25 closed
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
| Belief posterior | **accuracy won, play-weighting dead** | [[gus]] belief head; [[belief-bayes-ceiling]]. **#24 auction-conditioning MEASURED WIN** (`gus/model/auction.py` + `StudentTransformerFullVoidsAuction`): auction as a side feature (winner/bid/decl, no tokenizer change) → **+2.59pp held-out belief acc** vs an identical voids control (3-corpus + capacity-controlled, [[w42-champion-auction-belief]]). **But the better belief is marks-neutral when wired into *play* world-sampling** (rung #25, `champion/play.py`): three nulls across both belief models and both bidder regimes (−0.13 / −0.17 / −0.02 marks/game, 2026-06-14, [[arena]]) — belief→play is structurally dead (play already near-oracle). The belief's value lives in bidding/defense via self-play (#26), not play reweighting |
| Mark utility | **v2** | `champion/utility.py`: `race_wp` Pascal WP table + `MarksToSeven` (now with an optional equilibrium-aware pass model `pass_q_opp`/`pass_make_rate`, default off = v1) + `score_to_utility` for play risk. Score-conditioned **play** risk measured **negative** (rung #27 v2): `ScoreConditionedLensPlay` loses to `lens:ev` −1.20 marks/game, CI [−1.69,−0.70] ([[arena]]) — risk-shaped lenses sacrifice contracts (make-rate 48.9% vs 60.2%); confirms play-risk is the wrong lever. Pass model shifts 16.7% of sampled bids toward fighting for the auction (win-rate impact unmeasured) |
| Bid-strength net | **done + wired** (rung #22) | `champion/bid_net.py`: Gus-backed corpus (`bidding_continuous.py` rewired off the retired 817k policy) distilled to a hand → (9 decl × 13 bid) p_make MLP (MAE 0.053, ECE 0.007). **Now wired into the policy** as `NetPointsEvaluator` via `GusBidder(pmake_fn=...)` (CLI `net:wp`): 0.68 ms/hand, ~1500× faster than the live Gus sim. **Beats the heuristic 85/128 (+1.29 marks/game, CI [+0.72,+1.84])** — matches/exceeds the sim bidder's own +1.09 edge, validating the distillation, and reaches notrump/doubles-trump the 8-decl sim bidder cannot ([[arena]]) |
| Contract evaluator | done twice | `forge/bidding/` (2026-01) and `gus/bidding/` (2026-04); see inventory below |
| Auction policy | v0 (two tiers) | static risk-budget `HeuristicBidder` (`arena/bidders.py`), beats bid30 58.9% under identical play ([[arena]]); **Gus-backed** `champion.GusBidder` (2026-06-12, rung #21) — min positive-utility bid over a simulated P(make) table, pluggable `MarkEV`/`MarksToSeven` utility, static prefilter, one Gus eval per hand cached. **Beats the static heuristic 84/128 (65.6%), +1.09 marks/game, 95% CI [+0.54, +1.62]** under identical oracle play — wins on make-rate (65.8% vs 55.8%) and doubles-trump access, not auction volume ([[arena]]) |
| Full-game arena | **done** | [[arena]] (2026-06-12); 192 games ≈ 150 s; `BidContext` now carries game score for score-conditioned bidding; `gus[:samples[,wp]]` CLI bidder |
| Self-play consistency | **bridge landed** (rung #26) | `arena.cli --emit-snapshots` → `forge.cli.generate_eq_from_snapshots`: arena real-auction games → oracle E[Q] corpus (declarer leads) → belief-trainable `GameRecordGPU` with the auction. One policy→corpus→belief round, proven on MPS; full iteration remains |

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

Empirically refined (2026-06-14): the ranking holds **for the auction**, not for
play. The auction tier produced every CI-excludes-zero arena win (#21 +1.09, #22
+1.29 marks/game). Both *play-side* levers measured dead — score-conditioned play
risk (#27, −1.20) and belief-weighted play sampling (#25, three nulls even with
the +2.59pp #24 belief). So "belief-weighted worlds" and "score-conditioned
utility" earn their rank **as bidding and defense inputs**, realized through
self-play (#26); applied to card play they collapse into the card-play-polish
floor. The lesson the stack keeps repeating: edge is in the auction and in how
belief feeds it, not in reweighting near-oracle play.

## Self-consistency

A bid is information only if the policy that produces bids is the policy the
belief model is trained on. The champion reaches that fixed point by self-play
iteration: arena games with the current bidder+player → retrain the belief
model on those games → re-derive the policy via belief-weighted oracle search
→ repeat. Bidding conventions emerge as equilibrium artifacts rather than
authored rules. Existing evidence for the wiring step: [[belief-co-train]]'s
q-bootstrap-belief result — belief-sampled worlds beat corpus worlds. The
unified-core framing of this target — one organ, belief conditioning the search
rather than reweighting it after, and the precise solve / oracle /
[[expected-q-value|eq]] / belief / utility vocabulary — is recorded at [[jud]].

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
4. **Belief v2** — condition the belief head on auction + play history.
   **MEASURED WIN 2026-06-13** (rung #24, [[w42-champion-auction-belief]]): the
   auction enters as an explicit **side feature** (not bid *tokens*, which would
   grow `gus/model/tokenize.py`'s vocab and break every adapter) — `auction_feature_vector`
   (per-relative-seat bid/pass/winner + winning-bid level + a declared-trump one-hot)
   → `BidsEncoder` → added to the pooled state_emb, mirroring `VoidsEncoder`
   (`StudentTransformerFullVoidsAuction`). No tokenizer change ⇒ existing adapters
   load identically; `load_gus` auto-detects via `--auction`. Vs an identical
   voids-only control on the same real-auction corpus: **+2.59pp held-out belief
   accuracy** (corpus-level mean, 95% CI [+1.41,+3.76]; consistent across 3
   independent corpora A/B/C, 11/11 seed deltas positive). A **shuffled-auction**
   capacity control sits at voids level (−0.39pp) while the real auction is +2.63pp
   ⇒ the gain is auction *information*, not the +10k BidsEncoder params. No leakage
   (verified). The better belief is **marks-neutral** under oracle play
   (−0.29/game, CI incl. 0) — #25's lesson again; #24's value is belief quality
   (feeds bidding/defense + compounds in self-play), not direct arena marks.
5. **Belief-weighted world sampling** — **investigated, decisive null; closed
   2026-06-14** (`champion/play.py BeliefLensPlay`): importance-weights the MRV
   worlds by the Gus belief posterior (`champion/belief.py`), changing only the
   marginalization. Tested across both belief models and both bidder regimes:
   play-only belief −0.13, #24 auction belief −0.17 (static bidder), #24 auction
   belief −0.02 (net bidder, varied auctions incl. notrump) — three nulls, all CIs
   include zero, make-rates identical in the decisive run ([[arena]]). The
   hypothesis that the play-side null awaited a stronger belief was **falsified**:
   #24's sharper belief (ESS ~4 vs 5–8) still does not move play marks. Two
   reasons, and the second is load-bearing: (a) card play is already near-oracle,
   so reweighting *play* worlds is card-polish-tier leverage; (b) **the arena is
   information-blind by construction** — both sides are PIMC, so the harness
   cannot reward belief/concealment value via play-marks regardless of whether it
   exists ([[champion-design-review]], caveat 1; the point Fable flagged in the
   critical review and that this null re-demonstrates). So this is *not* "belief
   is useless in play" — it is "play-marks is the wrong instrument." **#24's
   belief value routes through bidding/defense via self-play (#26), not play
   reweighting.** The mechanism stays built and unit-tested (degrades to uniform
   exactly), available for a future much-stronger belief or a defense-phase
   application.
6. **Self-play fixed point** — iterate policy ↔ belief until conventions
   stabilize. **Data bridge landed + reviewed 2026-06-13** (rung #26): the belief
   corpus had no real auction (`generate_eq_continuous` deals from a seed with an
   imposed bid), so conditioning on it teaches nothing. `arena.cli
   --emit-snapshots` now dumps every contracted hand's real deal + per-seat
   auction; `forge.cli.generate_eq_from_snapshots` runs the SAME oracle E[Q]
   generation on those deals (the declarer leads the first trick, matching real
   play) and stamps the auction onto each `GameRecordGPU` → a corpus loadable by
   `JointWorldFullDataset`. Proven end-to-end on MPS. **Full iteration run
   2026-06-14** ([[w42-champion-selfplay-fixed-point]]): a belief-conditioned
   **bidder** (`champion/belief_bidder.py` — the hypothetical-completed-auction +
   belief-weighted oracle E[Q] → P(make) → score-conditioned util-max) closes the
   loop (the corpus = f(deal, decl, bids, bidder), so only a changing bidder
   iterates it). Over 4 self-play rounds the belief-KL **converges to a stable
   fixed point** (0.116 → ~0.08 plateau, vs a measured 0.045 seed floor) — genuine
   policy↔belief iteration. But the fixed point is a **stable over-bidder**: the
   belief bidder loses to `net:wp` by ~3.4 marks/game, because the oracle's
   double-dummy P(make) exceeds achievable PIMC play (strategy fusion at the
   auction, [[pimc]]). An optimism correction (`pmake_scale=0.70`, a **tuned knob** —
   the "0.58/0.83 gap" was never computed; the real, bid-dependent gap is measured in
   `champion/optimism_meter.py` → oracle 0.64 / realized 0.52 at bid 30, and the
   realized rate falls 0.52→0.11 across bids so a global scalar is the wrong *shape*)
   cuts the loss ~a third (~−3.4→−2.2) and roughly doubles the wins (~9→~20/80) —
   confirming the diagnosis — yet the PIMC-calibrated `net:wp` stays the stronger
   bidder. So the self-play *machinery* converges; belief value is **legibility,
   not marks** (the #24/#25 lesson again). The converged calibrated belief student
   is the playable champion, exported to [[plunge]] as the `onyx` difficulty.
7. **Marks-to-7 utility** — score-conditioned bidding and play risk (ICM
   analogue; absorbs the Lens v2 design). **v2 done 2026-06-12**
   (`champion/utility.py`, `champion/play_risk.py`): `race_wp` is the WP table
   (Pascal's recursion under a neutral one-mark-per-hand race); `MarksToSeven`
   scores a contract as Δ win-probability, now with an optional equilibrium-aware
   pass baseline (`pass_q_opp`, default off). The play-risk hook
   (`score_to_utility` + `ScoreConditionedLensPlay` + the `upside_10` lens) is
   built and measured — and it **loses** to plain EV play (−1.20 marks/game, CI
   excludes zero), a clean confirmation that *play-risk* is the wrong lever
   ([[arena]]). Caveat ([[champion-design-review]], caveat 2): this tested the
   **play** phase; Fable located mark-state value at the **auction** ("bites
   hardest at the auction"), where bid thresholds shift with the score — and that
   auction-side `MarksToSeven` test is unrun, the live opportunity. (#31's
   "bid-magnitude → belief is dead" is a different channel; not the same as score
   → auction-policy.) The lever is the auction and belief; the full equilibrium
   pass baseline stays rung #26.
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

**First receipts (rung #28, 2026-06-12):** [[w42-champion-teaching-battery]] runs
the champion's own lens:ev trajectories (256 games, 7,168 decisions, 19,264 action
rows) through the ch04/ch05 tactical detectors. The champion **agrees with the
book on 3 of 6 checkable claims** — it takes the setter's pounce-count (+3.9 to
+10.0 pts on the labeled action, CI excludes zero) — and the two "contradicted"
verdicts are negative controls that confirm it *learned* Roberson's prohibition:
reckless count to the bidder (−7.7 pts) and unsafe partner donation (−9.2 pts) are
correctly valued as losing, and the champion rarely does them (obey ~0.32). The
84-endgame, multi-step, and paired-bid auction claims await a forced-scenario
corpus — the champion's self-selected positions rarely reach them. First proof
the pedagogy chain produces real receipts.

## Links

- [[forge]] · [[gus]] · [[burl]] · [[lem]] · [[w42]] — the organs
- [[w42-lens-v1-utility-head-to-head]] — EV ceiling for fixed one-step utilities
- [[w42-bookval-v1-wave2-bid-aware-atlas]] — bid-aware mark machinery + power analysis
- [[belief-bayes-ceiling]] · [[belief-co-train]] — belief evidence base
- [[pimc]] — the flaw the self-play loop and the summit address
- [[book-strategy-player]] — plan algebra; natural fit is contract plans at the auction, not play-phase overlay
- [[w42-book-claim-synthesis-and-ai-directions]] — the single-decision blind spot that started this thread
