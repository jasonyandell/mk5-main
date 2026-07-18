---
title: Champion Ladder — the rung-by-rung record
kind: trail
first_seen: 2026-07-13
last_updated: 2026-07-17
status: active
---

Which clock: a rung `#N` is the GitHub issue number in milestone **Champion**
(beads retired 2026-06; the old archive is grep-able at `.beads/issues.jsonl`).
The ladder was set by a 2026-06-09 design session and ran rungs #20–#33
(2026-06-12 → 2026-07-06), followed by the measurement program that closed its
baseline (2026-07-11 → 2026-07-13). This trail walks it
rung by rung: what each rung asked, what it measured, the verdict, and the page
holding the receipt. The player it built is summarized at [[jud]], where the
current-best-player fact lives.

## Before the ladder — the bidding inventory (pre-wiki, promoted 2026-06-09)

Contract evaluation predates the wiki and was promoted into it at the design
session:

- `forge/bidding/` (2026-01): Monte Carlo contract evaluator — hand → P(make)
  over 9 declarations × 13 bids, Wilson CIs, convergence analysis (N=500 ≈
  ±0.04), PDF poster, multi-GPU support, and a continuous corpus generator with
  a finished 365-column parquet schema (`forge/bidding/schema.py`). The corpus
  generator **never ran**; no `data/bidding-results/` existed.
- `gus/bidding/` (2026-04): second-generation evaluator with [[gus]] playing
  all four seats; batched all-trumps × N games; `find_best_bid` returns
  (trump, bid, mark swing). Seconds per hand.
- `w42/bidding_risk_budget_claim_validation/hand_eval()`: static
  [[at-risk-points]] metrics (`unique_exposed_points`, `bid_ceiling_proxy`,
  trump counts) reused across the [[w42]] bidding claim tests.
- TS-side `BeginnerAIStrategy` (`src/game/ai/`): 5-sim Monte Carlo per bid,
  fixed 0.50 threshold; elementary but wired into the playable game.

Every piece answers "P(make) if I play contract (decl, B)" — the hand in a
vacuum. None answers the live auction question: pass vs bid given partner and
opponent bids, the defensive value of the pass alternative, and the score.
Forcing bid=30 in all evals dodged all three at once.

## The design session (2026-06-09)

[[champion-design-review]] preserves Fable 5's five verbatim turns — the
critical review ("chasing fireflies"), the bidding inventory, and the forward
design that set the marginal-value ranking (auction ≫ belief-weighted worlds >
score-conditioned utility ≫ card-play polish), the build order, and the two
load-bearing caveats (the [[arena]] is information-blind; score-conditioning
belongs at the auction). The rungs below graded its predictions.

## Rung #20 — the arena (2026-06-12)

**Asked:** make "best player" a measurable sentence — full games, real
auctions, marks to 7. **Built:** [[arena]] (issue #20 closed 2026-06-12).
**First physics** (identical `lens:ev` N=10 play both sides, 192 games, base
seed 1000): the static risk-budget `heuristic` bidder beats `bid30` 58.9%
(113/192; halves 59.4% / 58.3%), **+0.78 marks/game, 95% CI [+0.28, +1.26]**.
Mechanism: hand selection — 51.1% of contracts made vs 43.3%, taking 57.6% of
auctions at mean bid 30.9. Self-play symmetry sanity ~50%; 192 games ≈ 150 s on
M5 Max MPS. The auction was the predicted high ground and the first
measurement agreed.

## Rung #21 — Gus bidder (2026-06-12)

**Asked:** does a model-backed auction policy beat the static risk budget?
**Built:** `champion/bidder.py::GusBidder` — prices each legal bid by P(make)
from `gus/bidding/simulate.py` (Gus in all four seats), scores it with a
pluggable marks utility (`MarkEV` or `MarksToSeven`), takes the cheapest bid
whose utility clears a margin, prefilters statically hopeless hands, one cached
Gus eval per hand. CLI `gus[:N[,wp]]`. **Measured** (identical oracle play both
sides, 128 games, seed 0): beats `heuristic` **84/128 (65.6%; halves 73.4% /
57.8%), +1.09 marks/game, 95% CI [+0.54, +1.62]**. Quality, not volume: fewer
auctions (49.6% offense share) at 65.8% make vs 55.8%, and `doubles-trump`
reached 40 times — a declaration the pip-only static bidder structurally cannot
make. Cost ~49 min on MPS (one Gus sim per hand — rung #22's motivation).
Results: [gus_vs_heuristic_128](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence/tree/otis-night2-2026-07-15/arena/results/gus_vs_heuristic_128) (HF, [[huggingface-assets]]).

## Rung #22 — bid-strength net (2026-06-12)

**Asked:** can the ~1 s/hand Gus sim be distilled without losing the edge?
**Built:** `champion/bid_net.py` — the 2026-01 corpus generator rewired off the
retired 817k policy onto the Gus simulator (`forge/cli/bidding_continuous.py`),
distilled to a hand → (9 decl × 13 bid) p_make MLP (test MAE 0.053, ECE 0.007,
0.012 ms/call), wired into the policy as `NetPointsEvaluator` via
`GusBidder(pmake_fn=...)`, CLI `net:wp`. **Measured** (seed 0, same protocol as #21): **85/128 (66.4%),
+1.29 marks/game, 95% CI [+0.72, +1.84]** vs the heuristic — make-rate 69.3% vs
54.1%, *more* selective (42.8% offense share), and it reaches
`notrump`/`doubles-trump` the 8-decl sim bidder cannot. 0.68 ms/hand, ~1500×
faster; full-game auction sweeps unblocked. Results:
[net_vs_heuristic_128](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence/tree/otis-night2-2026-07-15/arena/results/net_vs_heuristic_128) (HF).

## Rung #23 — bid_value threading (2026-06-12)

**Asked:** close the [[gen-fleet]] priority-1 landmine — generation-side action
selection hardcoded bid-30 thresholds. **Fixed:**
`forge/cli/generate_eq_continuous.py` threads per-seed `bid_values` into
`generate_eq_games_gpu` (recorded in each `.pt`), with a
`--bid-value 30|42|84|seed` flag. Proof: regenerating the same 8 seeds at
bid=42 vs bid=30 changes 18–24 of 28 decisions per game. The same commit fixed
the `estimator.py` 84-threshold bug (P(make 84) was measured against 84 points,
not all-42) and the `cefb617` import breakage that had left the continuous
generator unrunnable.

## Rung #24 — auction-conditioned belief (2026-06-13)

**Asked:** does conditioning the [[gus]] belief head on the completed auction
improve the posterior? **Verdict: measured win** — **+2.59pp held-out belief
accuracy** vs an identical voids-only control, consistent across three corpora,
shown by a shuffled-auction capacity control to be auction *information*, not
parameters — and **marks-neutral in play** under oracle play, the #25 lesson
foreshadowed. Receipt: [[w42-champion-auction-belief]].

## Rung #25 — belief-weighted world sampling in play (2026-06-12 → decisive null 2026-06-14)

**Asked:** the then-highest-leverage architectural slot — importance-weight the
E[Q] worlds by the belief posterior instead of averaging uniformly. **Built:**
`champion/play.py::BeliefLensPlay` over `champion/belief.py`: a world's weight
is the softmax over worlds of Σ log P(seat | tile) under the belief head, with
a uniform floor (`uniform_mix=0.1`); `compute_eq_pdf` already took per-world
weights, and a new `compute_eq_weighted_mean` weights the E[Q] mean. Seat-row
alignment is exact: the belief head's three relative-opponent classes ARE the
MRV sampler's three opponent rows.

**Measured — three nulls across both belief models and both bidder regimes:**

| belief | bidder (auction info) | result | CI |
|---|---|---|---|
| play-only voids | heuristic (128 games, seed 3000) | −0.13/game | [−0.76, +0.48] |
| #24 auction | heuristic | −0.17/game | [−0.76, +0.44] |
| #24 auction | net (varied bids, notrump 66×) | −0.02/game | [−0.58, +0.52] |

The weights are genuinely active (ESS ~5–8 of 10 for the play-only belief; the
#24 belief is sharper at ESS ~4 and still moves nothing); in the decisive
net-bidder run make-rates are identical (65.1% vs 65.1%) and the halves dead
even (48.4% / 48.4%). **Verdict: belief-weighting in the play phase is a dead
lever** — structural, not belief-quality. Two reasons, the second load-bearing:
(a) card play is already near-oracle, so reweighting play worlds is
card-polish-tier leverage; (b) the arena is **information-blind by
construction** — both sides are [[pimc]], so play-marks cannot reward
belief/concealment value regardless of whether it exists
([[champion-design-review]] caveat 1, Fable's flag). Belief value routes
through bidding/defense (#26), not play reweighting. The mechanism stays built
and unit-tested (`belief_model=None` ⇒ uniform exactly). Results (HF):
[belieflens_vs_ev_128](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence/tree/otis-night2-2026-07-15/arena/results/belieflens_vs_ev_128), `belieflens_auction_vs_ev_128/`,
`belieflens_auction_net_vs_ev_128/` (siblings at the same HF revision).

## Rung #26 — self-play fixed point (bridge 2026-06-13, full iteration 2026-06-14)

**Asked:** iterate policy ↔ belief by self-play until conventions stabilize.
**Bridge:** `arena.cli --emit-snapshots` → `forge.cli.generate_eq_from_snapshots`
turns real-auction arena games into an auction-stamped, belief-trainable oracle
E[Q] corpus (declarer leads). **Verdict:** the loop **converges to a stable
fixed point** — which is a calibrated **over-bidder**: the belief-conditioned
bidder prices contracts with double-dummy P(make) ([[strategy-fusion]]) and
loses to `net:wp` even after a tuned optimism correction. Belief value is
legibility, not marks. Receipt: [[w42-champion-selfplay-fixed-point]]. The
principled dissolution of this over-bidder is rung #32's result.

## Rung #27 — marks-to-7 utility; score-conditioned play risk (2026-06-12)

**Asked:** does score-conditioned risk shaping earn marks? **Built** (v2):
`champion/utility.py` — `race_wp` (Pascal's recursion under a neutral
one-mark-per-hand race) and `MarksToSeven` (contract as Δ win-probability, with
an optional equilibrium-aware pass baseline `pass_q_opp`/`pass_make_rate`,
default off) — and `champion/play_risk.py::ScoreConditionedLensPlay`, which
picks its lens per game from the live mark score via `score_to_utility`
(protect a lead with `cvar_10`, chase from behind with the risk-seeking
`upside_10` reverse-tail lens, hold `ev` inside a WP band of 0.15), threaded
through a new `marks`/`marks_to_win` channel on `PlayPolicy.choose`.

**Measured, play side — negative** (identical heuristic bidders, 192 games,
seed 2000): `scorelens` **loses to `lens:ev` 72/192 (37.5%; halves 36.5% /
38.5%), −1.20 marks/game, 95% CI [−1.69, −0.70]**. Mechanism: make-rate 48.9%
vs 60.2% — risk-shaped lenses trade expected contracts for tail-shaping, and
within a hand marks-optimal ≈ maximize P(make), which is score-independent.

**Measured, bid side — null:** the pass-baseline hook shifts 16.7% of sampled
bids toward fighting for the auction, but `net:wp,pass0.4` vs `net:wp` over 128
games is **−0.07 marks/game, 95% CI [−0.62, +0.46]** — the q=0.4 aggression is
calibrated near break-even (54.1% offense share at 65.6% vs 69.1% make; the two
cancel). Caveat ([[champion-design-review]] caveat 2): this tested the *play*
phase and a pass-model pilot; the auction-side score test (`MarksToSeven`
thresholds varying with the score) was never run in isolation. Rung #32's A2
finding later showed the pass hook is structurally a *denial-bidding* lever
([[w42-jud-v0]]). Results (HF): [scorelens_vs_ev_192](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence/tree/otis-night2-2026-07-15/arena/results/scorelens_vs_ev_192),
`pass_vs_nopass_128/` (sibling at the same HF revision).

## Rung #28 — teaching battery (2026-06-13)

**Asked:** does the champion's own play obey the Winning 42 tactical claims?
**Verdict:** agrees on 3 of 6 checkable ch04/ch05 claims (setter-pounce family,
CI excludes zero); the two "contradicted" rows are negative controls confirming
it *learned* Roberson's prohibitions. First proof the pedagogy chain produces
receipts. Receipt: [[w42-champion-teaching-battery]].

## Rungs #29–#31 — the gap

No wiki page carries #29 or #30; their record lives in the GitHub milestone,
not the wiki. Of #31, the wiki retains only its verdict: **"bid-magnitude →
belief is dead"** — a channel distinct from score → auction-policy, a
distinction [[champion-design-review]] keeps explicit.

## Rung #32 — jud v0, the value-native bidder (2026-07-06)

**Asked:** does pricing contracts from **realized** self-play outcomes
(`V_realized`, replacing the double-dummy oracle at the auction) dissolve the
#26 over-bidder the principled way? **Verdict:** calibration PASS, round-0
parity MISS (a legible over-bidder via a winner's-curse-on-selection channel),
self-play loop PASS — the loop carries it to statistical parity with `net:wp`.
Receipt: [[w42-jud-v0]] (with the recipe-fork and A2 denial-bidding findings).
The follow-on [[w42-plateau-probe]] then falsified the registered structural
reading of the parity plateau: 3× on-policy data per round carried
`margin:wp`(head_8) **past** `net:wp` — the first learned bidder to beat the
hand-tuned champion on marks — saturating at this capacity. The
current-best-player fact and numbers live at [[jud]].

## Rung #33 — jud v1, the one organ (2026-07-06)

**Asked:** ONE net for bid and play — does the unification hold, and does the
loop close the play gap against E[Q] n=10? **Verdict:** the unification **holds
at the auction** and is **mechanism-limited at play**: greedy 1-ply value play
is a bad move-ranker (the loop moves it zero), `judsearch` recovers two-thirds
of the gap oracle-free but not parity, and neither more worlds nor better
calibration closes the rest. The night also named the policy-conditional
pricing law. A Zeb-protocol reconfirmation (`afd4802`) held the line: eq n=10
undefeated at pure play against every learned challenger since [[zeb]].
Receipt: [[w42-jud-v1]].

## After the ladder — the measurement program (2026-07-11 → 2026-07-13)

[[partnership-wall-research]] reframed the next step as measurement before
architecture. Its stops:

- [[partnership-failure-atlas-v0]] — joins 75,079 actions / 28,000 decisions;
  falsifies the assumption that the retained archive can attribute
  current-champion failures (an instrument-sufficiency result).
- [[partnership-decision-record-v1]] — one replay-verified row per play with
  separated public/actor/context/world identities and full policy provenance;
  the [[arena]]'s `--emit-decisions` surface.
- [[world-sampler-mrv-audit]] — falsified the legacy sampler's validity
  guarantee; the repaired `uniform-completion-dp-v1` replaced it.
- [[stage-0-closure]] (2026-07-13) — **Stage 0 closes**: C0 reproduces on two
  held-out blocks on the repaired sampler, the P0 symmetry and challenger
  re-grades land in band, CUDA correctness passes, and the legacy defect's harm
  is quantified (real, rare, tail-bound, not load-bearing).

With the baseline trustworthy, [[research-lane-selection]] (2026-07-13)
selected the next experiments without promoting an architecture:

- **Lane A** — [[auction-decoder]]: v0 instrument validated at
  [[auction-decoder-v0]] (auctions decode the hand exactly when the bidder
  consults it; consumers unbuilt; enriched-bid corpus named next).
- **Lane B** — [[jud-target-granularity]]: per-move targets at v1 capacity are
  a **marks null in both forms** (parent-side dense auxiliary and child-state
  values; 3× volume moves calibration only). Surviving residuals: the
  capacity×target interaction, on-policy loop data, opponents-in-rollout.

## Where the ladder stands

The current best player is jud v0's value-native bidder over oracle play —
the fact, numbers, and reproduction live at [[jud]]. The wall's precise
coordinates live at [[the-wall]]; promotion of any successor gates through
[[partnership-research-gates]]; the surviving search hypotheses are
[[belief-weighted-jud-mcts]] and [[convention-aware-blueprint-search]]. The
pattern the ladder kept repeating: every CI-excludes-zero win was an auction
lever (#21, #22, #32), every null or negative was a play-phase lever
(#25, #27, #33 play, Lane B) — the marginal-value ranking confirming itself
from the bottom up.

**Above the ladder (2026-07-17):** [[walt]] — exact best response in the
information-set game at ≤4 tiles — is the first play-side lever in this
record whose CI excludes zero: **+3.002 [+2.769, +3.219] marks/game** over
jud's greedy play head (n=512 paired), with the honest caveat that the
opponents WERE its field model. Its W1−W0 belief ablation, +0.283 [+0.184,
+0.391], grades MIGHT-#3 — term 2 carries 84.9% of the edge; term 1 is
only cashable through term 2, the synthesis with the #24/#25 nulls. Two
gates before any promotion: the `lens:ev` transfer test (#72) and the
[[lamir1-ceiling]] scar probe (#73). Receipts: [[walt]]; the machine:
[[walt-spec]].
