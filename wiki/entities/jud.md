---
title: Jud — the unified belief-state player
kind: entity
first_seen: 2026-06-14
last_updated: 2026-07-17
status: active
phase: 2026-07-13 — v0 and v1 built and graded; the v0 value-native bidder over oracle play is the current best player (fact below, reproduced at [[stage-0-closure]]). v1's play half is mechanism-limited; per-move targets at v1 capacity graded marks-null ([[jud-target-granularity]]), narrowing v2's residuals to capacity×target, on-policy loop data, and opponents-in-rollout. Surviving search consumer — [[belief-weighted-jud-mcts]].
---

## What it is

Jud is the project's player: one organ that bids and plays as the *same act*,
by conditioning search on a learned belief and training that belief by
self-play over whole games. It is the unification of pieces that existed as
separate artifacts — the [[forge]] oracle, [[gus]], the bidding evaluators, the
mark-utility machinery, and the [[w42]] concept vocabulary — and the same
object, read from the other side, is the teacher (the teaching half is a
declared side benefit; [[the-wall]] carries it). It began as a direction and a
vocabulary (2026-06-14); its first two slices are now built and graded — the
value-native bidder ([[w42-jud-v0]]) and the one-organ bid+play net
([[w42-jud-v1]]). The [[champion-ladder]] built the pieces as separable
artifacts; jud is the conception in which they become one thing, and a
commitment to precise words so they stop being conflated.

Naming (names doctrine): this push was first named **[[champion]]**,
aspirationally, before the artifact existed. The wiki page for the player is this one; the old
name persists on two BUILT artifacts — the GitHub milestone **Champion** (the
action ladder) and the repo directory `champion/` (utility, margin/jud nets,
decoders).

At every decision, bid or play: (1) maintain a posterior over the 21 hidden
tiles conditioned on all evidence — auction, plays, failures to follow suit;
(2) sample worlds from that posterior, not uniformly over consistent worlds;
(3) evaluate each world with the solve/oracle ([[expected-q-value]]) or its
student ([[gus]]) and marginalize; (4) choose under marks-to-7 win probability
conditioned on the score. This is the belief-conditioned-search architecture
behind the strongest bridge, Skat, and poker programs; Texas 42 is small for
the class (7 tricks, ~4×10⁸ worlds at deal collapsing rapidly with voids), so
near-equilibrium play is a realistic target. The residual [[pimc]] flaw
([[strategy-fusion]], information value) is mitigated by self-play consistency
and — optionally, the summit — depth-limited subgame re-solving on late tricks.

## Vocabulary (use these words precisely)

"Oracle" is colloquially overloaded; this project's pieces are specific.

- **solve** — perfect-information Texas 42, solved exactly: per deal, the true value
  of every move by exhaustive backward induction over the enumerated game
  (`forge/oracle/solve.py`, [[forge]]). Ground truth. *Not* "the oracle."
- **oracle** — the fast distillation of the solve into a value network
  (`domino-qval-large-3.3M`, qmae 0.94, ≈97% accurate). The runtime stand-in for the
  solve. solve and oracle share one **type** — the value of a *complete* world — and
  therefore one wall.
- **the brick wall** — because solve and oracle require a complete world, neither can
  be applied to a hidden-information decision directly. This is not a defect to
  repair; it is what a perfect-information value *is*: a function of perfect
  information.
- **eq** — the lift past the wall ([[expected-q-value]]): sample worlds consistent
  with what is known, evaluate each with the oracle, keep the spread. Its output per
  action is a **distribution** over outcomes — the honest object, because under hidden
  information the value of an action genuinely *is* a distribution, not a number. eq
  is the correct bridge. Its limit: it draws worlds by *consistency only*, conditions
  on no belief, and does not learn or grow.
- **the blob** — eq's per-action distribution. Real, but *melted*: smeared across
  every consistent world at equal weight. The melted blob IS [[candlewax]] — the
  Burl-era name for the same multi-peaked object, measured at founding (2026-01-06)
  and carrying the project's central consumption question ("distill it — for what
  purpose?"); see that page's concordance and "the wall, stated precisely."
- **belief** — a learned, conditioned weighting over worlds, P(world | all evidence).
  It exists (the [[gus]] belief head; auction-conditioned at
  [[w42-champion-auction-belief]]) but as a *post-hoc reweight* it measured
  marks-neutral in play (#25) — a result [[champion-design-review]] reframes as the
  [[arena]] being information-blind, not the belief inert ([[pimc]],
  [[belief-bayes-ceiling]]).
- **utility** — the collapse of a blob to a scalar to choose by (EV, p_make,
  marks-to-7). EV is the established winner for fixed one-step utilities
  ([[w42-lens-v1-utility-head-to-head]], +5.42 pts/hand). This half is settled.

## The idea

Belief belongs **inside the search**, not as a reweight applied after it.
Conditioning eq's worlds on belief sharpens the blob onto the worlds actually in
play; a utility then reads a shape instead of a smear. The same operation at every
depth makes **bidding and play one act** — a decision over a belief at the root is a
bid; the same decision deeper is a play; the same on the other side is defense.
Belief carries information when the opponents *inside a rollout* also update belief
from actions — the only place signaling can exist. And the belief that conditions
the search is trained on the whole games the search produces, so it learns toward a
self-consistent fixed point — bidding conventions emerge as equilibrium
artifacts rather than authored rules ([[champion-design-review]] forward
design; [[belief-conditioned-self-play]]).

A session sketch (`scratch/jud_demo/`, uncommitted, 2026-06-14) rendered one
mid-game position's eq blob melted (128 worlds, uniform) and belief-weighted: even
the weak post-hoc reweight un-melts the comb — on the headline action, world ESS
128 → 10.5 and the contract's p_make 0.20 → 0.61 — and it sharpens toward *truth*,
not optimism. The #25 belief value, seen directly in the distribution rather than
through the information-blind [[arena]] that could not score it.

## What the solve is, and is not

Solving perfect-information 42 is real, rare, and load-bearing: ground truth, a
bootstrap, and an exact referee. It is **not** the player — it lives on the
perfect-information axis; the player's problem is hidden information. The
over-bidding measured at [[w42-champion-selfplay-fixed-point]] would survive a 100%
solve: it is the gap between perfect-information value and achievable
hidden-information play ([[strategy-fusion]], [[pimc]]). So jud's value is *not*
the oracle distilled harder — its target is realized whole-game outcomes
(belief-state value), with the solve and oracle as bootstrap and referee.

Value-native pricing was the 2026-06-14 session's extension of Fable's written
design; a Fable 5 session (2026-07-05) endorsed it for the pricing path, with
[[rank-vs-price]] as the mechanism — play consumes rankings (PIMC optimism cancels
in argmax), bids consume prices (it lands whole). Provenance details:
[[belief-conditioned-self-play]], [[champion-design-review]].

## Asset map

| Organ | Status | Where |
|---|---|---|
| Exact perfect-info value | **done** | [[forge]] oracle; [[expected-q-value]] |
| Fast student | **done** | [[gus]] v3-10k, 0.551 regret; 0.49 with routing ([[blunder-detector]]) |
| One-step utility ceiling | **done** — EV wins | Lens(ev); [[w42-lens-v1-utility-head-to-head]] |
| Belief posterior | **accuracy won, play-weighting dead** | +2.59pp auction conditioning ([[w42-champion-auction-belief]], #24); belief-weighted *play* sampling a decisive null (#25, [[champion-ladder]]); [[belief-bayes-ceiling]] |
| Mark utility | **v2**; play-risk measured negative (#27) | `champion/utility.py` (`race_wp`, `MarksToSeven`); rung receipts on [[champion-ladder]] |
| Bid pricing | **value-native** | `net:wp` distillation (#22) → `V_realized`/`margin:wp` ([[w42-jud-v0]]) |
| Auction policy | **ladder of four** | `heuristic` → `GusBidder` (#21) → `net:wp` (#22) → `ValueBidder` (v0); receipts on [[champion-ladder]] |
| Full-game arena | **done** | [[arena]]; decision provenance via [[partnership-decision-record-v1]] |
| Self-play loop | **converges; value-native form wins** | #26 fixed point ([[w42-champion-selfplay-fixed-point]]) → v0 loop ([[w42-jud-v0]]) |
| Unified bid+play organ | **built; play mechanism-limited** | [[w42-jud-v1]]; per-move targets null at v1 capacity ([[jud-target-granularity]]) |

## Why the auction dominates

Once card play is near-double-dummy, remaining edge in trick-taking games
concentrates in auction accuracy and belief quality (the bridge lesson). The
stack's card play is already near-oracle (0.49–0.55 regret). Marginal-value
ranking: **auction ≫ belief-weighted worlds > score-conditioned utility ≫
card-play polish.** Empirically the ranking holds **for the auction**, not for
play: every CI-excludes-zero arena win is an auction lever (#21, #22, v0's
#32), and every play-side lever measured dead — score-conditioned play risk
(#27), belief-weighted play sampling (#25), greedy value play and per-move
targets at v1 capacity (#33, [[jud-target-granularity]]). So belief and utility
earn their rank **as bidding and defense inputs**, realized through self-play;
applied to card play they collapse into the card-play-polish floor.
[[rank-vs-price]] carries the mechanism; [[champion-ladder]] carries the
receipts.

## Current best player

**`margin:wp`(head_8) + `lens:ev`** — jud v0's value-native bidder
(`champion/margin_net_r8.pt`, CLI `margin:wp`) over oracle E[Q] play. It beats
the prior hand-tuned champion `net:wp + lens:ev` by **+0.38 [+0.09, +0.67]**
(reserved seed block) and **+0.42 [+0.12, +0.72]** (fresh block), 512 paired
games each ([[w42-plateau-probe]]) — the first learned bidder to beat the
hand-tuned one on marks. The promotion is **reproduced on the repaired
sampler** ([[stage-0-closure]], 2026-07-13): `+0.385 [+0.102, +0.668]` and
`+0.486 [+0.199, +0.775]` on the same two blocks with full `--emit-decisions`
fingerprints. The demonstrated advantage is a **bidding gain**; `lens:ev`
(eq n=10) remains the strongest measured play policy, undefeated against every
learned challenger since [[zeb]] (Zeb-protocol reconfirmation, 2026-07-06:
`judsearch` −1.39, `judplay` −2.73, both losing to `lens:ev`; a bonus pilot put
`margin:wp`(r8) at +0.59 [−0.19, +1.39] over the live non-distilled `gus:10,wp`
sim bidder — the bidding crown wasn't hiding behind the distillation;
[[w42-jud-v1]]). The gus pilot is **confirmed at n=128** (2026-07-17, table42
worktree, fresh block base_seed=0): `margin:wp`(r8)+`lens:ev` beats
`gus:10,wp`+`lens:ev` **+0.62 [+0.06, +1.23] marks/game**, 58.6% game win,
halves 57.8%/59.4% — CI excludes zero; the learned head beats the simulation
bidder it distilled past (`scratch/table42/jud_vs_gus/n128/`).

## The build, graded

- **v0 — the value-native bidder ([[w42-jud-v0]], 2026-07-06).** One added head:
  `V_realized` (`champion/margin_net.py`) prices contracts from realized 4-seat
  self-play outcomes; the bidder reads tail mass through `MarksToSeven`; play
  stays `lens:ev`; `pmake_scale` retires. Graded: **calibration PASS** (6×
  closer to realized than oracle), **round-0 parity MISS** (a legible
  over-bidder — optimism returning through *selection*, the winner's curse),
  **loop PASS** (four rounds dissolve the over-bidding to statistical parity).
  Belief stays implicit in v0 (`V_realized` conditioned on the auction learns
  what the belief would say), and the head trains on defend-side info-states
  too — pricing the pass counterfactual from data, closing the
  suppose-opponent-wins hole [[w42-champion-selfplay-fixed-point]] flagged as
  OOD for the oracle path. [[w42-plateau-probe]] then broke the plateau — data
  starvation, not the [[pimc]] price — producing the current best player
  above, and saturating at ≈ +0.3–0.4 marks/game at this capacity.
- **v1 — the one organ ([[w42-jud-v1]], 2026-07-06).** `champion/jud_net.py`
  extends `V_realized` to every decision (bid-time = play-time with an empty
  history); the v0 bidder consumes it with zero adapter; `judplay` is greedy
  1-ply value play, oracle-free at runtime. Graded: **the unification holds at
  the auction** (the unified head beats v0's own round-0 bidder) and is
  **mechanism-limited at play** — greedy value play is a bad move-ranker (the
  loop moves it zero), `judsearch` recovers two-thirds of the gap
  (−3.44 → −1.16, +2.28) but not parity, and neither more worlds nor a
  better-calibrated head closes the rest. The stack reached −1.43 from −4.37
  oracle-free in one night; the wall is per-move discrimination.
- **Field report — the table42 auction-calibration case (2026-07-16→17,
  [[table42]], [issue #66](https://github.com/jasonyandell/mk5-main/issues/66)).**
  First live-table instance of the head's auction claims being auditable
  against a replayed seat. At game night 1, the v1 head (ValueBidder over
  `jud_net.pt`) bid 31-in-fours on two trumps missing the boss, claiming
  **P(make)=0.68**; a 1000-world replay of the exact information set under
  jud self-play — the head's *own field*, so no field-mismatch excuse —
  measured **0.273** (median outcome: set; the realized 39-3 make was
  ~p90, carried by a partner who had passed holding 4-4+5-5). Predictions
  were registered before the run: jud 0.68, Jason-by-gut 0.20, Claude 0.4x
  — the human beat the head 6×. Nuance for the v0 loop-PASS above: the
  loop dissolved over-bidding to *aggregate* parity; this seat shows
  per-seat claims can remain wildly inflated inside an aggregate-calibrated
  head. Probe: `scratch/table42/probe66.py` (table42 worktree). The follow-up
  grid (`probe66_grid.py`) sharpened it three ways: the inflation is
  **hand-level, not suit-level** (majority-make claimed in *every* pip
  suit; blanks claimed 0.60 / measured 0.08); the fours declaration was a
  **0.003 claimed-tie broken across a 0.17 measured chasm** (sixes measures
  0.448 — the human's promotion reasoning beat the head's suit choice);
  and **notrump is calibrated (−0.01) while every pip suit inflates**,
  localizing the optimism by contract type. Remaining follow-ups on the
  issue.
  **Jason's conjecture (MIGHT, deciding probe on #66):** belief-shaping to
  a self-play fixed point can converge to an internally-consistent,
  externally-nonsense fixed point; the loop needs *outside* calibration
  anchors — e.g. a no-beliefs floor (never price a seat above what a
  u-consistent replay supports without evidence) — to prevent the failure
  mode. This is [[belief-policy-value-algebra]] CAN-#4 read as a repair
  path: V-error is repairable by evaluation data alone, and u-replays are
  exactly such data, cheap (this one took 6 seconds).
- **v2's residuals ([[jud-target-granularity]], 2026-07-13).** The named cue —
  per-move targets — is **graded a marks null at v1 capacity in both forms**:
  the parent-side dense E[Q] auxiliary triples in-distribution ranking and
  carries it to no consumer (its direct reader plays 0.9 marks *worse*), the
  consumer-aligned child-state form is also null, and 3× corpus volume moves
  calibration only. What survives for v2: the **capacity×target interaction**,
  **on-policy loop data** (r4's five-round cumulative corpus out-ranks fresh 3×
  — distributional, not volumetric), and **opponents-in-rollout** (untested).

### The laws the build established

- **Factorization law.** Learn what is unknown; compute what is exact. The
  hand-outcome distribution under the real policy is learned (`V_realized`);
  the marks race is computed (`race_wp`, Pascal). Do not learn marks-to-7; do
  not hand-tune hand outcomes.
- **Policy-conditional pricing law ([[w42-jud-v1]]).** `V_realized` prices
  honestly **only for the policy that generated its corpus** — pair the head
  with a better player and its prices go stale. Consumers cannot be mixed and
  matched; a value head must be retrained when the policy it prices changes.
- **The two per-move signals (do not conflate).** A *dense ranking auxiliary*
  (per-action E[Q] distilled from the oracle; consumer = move ranking) is a
  different signal from a *primary search value* (per-action realized
  continuation under an information-honest rollout policy; consumer = the
  search leaf). Teaching a leaf that future actors receive double-dummy
  information repeats the #26 over-bidder in play form. Both forms are graded
  at [[jud-target-granularity]]; the surviving search consumer for any future
  per-move leaf is [[belief-weighted-jud-mcts]] (J0–J4 decomposition).
- **The referee instrument.** Per position, oracle EV − `V_realized` EV = the
  price of hidden information (`champion/optimism_meter.py` is its static
  ancestor). When later rungs put the model inside the rollouts, this gap
  narrates conventions emerging: signaling is exactly what moves realized value
  toward double-dummy. Built by v0; not yet read as a live instrument.

## Links

- [[champion-ladder]] — the rung record (#20–#33) that built the pieces; the
  era's milestone name, Champion, persists on GitHub
- [[the-wall]] — the goal this player is graded against; also home of the
  teaching half (declared side benefit)
- [[w42-jud-v0]] · [[w42-plateau-probe]] · [[w42-jud-v1]] ·
  [[jud-target-granularity]] — the graded slices
- [[partnership-wall-research]] · [[partnership-research-gates]] — the
  measurement program and the gates before any v2-shaped build
- [[belief-weighted-jud-mcts]] — surviving search-consumer hypothesis
- [[rank-vs-price]] — why value-native is mandatory at the auction, optional in
  play; [[champion-design-review]] · [[belief-conditioned-self-play]] —
  provenance
- [[w42-book-second-pass]] — the signaling/concealment queue that is the v2
  opponents-in-rollout target
- [[otis]] — the fate-ledger sibling (2026-07-15): jud's target-granularity
  question asked with per-hand fate structure instead of per-move values;
  reached incumbent parity at v0 with a decisive calibration win
  ([[otis-v0]])
