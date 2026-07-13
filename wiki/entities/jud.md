---
title: Jud — the unified belief-conditioned core
kind: entity
first_seen: 2026-06-14
last_updated: 2026-07-13
status: active
phase: 2026-07-13 — v0 and v1 built and graded; the v0 value-native bidder over oracle play is the current best player (fact below, reproduced at [[stage-0-closure]]). v1's play half is mechanism-limited; per-move targets at v1 capacity graded marks-null ([[jud-target-granularity]]), narrowing v2's residuals to capacity×target, on-policy loop data, and opponents-in-rollout. Surviving search consumer — [[belief-weighted-jud-mcts]].
---

## What it is

Jud is the name for the unified belief-conditioned core the [[champion]] work
points at: one organ that bids and plays as the *same act*, by conditioning
search on a learned belief and training that belief by self-play over whole
games. It began as a direction and a vocabulary (2026-06-14); its first two
slices are now built and graded — the value-native bidder ([[w42-jud-v0]]) and
the one-organ bid+play net ([[w42-jud-v1]]). The [[champion-ladder]] built the
pieces as separable artifacts; jud is the conception in which they become one
thing, and a commitment to precise words so they stop being conflated.

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
self-consistent fixed point ([[champion]] self-consistency;
[[champion-design-review]] forward design; [[belief-conditioned-self-play]]).

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
[[w42-jud-v1]]).

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

- [[champion]] — the player-and-teacher this is the core for;
  [[champion-ladder]] — the rung record that built the pieces
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
