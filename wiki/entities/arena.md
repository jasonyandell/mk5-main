---
title: Arena — full-game harness
kind: entity
first_seen: local-2026-06-12
last_updated: pending-this-ingest
status: active
phase: landed; measuring stick for the champion ladder
---

## What it is

The full-game Texas 42 harness (`arena/`): four seats, real auctions, hands
until a team reaches 7 marks. Rung 1 of the [[champion]] ladder (GitHub
issue #20, closed 2026-06-12) — until it existed, "best player" was not a
measurable sentence, because every eval in the stack forced bid=30.

A player is a (BidPolicy, PlayPolicy) pair. Matches are paired-seed with
team rotation (half 1: A as seats {0,2}; half 2: seats {1,3}; identical
deal seeds), so card luck cancels and the auction itself becomes part of
what is measured.

```bash
python -u -m arena.cli --team-a heuristic+lens:ev --team-b bid30+lens:ev \
    --n-games 192 --n-samples 10 --device mps
```

## Design

- The zeb engine (`forge/zeb/game.py`) owns the play phase; the arena owns
  the auction ([[rules-of-42]] §Bidding: one bid each, left of shaker first, point
  bids 30–41 then marks, reshake on pass-out, forced 30 after max
  redeals), marks scoring (§Scoring: `max(1, bid // 42)` marks, set pays the
  defenders), and the game loop. A hand enters play as a fully-formed
  PLAYING state — the `_force_bid_30` construction from
  [[w42-lens-v1-utility-head-to-head]]'s harness, minus the force.
- Lockstep batching: each tick routes every live game's decision to the
  owning side's play policy as one batched call, so the GPU path amortizes
  across games (same shape as `w42/lens_v1/parallel_match.py`).
- The oracle never sees the bid value (`GameStateTensor` carries decl and
  bidder only), so auction-won contracts are in-distribution for the Q
  model; the bid feeds only the utility thresholds, which were already
  parameterized per game ([[gen-fleet]]'s `contract_threshold_bins`).
- Shipped bidders: `heuristic` (static Roberson risk-budget — bid the
  minimum legal raise while it stays within `42 − unique_exposed_points`
  for the best ≥3-tile pip trump; the wave-2.B arithmetic from
  [[w42-bookval-v1-wave2-bid-aware-atlas]], ported to `arena/hand_metrics.py`
  with an equivalence test), `bid30` (opens 30, else passes — the
  historical baseline as a live player), `random`, and `gus` (the
  model-backed `champion.GusBidder`, rung #21 — see below). Play:
  `lens:<utility>` over the forge E[Q] PDF, `scorelens[:band]` (rung #27 v2
  score-conditioned, see below), or `random`.
- `BidContext` carries the live game score (`marks`, `marks_to_win`), so a
  bidder can condition on it — the channel the marks-to-7 utility
  ([[champion]] rung #27) rides. The shipped static bidders ignore it; the
  `gus[:samples,wp]` bidder uses it.

## Gus bidder (rung #21, 2026-06-12)

`champion/bidder.py` adds the first model-backed auction policy: `GusBidder`
prices each legal bid by P(make) from `gus/bidding/simulate.py` (Gus in all
four seats, pip trumps + doubles, batched), scores it with a pluggable marks
utility (`MarkEV` score-blind, or `MarksToSeven` score-conditioned), and
takes the cheapest bid whose utility clears a margin — Roberson's "bid only
enough" with a simulated willingness number. A static prefilter skips
simulation on hands no trump structure could carry, and each hand is
evaluated once (cached across the bid and declaration). CLI: `gus[:N[,wp]]`,
where `wp` selects the marks-to-7 utility.

Headline (2026-06-12, identical oracle play both sides, lens:ev N=10,
128 games, base seed 0): **`gus` beats the static `heuristic` 84/128
(65.6%; halves 73.4% / 57.8%), mark margin +1.09/game, 95% CI
[+0.54, +1.62]** — CI excludes zero. Quality, not volume: the Gus bidder
took *fewer* auctions (49.6% offense share) but made 65.8% of its contracts
vs the heuristic's 55.8%, and reached `doubles-trump` 40 times — a
declaration the pip-only static bidder structurally cannot make. ~49 min on
M5 Max MPS (one Gus eval per dealt hand is the cost; the bid-strength net of
rung #22 is the fix). Result under `arena/results/gus_vs_heuristic_128/`.

**Net bidder (rung #22, 2026-06-12):** the distilled `net` bidder
(`champion.NetPointsEvaluator` via `GusBidder(pmake_fn=...)`, CLI `net:wp`)
replaces the ~1 s/hand Gus sim with a **0.68 ms/hand** forward pass — and
**preserves the edge: 85/128 (66.4%), +1.29 marks/game, 95% CI [+0.72, +1.84]**
vs the heuristic (seed 0, same as the gus run), make-rate 69.3% vs 54.1%. It is
*more* selective (42.8% offense share) and reaches `notrump`/`doubles-trump` the
8-decl sim bidder cannot. ~1500× faster bidding unblocks full-game auction
sweeps. Result under `arena/results/net_vs_heuristic_128/`.

## Belief-weighted world sampling — measured null (rung #25, 2026-06-12)

`champion/play.py BeliefLensPlay` is the highest-leverage architectural slot: it
keeps the validity-guaranteed MRV world sampler and the oracle E[Q] path
unchanged and changes only the marginalization — instead of averaging Q uniformly
over the sampled worlds, it importance-weights them by the Gus belief posterior
(`champion/belief.py`). A world's weight is the softmax over worlds of
Σ log P(seat | tile) under the belief head, with a uniform floor (`uniform_mix=0.1`)
to bound effective sample size. `compute_eq_pdf` already took per-world weights; a
new `compute_eq_weighted_mean` weights the E[Q] mean too. The seat-row alignment is
exact: the belief head's three classes (relative opponents P+1/+2/+3) ARE the MRV
sampler's three opponent rows, from the same current-player POV.

First pass (identical `heuristic` bidders, 128 games, seed 3000, n_samples=10):
**`belieflens:ev` vs `lens:ev` is a null — 58/128 (45.3%), mark margin
−0.13/game, 95% CI [−0.76, +0.48]** (includes zero); make-rate 54.9% vs 55.9%.
The weights are genuinely active (effective sample size ~5–8 of 10, min ~2), so
the mechanism works — but the play-evidence-only belief was too weak to move play
strength. The hypothesis at the time: the unlock is the stronger
**auction-conditioned belief (#24)**, then unbuilt.

**Tested 2026-06-14 — the unlock did not unlock (decisive null).** #24's
auction-conditioned belief landed (+2.59pp held-out accuracy), so the join was
finally run — it was always one flag away (`belieflens --gus-adapter <#24 auction
adapter>`; `load_gus` auto-detects the auction architecture). The better belief is
genuinely sharper here (ESS ~4 vs 5–8 — it concentrates weight on fewer worlds),
and it still does not move play marks, across two bidder regimes:

| belief | bidder (auction info) | result | CI |
|---|---|---|---|
| play-only voids | heuristic (low-variance bids) | −0.13/game | [−0.76, +0.48] |
| **#24 auction** | heuristic (low-variance bids) | −0.17/game | [−0.76, +0.44] |
| **#24 auction** | **net (varied bids, notrump 66×)** | **−0.02/game** | [−0.58, +0.52] |

The net-bidder run is the confound-resolver: even with informative auctions the
belief was trained to read, make-rates come out *identical* (65.1% vs 65.1%) and
the halves are dead even (48.4% / 48.4%). **Verdict: belief-weighting in the
*play* phase is a dead lever** — structural, not a belief-quality problem. With
card play already near-oracle (0.49–0.55 regret), the E[Q] averaged over uniform
consistent worlds is already near-optimal *in play*; a better posterior changes
the play decision rarely and the marks outcome not at all. #24's belief value is
real but routes through **bidding and defense via self-play (#26)**, not play
reweighting — the [[champion]] marginal-value ranking, now measured on its
play-side floor. The mechanism stays built and unit-tested
(`belief_model=None` ⇒ uniform exactly), available if a much stronger belief or a
defense-phase application ever wants it. Results under
`arena/results/belieflens_vs_ev_128/`, `belieflens_auction_vs_ev_128/`,
`belieflens_auction_net_vs_ev_128/`.

## Score-conditioned play — measured negative (rung #27 v2, 2026-06-12)

`champion/play_risk.py` adds `ScoreConditionedLensPlay`: a LensPlay that picks
its lens per game from the live mark score via `champion.utility.score_to_utility`
— protect a lead with the lower-tail-averse `cvar_10`, chase from behind with
the new risk-seeking `upside_10` lens (the reverse-cumulative top-10% tail,
mirror of `cvar_10`), hold `ev` near even (race-model WP band 0.15). The score
reaches play through a new `marks`/`marks_to_win` channel on `PlayPolicy.choose`,
threaded from `_LiveGame.marks` in the engine loop.

Headline (identical `heuristic` bidders both sides, 192 games, seed 2000,
n_samples=10): **`scorelens` LOSES to `lens:ev` 72/192 (37.5%; halves 36.5% /
38.5%), mark margin −1.20/game, 95% CI [−1.69, −0.70]** — CI excludes zero. The
mechanism is the make-rate: the score-conditioned team made only 48.9% of its
contracts vs the EV team's 60.2%. The risk-shaped lenses trade expected
contracts for tail-shaping, and in 42 the per-hand marks outcome is dominated by
EV-greedy play — within a hand, marks-optimal ≈ maximize P(make), which is
score-independent.

A clean, direct confirmation of the [[champion]] marginal-value ranking
(auction ≫ belief ≫ score-utility ≫ **card-play polish**): score-conditioned
*play* risk is the wrong lever. The mechanism (lens dispatch, the `upside_10`
utility, the score channel) is correct and unit-tested; the value is the
measurement that redirects effort to the auction and belief (#25). The bidding
side of marks-to-7 conditioning — `MarksToSeven`'s optional equilibrium-aware
pass baseline (`pass_q_opp`/`pass_make_rate`) — does move the policy (16.7% of
sampled bids shift, all toward fighting harder for the auction). Its win-rate
impact, made measurable by the fast net bidder, is **null**: `net:wp,pass0.4` vs
`net:wp` over 128 games is −0.07 marks/game, 95% CI [−0.62, +0.46] — the q=0.4
aggression is calibrated near break-even (A takes more auctions, 54.1% offense
share, at a slightly lower make-rate 65.6% vs 69.1%, and the two cancel). The
right q derived from self-play is rung #26's equilibrium. Results under
`arena/results/scorelens_vs_ev_192/` and `pass_vs_nopass_128/`.

## First physics (2026-06-12)

Identical oracle play both sides (lens:ev, N=10), bidders differ —
192 games, base seed 1000:

- **heuristic beats bid30 58.9%** (113/192; halves 59.4% / 58.3%),
  mark margin **+0.78/game, 95% CI [+0.28, +1.26]**.
- Mechanism: hand selection. The heuristic made 51.1% of its contracts vs
  the baseline's 43.3%, taking 57.6% of auctions at mean bid 30.9.
- Self-play symmetry sanity (identical specs both sides) sits at ~50%, as
  it must.
- 192 full games ≈ 150 s on M5 Max MPS — cheap enough to be the default
  eval for every rung above it.

The auction was the predicted high-ground ([[champion]]: auction ≫ play
polish) and the very first measurement agrees: a seven-line static bidder
is already worth ~+0.8 marks/game over always-bid-30.

## Perf pass 1 — 2.38× games/sec on MPS, byte-identical (2026-07-06)

Dispatch/sync reduction on the oracle decision path, merged as `d678598`.
Byte-identical results were the gate — `per_hand.csv`/`per_game.csv` match
baseline exactly on CPU (12 games) and MPS (32 games, both A/B pairs); the
torch RNG stream was never touched, only deterministic-math restructuring
around it (numpy-assembled state tensors in `zeb_states_to_game_state_tensor`,
vectorized order-preserving pool construction in `sample_worlds_batched`,
maskless `scatter_add` void aggregation, the MRV loop in
`sample_worlds_mrv_gpu` cut from ~45 to ~20 kernels/step with dead per-step
syncs removed, per-device lookup-table caches, memoized `current_player`).

**Key finding: the arena is dispatch-bound, not compute-bound.** On CPU the
oracle forward pass dominates the wall (62%); on MPS the forward shrinks and
the same surrounding work becomes kernel-dispatch + sync overhead — the
measured "40% GPU" was actually ~1,500–2,000 kernel launches and ~25–45
GPU→CPU syncs per tick.

Paired MPS bench (32-game `net:wp+lens:ev` self-play, seed 0, alternating
A/B/A/B on an M5 Max): **0.56 → 1.34 games/s (2.38×)**, reproduced across two
A/B pairs (both <1% run-to-run). Post-merge production throughput sits at
~1.34 games/s on a pooled 128-game A/B. Full profile and per-lever breakdown:
`docs/arena-perf-2026-07-06.md`.

## Perf pass 2 — fast batching, 1.55–1.63× on paired A/B (2026-07-06)

The straggler tail was the remaining waste: the lockstep batch decays to
width 1–2 as games finish. Structurally this can't be fixed byte-identically
(the MRV sampler draws `torch.rand` once per step over the *global* batch
composition, so any pooling change reshapes every subsequent world sample);
the user relaxed the gate from byte-identity to distribution-level
equivalence for this mode. **Pass 2 (fast batching)**: `run_paired` pools
both halves of the paired match into one lockstep batch — twice the width,
one straggler tail. For a fixed set of games all-at-once pooling is
tick-optimal (total ticks = the longest game), so there is no refill queue.
`arena.cli` defaults to `--fast-batching`; `--no-fast-batching` restores the
sequential halves, which remain byte-identical and are the regression path.
Measured: **1.55–1.63× at 32 games** (24.1 s → 15.5 s), **1.29–1.41× at 128
games**; distribution-equivalent to the exact path (made-rate p=0.78,
mark-margin p=0.64, hands/game p=0.91 over 256 games/mode). Fast mode
diverges from the exact realization (batch composition feeds the
world-sampling RNG) but is itself run-to-run deterministic on a fixed device.

## Limits (v0)

- Shipped bidders declare pip trumps only and never bid past 42; the rules
  layer itself supports mark bids (84, then +42 per raise). Special
  contracts (nello, sevens, plunge) are out of scope.
- No belief carryover between hands within a game; each hand's play starts
  from the uniform world prior ([[champion]] rungs #24–#25 fix this at the
  model level).

## Links

- [[champion]] — the ladder this measures; rung 1
- [[w42-lens-v1-utility-head-to-head]] — the play-only paired-seed
  predecessor and the lens utilities the arena reuses
- [[w42-bookval-v1-wave2-bid-aware-atlas]] — validated the risk-budget
  arithmetic the heuristic bidder runs on
- [[gen-fleet]] — bid-30 corpus shape; the arena is the consumer that
  finally exercises `contract_threshold_bins` with real bids. The matching
  generation-side fix (rung #23, `bid_value` now threaded through
  `generate_eq_continuous`) landed the same day — see [[champion]].
