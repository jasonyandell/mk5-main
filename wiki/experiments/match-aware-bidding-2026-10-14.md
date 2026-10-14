---
title: Match-aware bidding — Zeb's bidder with the 84 priced by match equity (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## Question

Jason: "we have a greater chance of making the bid by bidding higher iff doing so increases
our odds of winning the game. currently we only look at the hand. there are game winning
moves later on like bidding 84 instead of 42 if the enemy is at 6 marks. all upside no
downside." The bid's value is not P(make) or expected marks but P(win the game | score,
shaker, hand, call). Where does a bidder that prices its call by match equity differ from
deployed Zeb's hand-only bidder, by how much in theory, and does it win more games?

## Prior work checked

- **The oracle era built the yardstick and never tested the bid side.** `champion/utility.py`
  (rung #27, 2026-06-12): `race_wp`, a match-equity table under a coin per one-mark hand, and
  `MarksToSeven`, a contract priced as Δ win probability. Measured only on the play side
  (`scorelens`, −1.20 [−1.69, −0.70] marks/game) and as a pass-baseline pilot (null). "The
  auction-side score test (`MarksToSeven` thresholds varying with the score) was never run in
  isolation" ([[champion-ladder]] § Rung #27). Its bidders priced over oracle E[Q]; nothing here
  reuses them.
- **Silas cross-game slice 1** (`3e2d3135`, `silas/docs/cross-game.md` § B): the match frame,
  `WinValue::solve` over the score grid, and the dominance "one mark from game, 42 dominates
  84" proved over the hand's 43 outcomes. The converse (they need one: 84 weakly dominates 42)
  is the same theorem about `award`; no bidder used either.
- **Zeb's bidder reads only the hand.** `@jasonyandell/zeb`'s `forecast` takes hand,
  declarations, worlds and seed; `chooseBid` (zeb/index.js) bids the highest target 30..42 some
  declaration makes at ≥ 0.8; it never bids 84 ([[scenario-player]] § A live bidder). The arena
  had no Zeb bidder at all: every Zeb arena result to date is `bid30` or `heuristic`.
- texas-42's working tree: no match-equity or score-aware bidding (every "score-aware" hit is
  the in-hand point score). Its git history and unmerged branches were **not** searched (the
  worktree guard refused git outside this worktree).

## The math

**Match equity.** V(a, b, d) = P(we win the game to 7 | we hold a marks, they b, d = 1 when we
shake first this hand). One hand's law under the field is measured: w = the first shaker's side
takes the mark, f = the next hand's first shaker sits on the other side (a pass-out reshake
moves the shaker one seat). Every hand moves at least one mark, so V is an exact dynamic
program over the score grid, no iteration (`arena/match_equity.py`). Under a coin per hand it
is `race_wp` (tested equal).

**One-mark calls are score-blind.** Every contract 30–42 pays one mark, made or set. If every
branch of a decision moves exactly one mark, E[V] = V(a, b+1) + π·[V(a+1, b) − V(a, b+1)] with
the bracket positive and π the chance our side takes the mark. The score multiplies the
answer and never changes it: among pass, 30, …, 42 the best call maximizes π at every score.
So the score enters a bid only where the marks at stake differ — 84 and up. Since today's
bidder never bids 84, match-awareness can only *add* 84s to it; "protect the lead, bid lower"
is a correction to a bidder that 84s on hand strength, which this one does not.

**The 84 against the 42** (same make condition, all 42 points): 84 iff
p/(1 − p) > [V(a, b+1) − V(a, b+2)] / [V(a+2, b) − V(a+1, b)]. Under the coin law the
threshold is 0.5 at even scores, falls to 0.33/0.25/0.14 as they near 6, is **0 when they are
at 6** (a set loses the game at 42 or 84 alike; a make at 84 is two marks — Jason's "all upside")
and 42 dominates when we are at 6 (the book's rule). Against a lower point bid c0 (higher make
probability) the 84 must also beat c0's π.

**The bidder** (`arena/zeb_bid.py`, `zeb[:n<w>][,met|marks]`): the package's forecast
(bit-identical to the parity fixture) and `chooseBid` give c0; `met` adds one option at the
bidder's turn, the mark bid (84, or a mark over an 84) with p = the forecast's best pmake at 42,
taken when p·V(a+2, b) + (1 − p)·V(a, b+2) > E[V | c0]. E[V | c0] counts the field: a pass
leaves π = `pass_ours` (measured, by who holds the high bid and how many seats speak after);
a point bid t < 42 stands with `keep` (measured, by the same and t's band) else the overcaller
is set with `over_set`; a 42 is never overcalled by this field. Plunge's partner rule (never
overcall the partner) is kept for one-mark calls, but the mark bid is priced over a partner's
point bid too (a convention, not a law). `marks` prices the same option by expected mark margin
(score-blind). Information state only: the bidder's hand, the auction, the score, the shaker.

## Pre-registration (committed before any measurement)

**Players.** Play on both sides is deployed Zeb, `walt:k0,n512,F=margin_d250k_h128x64,f=16`
(compiled, seed 0; the net is the shipped `zeb/net/margin_d250k_h128x64.fnet`). Bidders:
`hand` = `zeb:n160` (the package's default 160 worlds: the 512-world forecast tripled the
pilot's wall time), `met` = `zeb:n160,met`, `marks` = `zeb:n160,marks`. Games to 7 marks,
`ArenaConfig` defaults (three reshakes, then the shaker is forced to 30).

**Blocks** (`scratch/met/blocks.json`, seeds disjoint):

- **AA** (run first): hand vs hand, both orientations (`run_match`, sequential), 64 games,
  seeds 9001…, half 4. Every cluster must be exactly antisymmetric; anything else is a harness
  bug and stops the study.
- **FIT**: hand vs hand, one orientation, 384 games, seeds 3001…, half 4. Fits the hand law
  (`win_flip`) and the odds (`keep`, `over_set`, `pass_ours`; a cell with fewer than 20 turns
  takes its holder's pooled value) with `fit_model.py`. The fitted constants are written into
  `arena/zeb_bid.py` and committed before `plan.py` runs on BASE.
- **BASE**: hand vs hand, one orientation, **2,048 games**, seeds 1001…, half 4. It does not
  depend on the model (hand-only bidders act on c0 alone; their pricing is a record).
- **P (primary)**: A = met vs B = hand, duplicate deals: every BASE game in both orientations.
  `plan.py` re-prices every BASE turn under the frozen model; in orientation t (A on team t)
  the game is the BASE game up to the first turn where team t's met call differs from c0
  (every bidder before it returned c0; play is deterministic and coordinate-seeded), so an
  orientation with no such turn takes the BASE result exactly and only the others are played
  (`sim_games.py`, the same engine from the deal). **Primary: A's game win rate over all
  2 × 2,048 games. Success: the 95% interval's lower bound > 50%.** Chance level 50% (A/A).
  Cluster = one BASE game, both orientations; 95% percentile bootstrap over clusters, 10,000
  resamples.
- **VALID**: P's bidders on the first 32 orientations (by seed, game, team) that never
  diverge, played anyway: each must reproduce its BASE game exactly (every hand's bids, points,
  dealer, and the winner). Any mismatch voids the shortcut and P is rerun as full duplicate
  games.
- **S (secondary)**: A = met vs B = marks, the same construction (an orientation diverges at
  the first turn where team t's met call or team 1 − t's marks call differs from c0). Read: A's
  win rate with the same bar; it isolates the score from simply pricing 84s. S does not decide
  anything alone.

**Theory beside the measurement.** For every P orientation that diverges, the model's change
in A's win chance at that first turn, E[V | 84] − E[V | c0]; their sum over 2 × 2,048 is the
model's predicted change in A's win rate (first order: later turns of the same game are not
counted). Also reported: the score at the divergences, the taken 84s' realized make rate
against their mean forecast p42, by who held the high bid.

**Size.** The pilot (12 games, seeds 8101…; never read for outcomes) ran 19 games a minute in
one orientation at 6 workers, so BASE takes about 110 minutes; P and S play only their
diverging orientations. BASE's size is fixed by that budget, not by power: if the divergence
rate is ~30% of orientations at a ±1 outcome, the interval's half-width is about ±1.2 points
of win rate, and the effect may well be smaller. An interval holding 50% is reported as a
null at that power. The size is not extended.

**Batch-drop rule.** Every batch runs under `timeout 300`, 6 workers. A failed block batch
retries at half 2, then 1, then the seed is DROPPED (logged in `logs/sched.log`) and the block
takes the next seed. A failed simulation batch retries one game at a time; a game that still
fails is DROPPED and its cluster counted at the BASE result for that orientation (reported).

**Decision.** No Zeb default changes either way. If P clears its bar, the page proposes the
mark-bid option for the next zeb version, Jason's call.

Scripts at registration (sha256 prefixes): `run_batch.py` 293b22b8414f, `sim_games.py`
071d90dc25dd, `sched.py` a4b3e6874227, `sched_sim.py` 11bea86359cb, `plan.py` b24fe6472aca,
`fit_model.py` 28a5e1b0a238, `blocks.json` 1f9fe76cd15c.

## Results

Pending.

## Links

[[scenario-player]] · [[champion-ladder]] · [[silas]] · [[rules-of-42]] · [[field-eps-play-2026-10-14]]
