---
title: Match-aware bidding — Zeb's bidder with the 84 priced by match equity (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
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

## Summary

**Yes. Pricing the 84 by match equity wins more games than deployed Zeb's bidder: 53.4%
[52.5, 54.3] of 4,096 games to 7 marks, against a chance level of 50% (A/A exact).** The model
predicted +2.9 points; the arena measured +3.4. The primary clears its pre-registered bar.
About a third of the gain comes from Jason's case, the opponents at 6. The largest single part
comes from bidding 84 where the hand-only bidder bids 42 at even-ish scores, which is right at
almost any score. Against the same 84 option priced score-blind (by expected marks), the
match-equity pricing still wins 51.4% [50.6, 52.2], so the score itself is worth about 1.4
points of game share on top of having the 84 at all. No default changes; the mark-bid option
is proposed for the next zeb version, Jason's call. Pre-registered at `c9a868cb`, model frozen
at `ee2c95b7`.

## Results

**AA.** 68 games, 34 clusters, every one exactly antisymmetric (three batches timed out under
machine load and were retried at half 2, so the block holds 68 games, not 64).

**The field model** (FIT, 384 games, 4,251 hands). The first shaker's side takes the mark
50.3% of hands: the hand law is a coin, and the measured V agrees with the coin `race_wp` to
three places; the 84 thresholds are the coin table on [[match-equity]]. Passing leaves our side
the mark 43–51% of the time when nobody has bid, 80–83% when the partner holds the bid, and
17–20% when an opponent does. A point bid stands 88–100% of the time; an overcaller is set 17%.

**VALID.** 32 orientations that never diverge, played in full: 32 reproduce BASE exactly
(every hand's bids, points, dealer, declaration, and the winner). The shortcut is exact.

| block | A vs B | orientations played / diverging / all | A's game share [95%] | model (first order) |
|---|---|---|---|---|
| **P** | met vs hand | 2,401 / 2,401 / 4,096 | **53.37% [52.47, 54.27]** | 52.94% |
| S | met vs marks | 2,993 / 2,993 / 4,096 | 51.42% [50.63, 52.22] | 50.43% |

Drops: none. In P, 2,401 of 4,096 orientations (58.6%) differ somewhere. In 1,913 of them the
same side still wins; A turns 313 losses into wins and 175 wins into losses. Per diverging
orientation that is +0.057 [+0.040, +0.075].

**Where P's gain lives**, by the context of each orientation's first differing turn (each
row's Σ Δ over all 4,096 orientations, so the rows add to the headline; exploratory, not
registered):

| first differing turn | n | measured | model |
|---|---|---|---|
| they need 1, hand-only passes, opponent holds the bid | 386 | +0.42 [+0.07, +0.76] | +0.51 |
| they need 1, hand-only passes, nobody or partner holds | 282 | +0.46 [+0.10, +0.83] | +0.21 |
| they need 1, hand-only bids a point bid | 60 | +0.12 [−0.05, +0.32] | +0.12 |
| they need 1, hand-only bids 42 | 14 | +0.07 [0.00, +0.17] | +0.07 |
| neither needs 1, hand-only bids 42 | 474 | **+1.61 [+1.15, +2.08]** | +1.11 |
| neither needs 1, hand-only bids a point bid | 419 | +0.37 [−0.07, +0.81] | +0.35 |
| neither needs 1, hand-only passes, opponent holds | 596 | +0.15 [−0.42, +0.73] | +0.39 |
| neither needs 1, hand-only passes, other | 169 | +0.17 [−0.15, +0.49] | +0.19 |

(points of game share; one turn at our 6 never reached the 84.) The opponents at 6 account for
742 first turns and +1.07 points. Upgrading a 42 to 84 at other scores accounts for +1.61. The
riskiest row, an 84 over an opponent's bid at an even-ish score, is flat.

**Calibration** (the forecast p42 against what happened):

| | taken | made | forecast |
|---|---|---|---|
| BASE's hand-only 42s, nobody had bid | 576 | 88.5% [85.9, 91.1] | 89.1% |
| BASE's hand-only 42s, over an opponent's bid | 93 | 80.6% [72.0, 88.2] | 89.1% |
| P's 84s, nobody had bid | 1,548 | 59.9% [57.5, 62.4] | 62.0% |
| P's 84s, over an opponent's bid | 1,466 | 28.0% [25.8, 30.4] | 31.5% |
| P's 84s, over the partner's bid | 193 | 48.7% [42.0, 55.4] | 51.9% |

The forecast reads only the bidder's hand. With nobody bid it is honest. Over an opponent's
bid it runs 3–8 points hot, because the opponent has just shown strength the forecast never
sees. That is the information-state cost of a hand-only forecast, and it caps the riskiest row.

**Reading.**

- *The math was right about where the score lives.* One-mark calls are score-blind, so all of
  the measured difference comes through the 84. Today's bidder never bids it, so every change
  is an added 84. "Protect the lead" shows up only as the 84s *not* bid at our 5 and 6.
- *Jason's case is real and free.* At their 6 the match-equity bidder bids 84 over 42, over a
  point bid, and over a pass. +1.07 points of game share come from those turns alone.
- *The biggest part is not score-awareness.* The 84 over a 42 that the hand-only bidder would
  make anyway (p42 ≥ 0.8) is right at almost every score: the threshold is 0.5 at even scores.
  S isolates the score: against the same option priced by expected marks, match equity still
  wins 51.4% [50.6, 52.2], about +1.4 points.
- *The pass threshold is a separate defect.* With an opponent's bid standing, passing leaves
  our side the mark 17–20% of the time, so by the one-mark theorem any overcall that makes more
  often than that beats the pass. The hand-only 0.8 cutoff ignores that. The match-equity
  bidder reaches some of those hands through the only channel it has, the 84, and that row is
  flat. A one-mark fix (overcall when p_t > `pass_ours`) is a different bidder and is not
  measured here.
- *What the arena cannot see.* The field here never bids 84 or 126 and
  never adapts to a bidder that does. A real table at its 5 or 6 would pre-empt with 84 itself
  and would overcall a 42 at their 6. P measures the gain against deployed Zeb, not against a
  table that answers in kind. A met-vs-met block was not run.

**Deviations from the registration.** None in the registered blocks or scripts (hashes match).
The pilot ran on an earlier bidder build and was never read. `grade.py` first counted the
hand-only bidder's *priced* (unacted) turns as B's divergences in P's model line; this was fixed
before the numbers above were read off. The by-turn table and the calibration anchor were added
after the registration, so both are exploratory.

**Decision.** P clears, so the page proposes the mark-bid option (`met`, with the field model
refitted on the version that ships) for the next zeb version, pending Jason. Zeb 0.2.0 and every
default stay as they are. Follow-ups: #160 (the pass threshold against an opponent's bid, met vs met, an auction-aware p42).

## Follow-ups (#160): the pass over an opponent, an honest forecast there, a table that bids 84

Three things the results above left open, measured before zeb 0.4 decides on `met`.

### Pre-registration (committed before any measurement)

**The two repairs** (`arena/zeb_bid.py`, both default off, both act only when an opponent
holds the high bid at the bidder's turn):

- **`over`** (defect a). When the 0.8 rule passes, overcall with the one-mark call 31..42 whose
  π (the chance our side takes the hand's mark: `Odds.take`, the frozen field model) is largest,
  if that π beats passing's (`pass_ours`, 17–20% over an opponent). Every branch moves one mark,
  so the larger π is the better call at every score: score-blind by the theorem on
  [[match-equity]].
- **`cal`** (defect b). There the forecast is replaced by p̃ = sigmoid(a + b·logit p), a
  two-parameter logistic recalibration fitted by maximum likelihood on every over-an-opponent
  contract that stood in this study's prior blocks (BASE's hand-only point bids and 42s, P's
  mark bids; `fit_cal.py`). It is fitted on held-out data: none of the follow-up's games exists
  yet. p̃ is read everywhere the bidder reads the forecast there: the 0.8 rule, `over`, and the
  mark bid's p42. The map is monotone, so the declaration does not change. **Why this and not
  Walt's auction read** (#149, `Situation.auction`, `test_auction_read.py`): that read is a
  likelihood on the *play* planner's world draw. It needs a model of how each opponent bids,
  and nobody has fitted one for Zeb's bidder. A forecast conditioned on the auction through it
  is a new estimator, while the shrink is two numbers fitted on data we already have. The
  shrink is the honest simplest; the auction read is the right long-run object.

**The fixed bidder** is `zeb:n160,met,over,cal`. Off the opponent-held turns it is `met`
exactly.

**Construction** (the prior study's shortcut, unchanged): the first 768 BASE games of the
prior study (seeds 1001–1192, all four games of each, by seed and game) are the clusters.
Every BASE turn an opponent held the bid has its forecast recomputed from the deal seed
(`refc.py`). Before any arm is planned, the hand-only bidder run on each recomputed forecast
must give the record's call, p_c0, p42 and priced call to the bit, or the study stops. `plan.py`
finds each orientation's first turn where the arm's bidders call differently from BASE. Only
those orientations are played (`sim_games.py`, the same engine from the deal; lockstep
across seeds, 30 games to a `scripts/cpu-slot` run, one slot at a time).

**Arms** (A on team t, both orientations of every cluster):

- **AA**: met vs met, the first 32 clusters where met diverges, both orientations played.
  Every cluster must be the same game mirrored (exact antisymmetry). Otherwise it is a harness
  bug and the study stops. Note: in a two-team game the pairwise share is the whole story. Met
  vs met is 50% by construction, and deployed Zeb against a met table wins 1 − P = 46.6%. So
  "met's gain against a table that bids 84" is P's +3.4 already. What the table question can
  change is the *repairs'* worth, so the realistic estimate is FM.
- **VALID**: F's bidders on the first 32 never-diverging orientations, played in full. Each
  must reproduce BASE exactly, or the shortcut is void and the arms are rerun as full games.
- **F (bar)**: A = `met,over,cal` vs B = deployed hand-only. **Bar: the 95% interval's lower
  bound > 50%.**
- **FM (the realistic estimate, reported)**: A = `met,over,cal` vs B = `met`. Here the
  opponents bid 84 and 126 by match equity too.
- **O (secondary, last, only if the slot budget allows)**: A = `over,cal` (hand mode) vs B =
  hand-only. These are the one-mark repairs alone, score-blind. Not running it is reported as
  not run.

Primary statistic: A's game share over the 2 × 768 orientations. Cluster = one BASE game, with
both orientations. 95% percentile bootstrap over clusters, 10,000 resamples. Chance 50% (AA).
Beside each arm: the model's first-order prediction, Σ ΔV at the first differing turn. Then
`over`'s overcalls, whether our side took the mark more often than passing's π, and the
calibration of A's over-an-opponent contracts. `over`'s calls are out of sample for the fit;
the 0.8 rule's calls and mark bids repeat the fit's kinds.

**Size.** The throughput pilot (24 of P's diverging games in one lockstep run; never read for
outcomes) took 185 s on a loaded machine. All 24 reproduced P's games exactly. Under one slot
that is about 8 games a minute. 768 clusters, with F and FM diverging in most orientations,
make about 2,600 games for AA through FM, roughly 5–6 hours. At P's spread the interval
half-width is about ±1.5 points. The size is not extended.

**Batch-drop rule.** A run that fails or is killed at 300 s is retried at half its games, down
to one. A game that still fails is DROPPED (logged in `logs/sched.log`), and its orientation is
counted at BASE.

**Decision rule for 0.4** (Jason's call; this is the recommendation the page will make):

- F clears and FM's lower bound > 50%: recommend `met,over,cal`.
- F clears and FM's interval holds 50%: recommend `met` alone. The repairs show no gain
  against a table that bids 84, and every line is a liability.
- FM's upper bound < 50%, or F fails: recommend `met` alone (P's evidence), and file the
  repairs.

No zeb crate change here in any case; the release carries it.

Scripts at registration (sha256 prefixes): `sim_games.py` 6fb1b3cfe300, `refc.py`
6fce0193c427, `plan.py` 396361403de5, `sched.py` 1586f6a0168e, `grade.py` 931c67b3ac88,
`fit_cal.py` d5c90a653386, `blocks.json` 86cd4628fd99.

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `zeb/match-aware-bidding/`: every block's
games and priced turns (`out/<block>/s<seed>.json`), the fit (`out/model.json`), the plan
(`out/plan.json`), the grade (`out/grade.md`), the logs (`logs/sched.log` holds every retry),
the scripts, `CARD.md` and `SHA256SUMS`. Recompute:

```bash
python -u scratch/met/sched.py AA FIT BASE
python -u scratch/met/fit_model.py FIT > scratch/met/out/model.json
python -u scratch/met/plan.py BASE
python -u scratch/met/sched_sim.py VALID; python -u scratch/met/sched_sim.py P; python -u scratch/met/sched_sim.py S
python -u scratch/met/grade.py VALID P S; python -u scratch/met/breakdown.py P
```

## Links

[[match-equity]] · [[scenario-player]] · [[champion-ladder]] · [[silas]] · [[rules-of-42]] ·
[[field-eps-play-2026-10-14]]
