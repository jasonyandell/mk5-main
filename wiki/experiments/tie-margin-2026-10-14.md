---
title: Make, then margin — Zeb's exact ties broken by the plan's mean margin (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Question

Issue #120. Under the `make` utility over 512 worlds, exact ties are common, and Zeb's tie
rule (≺ in [[scenario-player]]'s Θ) takes the lowest tile id. Two studies point at those
ties. In [[metasplat-2026-10-13]], at a decided contract the book's tile has the higher mean
margin in 44% of departures against Zeb's 21%. In [[book-head-to-head-2026-10-13]], book
departures at a decided hand cost no marks and gain +2.28 points each. Three questions
follow. Is the book's tile the margin-best tile of Zeb's tied set? Does a lexicographic rule
(make, then the plan's margin) change Zeb's strength? And how much does it raise Zeb's
book agreement?

## Pre-registration (committed before any measurement)

**What a tie is.** At a Walt decision, the tied set T is the legal tiles whose make value q
equals the best q exactly, as a double, in the decision's own sample. That sample is 512
worlds at the deployed size, or the whole fiber when the fiber is smaller. There is no noise
bar. Under uniform weights q is (worlds made)/n, so a tie means equal counts of worlds made.
A noise bar would let the rule give up a measured make edge for margin, which changes the
objective rather than breaking a tie. Every tie in this study is that exact tie.

**The rule, `make-then-margin`** (`Config.tie = "margin"`; arena `t=margin`; default
`"lowest"`). When |T| ≥ 2 the decision picks the tile of T with the best mean margin for its
own side under the plan: M(t) = Σ_r w_r·m_r(t). Here m_r(t) is the bidder-team final margin
that scenario r reaches when the root plays t and the plan continues. That is
`Analysis.outcomes`: the plan's line, then the tape where the plan stops (contract decided,
or horizon). The bidding side takes the largest M, a setter the smallest, and ties in M go
to the lowest tile id. The rule applies at every decision a Walt makes: the live seat's
root, every modeled rung's decision, and a convention sender's restricted choice. The plan
itself is unchanged: its interior choices still take the lowest id among equal values, so
q is unchanged wherever no modeled rung decides. It is `make` only. Both engines,
bit-identical; default off; Zeb's fixture untouched.

**Step 1, the tied sets.** Positions: the face-off's 20,977 gated rows (#119;
`scratch/faceoff/out/positions.jsonl`, 21 gates of about 1,000 rows each). Each row is
re-planned by deployed Zeb at current code (`walt:k0,n512,F=margin_d250k_h128x64,f=16`,
compiled, seed 0). The population is the rows where |T| ≥ 2 and T holds a book tile.
- **Headline:** the share where the margin-best tile of T is a book tile.
- **Chance:** the expected share for a uniformly random tile of T, |T∩B|/|T|.
- **Beside it:** the lowest-id rule's share.
- **Subsets:** the rows where the lowest-id tile is outside the book (the ties the book
  lost); dead rows (every legal tile ties) against live rows; and the share where the
  margin does not separate the tiles.

Rows are pooled with equal weight, as in the face-off. The 95% intervals are percentile
bootstraps over deals (chunk, deal), 10,000 resamples.

**Step 3, the book-agreement column.** On all rows, book agreement for re-planned Zeb under
each rule, with the exact random-tile baseline, plus the paired difference with its deal
interval. The face-off's pour figure (51.1%) is quoted beside them; it came from an older
player.

**Step 2, the grade.** Arena: `bid30` both sides (the hub's protocol), games to 7 marks,
each deal played in both orientations. Batches are capped at 300 s and pooled over base
seeds 1, 2, …. A cluster is one game index of one batch, both orientations. The headline is
marks/game for the challenger, with a 95% percentile bootstrap over clusters (10,000).
Points per hand stand beside it, and a game to 250 points if the arena supports one. The
challenger is deployed Zeb with `t=margin`, graded on two blocks.
- **G1: against deployed Zeb**, at least 768 games.
- **G2: against the champion `k1,n512,m8`**, at least 384 games. The standing block for
  plain Zeb is +0.04 [−0.27, +0.33].

**Decision.** The rule becomes a default, or a new zeb version, only if G1's marks
interval clears zero. Even then this page proposes the change and does not make it.
Points alone do not move a default. The champion stays `k1,n512,m8` unless G2 clears zero.

## Summary

**Neither result is large.** Where Zeb is indifferent, the book's tile is the margin-best tie
at 59.7%. Chance is 51.8% and the lowest-id rule 50.4%. The whole excess is in dead
positions, where every legal tile makes alike. The rule raises Zeb's book agreement from
52.2% to 54.3%, against 63.5% if every tie went to the book.

**At the table, the rule costs no marks and buys points.** Against deployed Zeb it scores
+0.01 [−0.04, +0.07] marks/game and +4.55 points per hand. In the book's game to 250 it wins
55.2% of games, by +20 points a game. Against the champion it scores +0.18 [−0.01, +0.36]
marks/game.

The pre-registered bar is G1's marks interval clearing zero. It does not, so nothing ships
and no default moves. The case for a change is a points case, and that call belongs to
Jason (below).

## The ties (deployed Zeb, re-planned)

Re-planned at current code, deployed Zeb has an exact tie for best at 25.0% of the face-off's
20,977 gated rows; every legal tile ties at 18.0% ("dead"). A book tile sits in the tied set
at 22.8% of rows (4,790). At 11.3% the lowest id is outside the book, so the book lost the
tie, which matches the 11.2% the face-off read off the pour. In those 4,790 rows the tied set
has a median of 2 tiles (mean 2.88), and the margin names a unique best at 82.5% of them.

## Step 1: is the book's tile the margin-best tie?

Shares of rows; 95% intervals over 19,370 deals.

| population | n | margin-best tie in the book | lowest id in the book | a random tie in the book (chance) | margin − chance |
|---|---|---|---|---|---|
| a book tile is tied for best | 4,790 | **59.7% [58.4, 61.1]** | 50.4% | 51.8% | **+8.0 pp [+6.7, +9.3]** |
| … live (the legal tiles do not all tie) | 1,015 | 63.7% | 59.6% | 62.3% | +1.4 pp [−1.1, +4.0] |
| … dead (every legal tile ties) | 3,775 | 58.7% | 47.9% | 48.9% | +9.8 pp [+8.3, +11.3] |
| the book lost the tie (lowest id outside) | 2,376 | 47.1% [45.1, 49.1] | 0% | 41.2% | +5.8 pp [+3.8, +7.8] |

The answer to #120's step 1 is "somewhat, and only where the contract is already decided".
In dead positions, margin under the plan leans toward the book's tile by about ten points
over chance. In live ties it does not lean measurably. So the stated reason for a book
tie-break is the reason [[metasplat-2026-10-13]] and [[book-head-to-head-2026-10-13]] found:
the book plays for count once the contract is settled. Nothing beyond that is found, and at
two in five of the ties the book lost, the margin still picks another tile.

## Step 3: the book-agreement column

All 20,977 rows, each gate about 1,000 rows, pooled with equal weight as in the face-off.

| player | book agreement [95%] |
|---|---|
| a random legal tile (exact expectation) | 37.0% |
| Zeb, the pour (the face-off's figure, an older player) | 51.1% [50.4, 51.8] |
| Zeb re-planned, lowest id (deployed) | 52.2% [51.5, 52.9] |
| **Zeb re-planned, make then margin** | **54.3% [53.6, 55.0]** |
| Zeb re-planned, every tie toward the book (the ceiling) | 63.5% [62.8, 64.2] |

The paired gain is **+2.1 pp [+1.7, +2.5]**. The rule moves the tile at 13.3% of rows, and
where it moves, the plan's margin rises by a mean of +3.4 points (median +1.9). It moves
toward the book where the book plays for count:
- *the donation lead* (C1a): 57% → 72%;
- *the double ahead of the off* (C11): 44% → 58%;
- *the setter's leads that call for count* (C15): 54% → 63%.

It moves away from the book where the book gives up count to keep shape:
- *low from a pair* (C13c): 63% → 50%;
- *lead a singleton* (C13d): 45% → 39%.

E[Q] n=10 (56.4%) and the field net (57.8%) still agree with the book more, with no tie rule
involved. The margin breaks Zeb's ties in its own units. It is not the book's opinion, and
most of the gap to the 63.5% ceiling stays open.

## Step 2: the grade

`bid30` both sides, paired orientations, 95% intervals over deal clusters. Every number is
for the challenger, deployed Zeb with `t=margin`. *Points/hand* is the raw split of a hand's
42. *Settled* is the book's settlement (made: each side its points; set: the setters the bid
plus theirs).

| block | against | game | games | marks/game [95%] | game share [95%] | points/hand | settled/hand | final score/game | made, A / B |
|---|---|---|---|---|---|---|---|---|---|
| G1 | deployed Zeb | to 7 marks | 768 | **+0.01 [−0.04, +0.07]** | 50.1% | **+4.55 [+4.34, +4.77]** | +2.57 [+2.35, +2.78] | — | 37.1% / 37.0% |
| G2 | champion `k1,n512,m8` | to 7 marks | 392 | **+0.18 [−0.01, +0.36]** | 51.8% | +4.80 [+4.43, +5.17] | +3.14 [+2.48, +3.81] | — | 38.3% / 36.6% |
| G1p | deployed Zeb | to 250 points | 384 | −0.02 [−0.09, +0.06] | **55.2% [53.1, 57.6]** | +4.24 [+3.86, +4.62] | +2.39 [+1.96, +2.82] | **+20.2 [+16.6, +23.9]** | 37.0% / 37.3% |
| G2p | champion `k1,n512,m8` | to 250 points | 192 | −0.08 [−0.32, +0.15] | 55.7% [51.0, 60.4] | +4.22 [+3.56, +4.86] | +2.12 [+0.93, +3.28] | +18.1 [+8.0, +28.0] | 35.7% / 36.8% |

- **Marks: null, and tightly so.** The two Zebs part only at exact ties of the make estimate,
  so the made rates agree to 0.1 point and G1's interval is ±0.05 marks/game. The rule
  never gives up an estimated make, and the table shows it buys none.
- **Points: about 4.5 a hand, every block.** About 2.5 of them survive the book's
  settlement. In the game to 250 that is 55% of games and about 20 points a game, against
  either opponent.
- **Against the champion:** +0.18 [−0.01, +0.36] marks/game. The standing block for plain
  Zeb read +0.04 [−0.27, +0.33] ([[scenario-player]], 2026-10-12), on other deals, so the
  difference is not a paired number. Both intervals hold zero, and the champion stays
  `k1,n512,m8`.

The points game ran on `ArenaConfig.points_to_win` from commit `d59a925b` (another session's
work, unmerged on 2026-10-14). Its `engine.py` and `auction.py` were copied verbatim into the
study's scripts, so this checkout's arena is unchanged.

## Recommendation

- **Under the pre-registered rule: no change.** G1's marks interval holds zero, so Zeb 0.2.0,
  the arena defaults and the champion stay as they are. `t=margin` lands default off.
- **Proposed, for Jason to decide.** Make make-then-margin the tie rule of the next zeb
  version. It is lexicographic, so it never trades an estimated make. It costs no measured
  marks (±0.05 a game). It is +4.5 points a hand, and a clear win wherever points count
  (55% of games to 250). The case is points, not marks: under a marks-only scoring the
  change is free and pointless. Adopting it changes bytes at ties, so it needs a new zeb
  version and a parity fixture position where the rule moves the tile.
- **Cross-checked against #77's exploiter** ([[exploiter-endgames-2026-10-14]] § Cross-check
  result): the rule cuts what knowing Zeb is worth at three-trick endgames from +0.50 to
  +0.29 [+0.24, +0.34] margin points (paired drop +0.21 [+0.14, +0.28]), leaves the
  contract exploit unchanged (+0.0038 against +0.0040), and on the two-rung player is +4.1
  points a hand with marks unsettled (+0.08 [−0.47, +0.65]).

- **A near-tie band does not beat exact ties** ([[tie-band-2026-10-14]]). Margin within 0.01
  of the best make probability, against this rule, wins 49.1% [47.7, 50.4] of games to 250. Its
  marks are +0.02 [−0.050, +0.086]. Exact ties stand.

## Caveats

- The margin past a decided contract is the tape's ([[scenario-player]] § the trace, #107),
  and both the rule and step 1 read it. A margin planned to the end would be a different
  rule, about the cost of `points`.
- The rule acts at every decision a Walt makes. For deployed Zeb (one rung over the net)
  that is the root alone. For a two-rung player it also changes the modeled seats; that
  player was graded later, too coarsely to settle marks ([[exploiter-endgames-2026-10-14]]).
- Step 1's rows are the face-off's sample of Zeb's own pour trajectories, re-planned by
  today's Zeb, so tied sets are today's player's. The pour's 51.1% is quoted, not
  recomputed.

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `zeb/tie-margin/`. It holds per-row step-1
reads (`out/tied.jsonl`), per-game and per-hand arena records for every batch
(`out/grade/<block>/s<seed>.json`), summaries, logs, and the scripts. `CARD.md` and
`SHA256SUMS` are with them. Recompute:

```bash
python -u scratch/tie-margin/tied.py --check 512          # repeat until 20,977 rows; ~30 rows/s
python -u scratch/tie-margin/report_tied.py > scratch/tie-margin/out/tied.md
bash scratch/tie-margin/run_all.sh                        # capped batches, timeout 300 each
python -u scratch/tie-margin/report_grade.py G1 G2 G1p G2p
```

Code: `Config.tie` (`scenario/planner.py`, `scenario/core/src/planner.rs`), arena `t=margin`.
Tests: `scenario/tests/test_tie.py`.

## Links

[[scenario-player]] · [[tie-band-2026-10-14]] · [[metasplat-2026-10-13]] · [[book-head-to-head-2026-10-13]] ·
[[book-study-2026-10-12]]
