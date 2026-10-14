---
title: The farmboy field as a play field — Zeb planning against F_σ at ε = 0.2 (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Question

Deployed Zeb plans against a deterministic field: every modeled seat plays the field net's
move. `field_eps` = ε (arena `E=`, #108) makes that field stochastic, F_σ: in each scenario a
modeled seat plays a uniform legal tile with probability ε, else the net's move. This is the
80/20 "farmboy" mix of [[brainstorm-2026-10-14-ledger]] row A. Until now it was graded only as
a *belief* (the likelihood behind #111's posterior), never as the field Zeb *plans against*.
Does Zeb planning against F_σ at ε = 0.2 beat deployed Zeb (ε = 0) at the table?

The ledger's prediction is null or slightly negative: a best response to a sloppier model of a
table that is not sloppy.

## Summary

**No. Planning against the 80/20 field does not beat deployed Zeb.** The primary misses its bar:
−0.03 [−0.11, +0.06] marks/game over 1,536 games to 7 marks. The interval holds zero, and it
also rules out a gain above +0.06. To 250 points it wins 49.2% [47.0, 51.4] of games, −2.1
[−6.6, +2.4] points a game. The A/A arm is exact. The farmboy field changes Zeb's play in every
one of the 768 primary deal pairs, but it buys no marks or game share. The prediction (null or
slightly negative) holds. No default moves. Pre-registered at `9b5f4dd1`.

## Prior work checked

- `git log` and the wiki hold no play grade of `E=` as the planner's modeled-seat field. #108
  (2026-10-12, [[scenario-player]] § A stochastic field) measured only the likelihood's support
  and ESS. #111's +0.026 marks/hand used `d=posterior` at ε = 0 (the net's 0/1 filter). Research
  night's E1c (softened posterior, null at power) predates `field_eps` and reweighted beliefs; it
  did not change the plan's field. [[reads-plays-soft-2026-10-14]]'s ε is the *reading's*
  softening (`r=…:<eps>`), not `E=`.

## What ε = 0.2 means, in code

`Config.field_eps` (`scenario/planner.py`), `field_move` (`scenario/core/src/planner.rs`): at
rung 0 with a field net, a modeled seat in scenario r explores when that scenario's record-keyed
coin falls below ⌈ε·2^53⌉/2^53, and then plays the dice's uniform legal tile from r's tape; else
the net's move. So P_σ(t) = ε/n + (1 − ε)·[t = the net's move] over the n legal tiles. Floored
live moves stay the net's. The live seat's worlds stay uniform (`d=uniform`); only the modeled
seats' play inside the plan changes. ε = 0 is deployed Zeb byte for byte.

## Pre-registration (committed before any measurement)

**The players.** Deployed Zeb, `walt:k0,n512,F=margin_d250k_h128x64,f=16` (compiled, seed 0,
`t=lowest`, Zeb 0.2.0's play), is B. A is the same spec plus `E=0.2`. Field net:
`margin_d250k_h128x64.npz` (sha256 `1615989b…5592d`, the net shipped in `zeb/net/`).

**The arena.** `bid30` both sides (the hub's protocol). Duplicate deals: each deal seed is played
in both orientations. A batch is `half` = 4 games per orientation under one base seed. A cluster
is one game index of one batch, both orientations. Intervals are 95% percentile bootstraps over
clusters, 10,000 resamples (`report_grade.py`, tie-band's unchanged). Batch lines print
throughput only, and no block is read until its last batch is in.

**Primary (P).** A against B, games to 7 marks, marks/game for A, **1,536 games (768 clusters)**,
base seeds 1001, 1002, … in order. **Success: the interval's lower bound > 0.** Chance level is 0
(A/A). Anything else is reported as measured: an interval holding 0 is a null; an upper bound
< 0 says the sloppier model costs marks.

**Size, from the pilot.** A one-minute pilot (seeds 8001 marks / 8501 points, half 2, two
processes, load ≈ 13–18 on 18 cores; its outcomes were not read and belong to no block) ran
4 games in 32 s and 26 s: about 8 s a game per process. With six workers that is roughly 45
games a minute, so P takes about 35 minutes. The prior per-cluster spread comes from the hub's
mirrored block of two different Zebs (384 games, ±0.30 marks/game): σ_c ≈ 2.1, so 768 clusters
give a half-width of about 1.96 · 2.1 / √768 ≈ 0.15. The size is fixed; it is not extended if
the realized half-width is wider.

**Secondary (S).** A against B, games to 250 points under the book's settlement
(`points_to_win=250`), **768 games**, base seeds 2001, …. Read: game share for A, with the bar
lower bound > 50% for a secondary win. Reported beside it: marks/game, points/hand, settled
points/hand, final score/game, made rates. S does not decide anything alone.

**A/A arm (run first).** A = `E=0` against B = deployed Zeb, games to 7 marks, 64 games, seeds
9001, …. ε = 0 never draws the coin, so every cluster must be exactly antisymmetric
(`aa_check.py`). Anything else is a harness bug and stops the study.

**Batch-drop rule.** Every batch runs under `timeout 300`; half 4 is about 65 s at the pilot's
load, so under 150 s at twice that. A killed or failed batch writes nothing and is retried at
half 2, then half 1. If half 1 also fails, that seed is dropped (logged `DROPPED` in
`logs/sched.log`), and the block takes the next seed in order until it holds its registered
games. Drops are reported with the results. A batch stopped by hand for machine load is rerun
at the same seed and size, and is not a drop. Concurrency is capped at 6 workers (shared
machine).

**Decision.** No Zeb default changes either way. If P clears, the page proposes `E=0.2` as a
candidate for the next zeb version, pending Jason. If not, it reports the miss.

## Results

Deployed Zeb with `E=0.2` (A) against deployed Zeb (B), `bid30` both sides, duplicate deals,
95% intervals over deal clusters. *Points/hand* is the raw split of a hand's 42. *Settled* is
the book's settlement.

| block | game | games (clusters) | marks/game [95%] | game share [95%] | points/hand | settled/hand | final score/game | made, A / B |
|---|---|---|---|---|---|---|---|---|
| AA (`E=0`) | to 7 marks | 64 (32) | 0 [0, 0] | 50.0% | 0 | 0 | — | 38.3% / 38.3% |
| **P** | **to 7 marks** | 1,536 (768) | **−0.03 [−0.11, +0.06]** | 49.2% [47.7, 50.8] | +0.10 [−0.06, +0.27] | −0.03 [−0.35, +0.28] | — | 36.9% / 37.2% |
| S | to 250 | 768 (384) | −0.09 [−0.20, +0.02] | **49.2% [47.0, 51.4]** | +0.10 [−0.18, +0.38] | −0.25 [−0.78, +0.28] | −2.1 [−6.6, +2.4] | 37.6% / 38.6% |

- **A/A.** All 32 clusters are exactly antisymmetric (`aa_check.py`). ε = 0 is deployed Zeb.
- **P: misses.** The lower bound is −0.11 against a bar of 0. The made rates agree within 0.3
  points. Per hand, −0.003 [−0.010, +0.005] marks. The realized half-width (±0.085) is tighter
  than the planned ±0.15, because the two Zebs agree more often than two different Zebs do.
- **The field does change the play.** None of P's 768 clusters is antisymmetric, so every
  duplicate pair diverges somewhere. How often the tile changes per decision was not counted.
- **S (secondary): misses.** 49.2% of games to 250. Every column holds zero, and S's made rate
  and marks lean a little toward B, inside the noise.
- **Reading.** The modeled seats here really are nets, or Zebs that play close to the net. A
  plan that expects a uniform tile 20% of the time spreads weight on lines the table rarely
  plays. Here that costs nothing measurable and gains nothing. The farmboy field's value, if any,
  is as a *belief* (a posterior with full support), not as a play field.

**Drops and deviations.** None. 296 batches (AA 8, P 192, S 96), none killed or dropped; batch
wall time median 50 s, max 69 s, under `timeout 300` at 6 workers on a shared machine. The
throughput pilot (8 games, seeds 8001 and 8501) belongs to no block and was never read; it is
published with the data.

**Decision.** P does not clear, so nothing is proposed. `field_eps` stays default 0, and Zeb
0.2.0 is unchanged.

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `zeb/field-eps-play/` holds every batch's
games and per-hand tallies (`out/grade/<block>/s<seed>.json`), the block summaries
(`summary.json`, `out/grade.md`), the logs (`logs/sched.log` has the kill/drop record) and the
scripts, with `CARD.md` and `SHA256SUMS`. Recompute:

```bash
python -u scratch/field-eps-play/sched.py AA P S      # cap in scratch/field-eps-play/maxjobs (default 6)
python -u scratch/field-eps-play/aa_check.py AA
python -u scratch/field-eps-play/report_grade.py AA P S
```

## Links

[[brainstorm-2026-10-14-ledger]] · [[scenario-player]] · [[tie-band-2026-10-14]] ·
[[research-night-2026-10-12]] · [[reads-plays-soft-2026-10-14]]
