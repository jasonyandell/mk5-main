---
title: The farmboy field as a play field — Zeb planning against F_σ at ε = 0.2 (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: pre-registered
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

Pending.

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `zeb/field-eps-play/` (with `SHA256SUMS`),
once measured. Scripts: `scratch/field-eps-play/` (`grade.py`, `sched.py`, `report_grade.py`,
`aa_check.py`, `publish.py`), published there.

## Links

[[brainstorm-2026-10-14-ledger]] · [[scenario-player]] · [[tie-band-2026-10-14]] ·
[[research-night-2026-10-12]] · [[reads-plays-soft-2026-10-14]]
