---
title: Zeb reads the plays through a soft set — it pays at a net table and against Zeb (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Summary

Issue #145. The sharp reading ([[reads-plays-2026-10-13]]) drew Zeb's worlds from a posterior that
holds every other seat to the field net's 2-point set. It was a null in marks, and its posterior
was empty at 28% of trick-5 and 51% of trick-6 decisions. Those decisions fell back to the
uniform, which is where [[lumps-voi-2026-10-13]] puts the value. Here the set is softened the
way F_σ softens the net's move. A play of one of n legal tiles weighs

    ε/n + (1 − ε)·[tile ∈ A]·w,   w = 1 (consistent) or 1/|A| (uniform),   ε = 0.28

so no world vanishes. ε was fitted to deployed Zeb's own choices before any arena run. The
reader was measured at a table where the other three seats *are* that model, and against
deployed Zeb. No perfect-information value enters ([[strategy-fusion-fence]]): every value is
Zeb's plan over the drawn worlds.

**What it shows.** The soft reader **pays in both tables.**

- *Net table* (the model is true), the pre-registered primary: uniform − plain
  **+0.23 [+0.05, +0.41] marks/game**, +8.1 [+1.0, +15.5] settled points/game, 192 game pairs.
- *Zeb table* (the reader against plain Zeb): consistent **+0.30 [+0.14, +0.47]**, uniform
  **+0.29 [+0.10, +0.46]** marks/game, 168 game pairs each. The A/A control (plain Zeb at
  another seed through the same harness) is −0.03 [−0.20, +0.15].

These are a gain the size of #111's (+0.026 marks/hand against nets): +0.021 marks/hand at the
net table, +0.027 against Zeb. The fallback rate is **0 at every trick** (sharp: 28% / 51% at
tricks 5 / 6).

**Restricted choice stays unpriced**, at both tables. Uniform − consistent is +0.09
[−0.08, +0.26] at the net table and −0.02 [−0.22, +0.20] against Zeb. That is null even
where the uniform likelihood is the true one. What pays is the softening, not the set size
as evidence. On the 120 game pairs the sharp study shares with this one, soft − sharp is
+0.38 [+0.15, +0.62] (consistent) and +0.25 [+0.01, +0.49] (uniform). That comparison is
exploratory: it was not pre-registered.

## Prior work, and what this adds

- **Night 1's `dice` and `field` tilts** (retired 2026-10-12) *reweighted* 512 uniform worlds by
  a play likelihood. Nulls and losses; ESS collapse named. The soft reading is drawn from
  (exact or enumerated draws, rejection above 2^20), never reweighted.
- **#108/#111, F_σ and the posterior sampler** ([[scenario-player]] § Foundation repair): ε over
  the net's *move*, +0.026 marks/hand against seats that are the deterministic net. In Zeb
  self-play the deterministic posterior was empty late and the softened one null at power (E1c).
- **[[reads-plays-2026-10-13]]**: the net's *set* (δ = 2 points), sharp. Null; empty late.

New here: ε over the set (so at |A| = 1 it is F_σ exactly), with ε *fitted* to Zeb rather than
chosen. A net table whose seats play the reader's model, so the posterior is the true one. And
an A/A control.

## Fixed before running

Committed at `7188cd90` before any measured batch (`scenario/reads_soft.py`, top of file):

- **ε = 0.28**: the maximum-likelihood ε of the soft uniform model on deployed Zeb's 714
  choices of the research-night pour (Zeb self-play, the same net): ML 0.2835, profile 95%
  [0.22, 0.36]. 7.6% of Zeb's choices fall outside the 2-point set (mean set 2.52 of 3.66
  legal). The same ε for the net seats.
- **Players.** Plain = deployed Zeb `k0,n512,F=margin_d250k_h128x64,f=16`. Reader = plain +
  `d=posterior,r=<likelihood>:4194304:0.28`. `bid30` bidders, games to 7 marks, paired halves
  (team A as team 0 and as team 1 over the same deals).
- **Net table.** Team A = {X at the team's lower seat, a net-seat partner} against two net
  seats, X ∈ {plain, consistent, uniform}. A *net seat* explores a uniform legal tile with
  probability ε, else picks uniformly within the net's set (`NetSeat`). Its draws are keyed on
  (deal, record), so a state is played identically in every arm. 4 batches × 48 game pairs per
  arm, seeds 1–4.
- **Zeb table.** Two readers against two plain Zebs, consistent and uniform; 7 × 24 game pairs,
  seeds 1–7.
- **Effects and CI.** Marks per game (a game pair's sum / 2), 95% bootstrap over game pairs
  (4,000 resamples): the cluster is the game pair. Net: uniform − plain (**primary**),
  consistent − plain, uniform − consistent. Zeb: each reader's margin (chance 0 by symmetry),
  uniform − consistent. Also settled points per game (the book's 250 tally, kept on every hand).
- **Bars.** An effect pays when its interval excludes 0 above, costs when it excludes 0
  below, else it is a null.
- **Batches** under `timeout 300`; an overrun dropped, not rerun. *None overran.* A timing
  pilot (seeds 900–904) is excluded; its rows were read for timing and fallbacks only.

Added after the measured runs, and marked so: the **A/A control** (plain at seed 1 against
plain, the zeb table's 168 pairs) and the soft − sharp comparison.

## Marks

| effect | game pairs | marks/game | settled points/game | bar |
|---|---|---|---|---|
| **net: uniform − plain (primary)** | 192 | **+0.23 [+0.05, +0.41]** | +8.1 [+1.0, +15.5] | pays |
| net: consistent − plain | 192 | +0.14 [−0.05, +0.34] | +5.3 [−2.3, +12.9] | null |
| net: uniform − consistent (restricted choice) | 192 | +0.09 [−0.08, +0.26] | +2.8 [−4.0, +10.0] | null |
| zeb: consistent vs plain | 168 | **+0.30 [+0.14, +0.47]** | +11.7 [+5.3, +18.2] | pays |
| zeb: uniform vs plain | 168 | **+0.29 [+0.10, +0.46]** | +13.7 [+6.2, +20.7] | pays |
| zeb: uniform − consistent (restricted choice) | 168 | −0.02 [−0.22, +0.20] | +2.1 [−6.2, +10.4] | null |
| *A/A: plain at seed 1 vs plain (added)* | 168 | −0.03 [−0.20, +0.15] | | chance |

- **Net table, per arm against the nets.** X's team: plain +1.48, consistent +1.62, uniform
  +1.71 marks/game. Contracts made bidding: 41.4% / 42.7% / 43.2%. Per hand: +0.136 / +0.150 /
  +0.157.
- **Zeb table.** The reader's contracts made bidding: 37.7% (consistent) and 37.3% (uniform),
  against plain's 34.8% / 34.6% in the same games. The A/A arm: 36.6% / 36.7%.
- **Per batch**, the zeb table's reader margin is positive at 6 of 7 seeds in each arm.
- **Against the A/A arm on the same pairs** (exploratory): consistent +0.33 [+0.09, +0.58],
  uniform +0.32 [+0.08, +0.55].

The bars were met at the primary and at both Zeb-table arms. Restricted choice is null at
both. The marks come from the soft posterior itself, not from the 1/|A| factor.

## Mechanism: the posterior no longer empties

| trick | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| fallback, soft (every arm) | 0 | 0 | 0 | 0 | 0 | 0 |
| fallback, sharp ([[reads-plays-2026-10-13]]) | 0% | 0.5% | 2.9% | 11% | 28% | 51% |
| distinct worlds of the 512 drawn (median, zeb uniform) | 512 | 512 | 507 | 406 | 147 | 9 |
| importance ESS fraction of soft L on the uniform 512, zeb (consistent / uniform) | 0.87 / 0.66 | 0.53 / 0.49 | 0.34 / 0.31 | 0.31 / 0.27 | 0.27 / 0.30 | 0.55 / 0.52 |
| the same, sharp L at those decisions | 0.86 / 0.59 | 0.48 / 0.30 | 0.27 / 0.14 | 0.17 / 0.09 | 0.08 / 0.05 | 0 / 0 |
| real deal's rank under soft uniform L (0.5 = chance), zeb table | 0.50 | 0.77 | 0.86 | 0.80 | 0.80 | 0.72 |
| real deal's rank, net table | 0.53 | 0.72 | 0.88 | 0.87 | 0.82 | 0.83 |

ESS rows: every 50th decision at the net table, every 100th at the zeb table, 90–129 per trick
(`ess_net.json`, `ess_zeb.json`). At trick 6 the fiber is usually at most 512, so the distinct
count is the fiber's size. The real deal is used only for the rank diagnostic, never to play.

The soft posterior is much broader than the sharp one late in the hand: an ESS of about 0.3
against 0.05–0.08 at trick 5. It still ranks the real deal at the 72nd–88th percentile from
trick 2 on. So late in the hand it carries information without excluding the truth. The sharp
reader there either excluded the truth or fell back to knowing nothing.

## How often reading changes the move

| | net consistent | net uniform | zeb consistent | zeb uniform | chance (plain re-seeded) |
|---|---|---|---|---|---|
| every decision | 15.1% | 15.2% | 14.2% | 15.3% | 14.7–14.8% |
| exact fiber (no sampling either side) | **3.4%** | **3.5%** | **3.6%** | **4.1%** | 0 |
| enumerated draw | 21.0% | 20.2% | 20.1% | 20.1% | 15.4–16.0% |
| rejection (tricks 1–2) | 20.1% | 21.7% | 18.1% | 21.9% | 31.1–31.3% (shares no stream: not comparable) |
| uniform vs consistent reader | | 13.5% | | 13.3% | |

By trick at the zeb table (consistent / uniform / chance): 14% / 18% / 34% at trick 1, 24% /
26% / 25%, 23% / 23% / 20%, 17% / 17% / 10% at trick 4, 6.8% / 7.3% / 1.3% at trick 5, 1.5% /
1.9% / 0 at trick 6. The excess over chance sits at tricks 4–5, as in the sharp study. At
ε = 0.28 the soft reader moves fewer exact-fiber decisions (3.4–4.1%) than the sharp one did
(10–11%). It moves less and gains more. The lift of the mass the reading moves at changed
decisions again leads with the **5-5** and **6-4** (+0.05 / +0.04 at the zeb table), then 5-0,
4-1, 4-4: lumps-voi's tiles.

## Caveats

- **One ε.** It was fitted to Zeb under the uniform-within-set model and used for both
  likelihoods. The consistent likelihood is not a normalized model of play, so its ε is not
  its own ML. Neither ε nor δ was swept.
- **The net table's seats are weak** (28% random). Its effect is of reading a noisy field
  correctly, not of reading strong players.
- **Restricted choice** is null at the true model with 192 pairs. Its interval allows roughly
  ±0.2 marks/game, so an effect smaller than that is not excluded.
- **The soft − sharp and A/A comparisons were added after the measured runs.** The sharp data
  are [[reads-plays-2026-10-13]]'s, from an earlier commit with the same plain Zeb bytes.
- **Two of the six pre-registered effects (both restricted choice) are nulls, one is null at
  the net table (consistent − plain).** Three pass. Read the three that pass as three looks at
  one mechanism, not three independent findings: the two Zeb arms share their deals.
- **No Zeb default changed.** `Config.reading` stays off; whether to ship it in Zeb is a decision
  for Jason. The soft draw has no early exit: it evaluates every world of an enumerated
  fiber (up to 2^20) and, at tricks 1–2, a median of 2.7k–47k rejection proposals. Its
  per-decision cost against plain was not timed here.

## Replication

The Zeb-table result has to replicate before reading can be a default. This
section was written and committed before any measured replication batch.

### Pre-registration

Fixed in `scenario/reads_soft.py` (`REPLICATION`, `REP_*`):

- **Fresh deals.** Base seeds 101–118 (marks games) and 201–209 (games to 250).
  None overlaps the seeds the study above used (1–7, and its pilot's 900–904).
  The replication's own timing pilots (990, 991) are excluded.
- **The same players and table.** Two consistent readers (ε = 0.28, margin 2^22)
  against two plain deployed Zebs `k0,n512,F=margin_d250k_h128x64,f=16`, with
  `bid30` bidders and paired halves.
- **The A/A arm** runs in the same run on the same seeds: plain Zeb with planner
  seed 1 against plain Zeb.
- **Size.** 18 batches × 16 game pairs = **288 game pairs** per marks arm, and
  9 × 16 = 144 game pairs of games to 250. The batch size came from the pilot:
  24 pairs risked the 300 s cap with two streams running. Every batch runs
  under `timeout 300`. A batch that overruns is dropped, not rerun, and the
  page will say so.
- **Primary.** The consistent reader's margin against plain Zeb, in marks per
  game (a game pair's sum / 2). The 95% bootstrap runs over game pairs (4,000
  resamples, seed 42). **Success means the interval's lower bound is above 0.**
- **Registered secondaries.**
  - The uniform reader's margin, in marks per game.
  - The A/A margin, which should cover 0.
  - Settled points per game for each reader.
  - The consistent reader's share of games won in the book's game to 250
    points (chance 0.5).
- **Exploratory, not registered.**
  - Consistent − A/A and uniform − consistent, on shared pairs.
  - A sweep of the consistent reader's ε at 0.1 and 0.5, on the first 6 seeds
    (96 pairs).
  - The time per decision (`cost`).

### Results

*Pending: the batches run after this commit.*

## Data

Public in `jasonyandell/texas-42-zeb-research-2026-10` under `zeb/reads-plays-soft/` (CC0,
`SHA256SUMS`, `NOTES.md` lists the columns): `fit.json`; per batch `meta.json`, `games.jsonl`,
`decisions.jsonl.gz`; `summary.json`, `ess_net.json`, `ess_zeb.json`, `run_all.sh`. Recompute
([publish-data-to-huggingface](../playbooks/publish-data-to-huggingface.md)):

```bash
NET=scratch/field_net/v3/margin_d250k_h128x64.npz
python -u -m scenario.reads_soft fit --pour scratch/research-night/empirical/pour40 --net $NET
timeout 300 python -u -m scenario.reads_soft arena --net $NET --table net --arm uniform --seed 1 --games 48 \
    --out runs/net_uniform_s1          # run_all.sh: net plain|consistent|uniform s1-4; zeb consistent|uniform s1-7
timeout 300 python -u -m scenario.reads_soft arena --net $NET --table zeb --arm plain --seed 1 --games 24 \
    --out runs/zeb_plain_s1            # the A/A control, s1-7
python -u -m scenario.reads_soft summarize --runs runs/* --out .
python -u -m scenario.reads_soft ess --net $NET --runs runs/net_uniform_s* runs/net_consistent_s* --every 50 --out ess_net.json
```

Code: the soft reading at mk5-main `77a3b1dd` (`Reading.eps`, both engines; arena
`r=<likelihood>:<margin>:<eps>`). The study at `7188cd90` (pre-registration), plus the A/A
arm. Tests: `scenario/tests/test_reading.py` (referee = core under ε > 0 on every draw path,
the soft posterior never empties, ε = 0 is the sharp set, margin 0 is F_σ, Silas's
restricted-choice counts at any ε) and `scenario/tests/test_reads_soft.py` (the net seat plays
the model, the table, the pairing).

## Links

[[reads-plays-2026-10-13]] · [[lumps-voi-2026-10-13]] · [[scenario-player]] · [[silas]] ·
[[arena]] · [[strategy-fusion-fence]]
