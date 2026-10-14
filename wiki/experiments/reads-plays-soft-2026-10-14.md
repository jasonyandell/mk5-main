---
title: Zeb reads the plays through a soft set — it pays at a net table; against Zeb the registered replication failed for consistent and held for uniform (2026-10-14)
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
- **Replication** (§ Replication, registered, 256 fresh game pairs): the consistent reader
  scored **+0.12 [−0.02, +0.27]**. That **fails** the registered bar (lower bound > 0).
  Uniform held at +0.23 [+0.08, +0.37], over 240 pairs. A/A came in at −0.04
  [−0.17, +0.09]. The consistent reader won 0.531 [0.503, 0.559] of games to 250. The gain
  against Zeb is smaller than the first run said, and in the primary form it is not
  established.
- **Cost:** up to 1.7× plain Zeb's median per decision at tricks 2–3, and nothing measurable
  from trick 4 on (native core, one thread). The WebAssembly package cannot reach the reading.

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
  per-decision cost is timed in § Replication (up to 1.7× plain's median at tricks 2–3).

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

**The registered primary did not replicate.** The consistent reader against plain Zeb was
**+0.12 [−0.02, +0.27] marks/game** over 256 fresh game pairs. The lower bound is below 0,
so by the bar fixed in advance this is a failure. The point estimate is positive and about
two fifths of the original +0.30. The uniform reader, a registered secondary, held up:
+0.23 [+0.08, +0.37]. The A/A arm sits at chance, as it should.

Run at `c9852cf6` (the pre-registration commit), 2026-10-14, 05:06–06:47.

| registered effect | game pairs | estimate | original (seeds 1–7) |
|---|---|---|---|
| **primary: consistent vs plain, marks/game** | 256 | **+0.12 [−0.02, +0.27]: fails the bar** | +0.30 [+0.14, +0.47] |
| consistent vs plain, settled points/game | 256 | +6.0 [+0.2, +11.8] | +11.7 [+5.3, +18.2] |
| uniform vs plain, marks/game | 240 | **+0.23 [+0.08, +0.37]** | +0.29 [+0.10, +0.46] |
| uniform vs plain, settled points/game | 240 | +7.4 [+1.4, +13.3] | +13.7 [+6.2, +20.7] |
| A/A: plain at seed 1 vs plain, marks/game | 288 | −0.04 [−0.17, +0.09] | −0.03 [−0.20, +0.15] |
| consistent vs plain, share of games to 250 won | 144 | 0.531 [0.503, 0.559] (chance 0.5) | not measured |

**Dropped batches.** Six batches overran `timeout 300` and were dropped under the registered
rule, not rerun. Five were registered: consistent at seeds 114 and 116, and uniform at
seeds 113, 114 and 116. One was exploratory: ε = 0.5 at seed 102. Every overrun came while
other sessions held the machine at a load average of 65–90 on 18 cores. A batch's outcome
cannot depend on its wall time, because every player is deterministic given the state. So
dropping a batch loses power but does not bias the result. The A/A arm and the
games-to-250 arm lost nothing. After 06:11 the exploratory sweep was stopped by hand to
protect the registered arms. It was resumed after they finished.

Per batch, the consistent reader's margin was positive at 11 of 16 seeds. The uniform
reader's was positive at 14 of 15, and the A/A arm's at 7 of 18. Contracts made when
bidding were 38.5% for the consistent reader and 37.4% for plain in the same games. For the
uniform reader they were 39.0% and 37.0%, and in the A/A arm 38.0% and 38.3%.

**Exploratory** (not registered):

| effect | game pairs | marks/game |
|---|---|---|
| consistent − A/A, same pairs | 256 | +0.18 [+0.01, +0.34] |
| uniform − consistent, same pairs (restricted choice) | 240 | +0.12 [−0.04, +0.29] |
| original + replication pooled, consistent | 424 | +0.20 [+0.09, +0.30] |
| original + replication pooled, uniform | 408 | +0.25 [+0.14, +0.37] |
| original + replication pooled, A/A | 456 | −0.03 [−0.14, +0.07] |

The pooled rows include the original's 168 pairs, which were the reason for replicating. A
pooled estimate therefore carries their winner's curse, and the replication's own
intervals are the honest ones.

**What it means.** The reading's gain against Zeb is smaller than #145 reported. The
consistent form, the pre-registered primary, did not clear 0 on fresh deals. The uniform
form did clear it, at +0.23 marks/game. It is the arm the net table's primary also used. In
this run the uniform form led the consistent one by +0.12, but restricted choice is still
null on the same pairs, as it was in #145. Uniform passing here is a registered secondary.
It is a reason for a uniform-primary replication, not a finding that the uniform form pays
on its own.

#### The ε sweep (exploratory)

The consistent reader at ε = 0.1 and ε = 0.5, at the first six replication seeds. Each batch
holds 16 game pairs. One batch was dropped (ε = 0.5 at seed 102). The arms were run in two
sittings: stopped by hand at 06:11, then resumed in one stream at 06:49 once the registered
arms were done.

| ε | game pairs | vs plain, marks/game | − (ε = 0.28) on the same pairs |
|---|---|---|---|
| 0.1 | 96 | +0.11 [−0.13, +0.34] | +0.06 [−0.18, +0.30] |
| 0.28 (registered, all seeds) | 256 | +0.12 [−0.02, +0.27] | |
| 0.5 | 80 | +0.21 [−0.03, +0.48] | +0.20 [−0.06, +0.49] |

Every interval covers 0. At this size, a sweep can tell neither ε from 0.28. Nothing here
argues for refitting ε.

#### Cost per decision

`python -u -m scenario.reads_soft cost --runs <consistent s101–105> --per-trick 40
--py-per-trick 4`. It draws 40 reader decisions per trick from the replication's consistent
arm (the reader's seat, a choice of at least two tiles). At each state it times plain Zeb,
the consistent reader and the uniform reader in turn, on one thread, at 512 worlds. The
first 4 per trick are also timed on the Python referee. The machine was **not quiet**: the
load average was 30 at the start and 24 at the end, on 18 cores, with other sessions
running. Read the ratios, not the absolute times.

Compiled core, ms per decision, median / p95:

| trick | plain | consistent | uniform | consistent / plain (median) |
|---|---|---|---|---|
| 1 | 126 / 304 | 143 / 336 | 147 / 328 | 1.14 |
| 2 | 75 / 292 | 127 / 311 | 153 / 328 | 1.69 |
| 3 | 44 / 89 | 67 / 181 | 60 / 188 | 1.53 |
| 4 | 31 / 52 | 28 / 53 | 31 / 56 | 0.92 |
| 5 | 0.1 / 9 | 1.4 / 9 | 1.4 / 11 | (sub-ms) |
| 6 | 0.1 / 0.3 | 0.2 / 0.7 | 0.2 / 0.7 | (sub-ms) |

On the Python referee, with 4 per trick (so only indicative), the medians at tricks 1–3 were
0.92 / 0.47 / 0.22 s for plain and 1.01 / 0.66 / 0.43 s for consistent.

The reading costs most at tricks 2–3, the rejection-draw tricks: about 1.5–1.7× plain's
median there. The p95 changes by at most about 10% at tricks 1–2, but doubles at trick 3
(89 to 181 ms). From trick 4 on there is no measurable cost. Every decision's p95 stays under
about 0.35 s on one native thread.

**The deployed budget.** No per-decision time budget is written down. The nearest thing is
the zeb README's latency table. `@jasonyandell/zeb` in WebAssembly takes 44 ms (160 worlds,
the phone default) or 154 ms (512 worlds) for an opening lead, on one M5 Max thread. Its
README calls 160 worlds "a mobile budget". **The reading cannot be reached from the package
today.** `zeb/src/lib.rs` builds `Config { reading: None, draw: Draw::Uniform, .. }`, and
`Player` exposes no option for it. So the WebAssembly cost is unmeasured. Scaled by these
native ratios (an estimate, not a measurement), the reader would put a trick-1 decision near
0.18 s at 512 worlds, or 0.05 s at 160. At tricks 2–3 it costs up to 1.7× a base that is
already smaller than the opening lead's. Exposing it would be a new option and an API change for Jason to
decide on. Nothing here changes a default.

## Replication 2

The first replication failed its consistent primary and passed its uniform secondary. This run
asks the uniform reader alone, as the primary, on fresh deals. It is the last run of this arm
unless it passes. This section's pre-registration was written and committed before any
measured batch.

### Pre-registration

Fixed in `scenario/reads_soft.py` (`REPLICATION_2`, `REP2_*`):

- **Fresh deals.** Base seeds 301–354 (marks games) and 401–424 (games to 250). None overlaps a
  seed of #145 (1–7, pilot 900–904) or of the first replication (101–118, 201–209, pilots 990
  and 991); the published runs of both were checked. The timing pilot (995, 996) is excluded.
- **The same players and table.** Two uniform readers (ε = 0.28, margin 2^22) against two plain
  deployed Zebs `k0,n512,F=margin_d250k_h128x64,f=16`, with `bid30` bidders and paired halves.
  The A/A arm (plain Zeb at planner seed 1 against plain Zeb) runs at the same seeds in the
  same run.
- **Primary.** The uniform reader's margin against plain Zeb, in marks per game (a game pair's
  sum / 2). The 95% bootstrap runs over game pairs (4,000 resamples, seed 42), as before
  (`replicate --primary uniform`). **Success means the interval's lower bound is above 0.**
- **Registered secondaries.**
  - The uniform reader's share of games won in the book's game to 250 points (chance 0.5).
  - The A/A margin, which should cover 0.
  - The uniform reader's settled points per game.
- **Size and load.** 54 batches × 6 game pairs = **324 game pairs** per marks arm, and 24 × 6 =
  144 pairs of games to 250. Another heavy experiment shares the machine, so at most 6 batches
  run at once. The pilot put 8 pairs at about 120 s with three batches running; 6 pairs aims
  at under 150 s.
- **Batch-drop rule.** Every batch runs under `timeout 300`. A batch that exits nonzero (an
  overrun exits 124) is dropped, not rerun, and listed here. The effects use the game pairs
  present. No effect is computed until every batch has run: no early stop, no top-up.
- **Exploratory, not registered.** Uniform − A/A on shared pairs; the per-batch signs; a pool
  with the earlier uniform runs. That pool carries their winner's curse, since their passing
  is why this run exists.

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
the model, the table, the pairing, the replication's effects).

The replication is published in the same repo under `zeb/reads-plays-soft-rep/`. It has
`SHA256SUMS` and a `NOTES.md`, and holds:

- per batch: `runs/zeb_<arm>[_e<ε>]_s<seed>/` and `runs/zeb250_consistent_s<seed>/`
- `replication.json`
- `cost.json`
- `run_rep.sh` and `run_sweep_resume.sh`
- `logs/progress.log`, which records every batch's exit and load average, the overruns
  among them

Pre-registered at `c9852cf6`.

```bash
python -u -m scenario.reads_soft replicate --runs runs/* --out replication.json
python -u -m scenario.reads_soft cost --net $NET --runs runs/zeb_consistent_s10[1-5] --per-trick 40 \
    --py-per-trick 4 --out cost.json
```

## Links

[[reads-plays-2026-10-13]] · [[lumps-voi-2026-10-13]] · [[scenario-player]] · [[silas]] ·
[[arena]] · [[strategy-fusion-fence]]
