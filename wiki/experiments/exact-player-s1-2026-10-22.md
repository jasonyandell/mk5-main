---
title: "The exact player, S1: trick 4 and the probe's positions under a two-type mixture, every mixture parameter at once"
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Stage S1 of the exact-player track (Exact-1, 2026-10-22), after [[exact-player-s0-2026-10-22]].
The same player declares a family rather than a single model: two table types under a prior that
moves with a parameter p. One exact solve gives the information-state optimum for every p in
[0, 1] at once. The questions:

1. How far toward the hand's start can that be done exactly? Trick 4 means four own tiles.
2. Does the parameter matter to the act?
3. What does exactness cost beside Walt's recorded time on the same positions? This is data only,
   with different players solving different models.

## The family

- **lean-low.** Every other seat plays its lowest legal tile with probability ½ + 1/(2n), and
  every other legal tile with 1/(2n), where n is the number of legal tiles.
- **lean-high.** The same toward its highest tile.
- **The prior.** lean-low with 1 − p, lean-high with p.

Both laws have full support, so no record is impossible and the counted posterior is defined at
every p. The root is the counted posterior and settlement is `decided`. The act at p = ½ is
reported. `silas::family` gives each decision's values as piecewise rational functions of p,
together with every threshold.

Each decision has three checks:
- `--recount 3`: the second construction at the first, middle and last parameter classes;
- `--quotient`: the controller, computed only where the course has at most 64 classes;
- a budget of 1.2×10⁸ kernel reads.

The measured build is `6bfff632`.

## Pre-registration and amendments

`scratch/desks/family-2026-10-22/PREREG-S1.md`, approved by the orchestrator before either batch.
It has three amendments, each dated, and none made after a result in its scope was read.

1. **Amendment 1, the orchestrator's memory ruling.**
   - T4 runs serially.
   - Free memory means free + inactive + speculative pages.
   - A process that drives free memory under 10 GB is killed and counted as a memory refusal.
   - P runs in shards of 20.
   - The data goes to the existing research dataset.
2. **Amendment 2, P's cap and the dead throttle.** It was written during P, before any P result
   was read.
   - The first P launch queued all 304 shards at once, because the driver's `jobs -r` ran in a
     subshell. It was relaunched in 6 lanes, keeping 14 complete shards.
   - Every shard over the 300 s cap was rerun one position per process. 14 shards, or 280
     positions, were rerun, and one position stayed over the cap.
3. **Amendment 3, the T4 abort.**
   - At 09:54, with 45 of 147 done, t-abs (fiber 4,200) was killed as a memory refusal, and the
     batch aborted before t-abt. The pressure came from outside S1: Exact-GPU's Metal ladder for
     trick 1 held about 5 GB resident and about 4.4 GB was wired.
   - On the orchestrator's ruling, T4 resumed at t-abt in pour order, under the same guard.
   - Before any process the batch now holds, and never contends, until free memory has stayed at
     or above 10 GB for 5 minutes. Trick 1 has priority on the machine.
   - t-abs stays a memory refusal. Its peak RSS was not recorded, because `time -l` dies with a
     killed process. Later memory refusals record a sampled RSS.

## Results: batch P (the probe's 6,065 positions, deals 0–499 of both pours)

**Coverage and the gate.**
- 6,064 of 6,065 positions were solved exactly. The other, champion/465/16 at three own tiles,
  went over the 300 s cap as a single process. That clock is wall and load-dependent, so the
  refusal is not a property of the position (FAIR CLOCKS).
- **G1 holds everywhere.** The second construction agreed exactly at three parameter classes of
  every solved decision.
- **G6 holds.** 100 rows (every 61st), rerun one per process, are byte-identical apart from
  `wall_ns`.

**The course does not depend on the laws.** Course choices and the share decided at the root are
the same as S0's, position for position. At three own tiles, undecided, the median is 4,742 and
the 90th percentile 86,635. The likelihood doubles the kernel reads: two types, each replayed.

**The family's shape** (undecided decisions):

| pour, own tiles | n | course thresholds: median / p90 / max | root set constant in p | act at ½ differs from p = 0 or 1 |
|---|---|---|---|---|
| champion, 3 | 968 | 0 / 40 / 384 | 86.1% [83.7, 88.1] | 13.7% [11.7, 16.1] |
| zeb, 3 | 957 | 0 / 35 / 772 | 87.7% [85.4, 89.6] | 12.2% [10.3, 14.5] |
| champion, 2 | 736 | 0 / 0 / 1 | 93.3% [91.3, 94.9] | 6.5% [5.0, 8.5] |
| zeb, 2 | 764 | 0 / 0 / 1 | 95.4% [93.7, 96.7] | 4.6% [3.3, 6.3] |

Two definitions:
- A course threshold is an interior p at which some decision inside the plan changes its set.
- A root switch is a change of the root's maximizing set between consecutive parameter classes.
  This includes the tie sets at a threshold and at p = 0 or 1.

Every root switch sits next to a threshold or an endpoint. This was checked at all 6,851
decisions with a choice, across P and S0's batch A.

So for about seven decisions in eight at three tricks left, the act is the same whichever way the
table leans. For the rest, the act at even odds differs from what a seat sure of either type would
do.

**The controller** (where computed: at most 64 classes; 232 of 1,925 three-own-tile decisions
skipped and counted):

| own tiles | Reach Every, raw median / p90 / max | Reach Plan, raw median / p90 / max | fork max (Every, Plan) |
|---|---|---|---|
| 3 | 145–153 / 874–940 / 3,897 | 47–50 / 309–324 / 2,580 | 17, 9 |
| 2 | 3 / 3 / 5 | 2 / 2 / 4 | 4, 3 |

The ranges span the two pours.

**Instructions.** 1.91×10¹⁴ over 6,064 decisions, or 3.2×10¹⁰ each, pooled. That includes three
recounts and the controller. Instructions retired come from cpu-slot, which ran the solver directly.

**Speed beside Walt (data only).**
- Our side is the wall per decision for belief plus solve, measured in process. It is descriptive,
  and shards often ran at 0.65 of a core under contention, so it overstates our CPU.
- Walt's side is the probe's recorded `cpu_live_s` (its live rung) and `cpu_exact_s` (its twin at
  n = 4,096).
- Walt's times are in milliseconds, and about three in four two-own-tile positions read 0. Those
  are left out of the median ratio and counted; the total ratio keeps them.
- Bootstrap 95% intervals use 4,000 resamples, deals resampled together.

| pour, own tiles | median ours/live | total ours/live | median ours/exact | total ours/exact |
|---|---|---|---|---|
| zeb, 3 | 3.16 [2.99, 3.32] | 17.2 [13.9, 21.2] | 2.57 [2.40, 2.79] | 8.2 [6.7, 10.0] |
| champion, 3 | 0.92 [0.85, 1.00] | 2.84 [2.50, 3.19] | 0.83 [0.78, 0.92] | 1.80 [1.61, 2.00] |
| zeb, 2 | 0.87 [0.82, 0.89] | 1.31 [1.24, 1.38] | 0.65 [0.63, 0.68] | 0.97 [0.87, 1.07] |
| champion, 2 | 0.39 [0.38, 0.41] | 0.54 [0.49, 0.59] | 0.38 [0.37, 0.40] | 0.53 [0.49, 0.58] |

These are different players solving different models:
- Walt plans on the flat fiber against its net or its rung-0 self.
- This player solves the two-type mixture for every p on the counted posterior.

So the ratio says what exactness costs here, not which plays better. At the median position,
exact-for-every-p costs about what Walt's sampled planner costs, up to about 3× on the zeb pour's
three-own-tile positions. The heavy tail (the total ratio) is where it costs more, up to 17×.

**Against the predictions.**
- **Held:** G1.
- **Missed:**
  - **Median course thresholds.** Predicted 15; measured 0, with a 90th percentile of 35–40.
  - **Root set constant in p.** Predicted 75%; measured 86–88% at three own tiles, so p matters
    less than predicted.
  - **Speed.** Predicted 10× Walt's live median and 4× its exact; measured 0.9–3.2× and 0.8–2.6×.
    Exactness is cheaper than predicted at the median, and the pour matters.
- **Near:** the act at ½ differs from an endpoint's. Predicted 15%; measured 12–14% at three own
  tiles.

## Results: batch T4 (147 trick-4 decisions, deals 0–24 of both pours): PARTIAL

**Stopped on Jason's instruction** (relayed by the orchestrator and the Mayor, about 11:00): T4 is
written as it stands. It ran in pour order, so 76 positions were attempted: 76 of the zeb pour's
77, and none of the champion pour's 70. The 71 not run are not refusals. They are outside every
denominator below, and T4 says nothing about the champion pour.

**Outcomes by cause** (76 attempted):

| outcome | n | fiber: min / median / max | what decides it |
|---|---|---|---|
| solved exactly | 41 | 70 / 560 / 3,920 | |
| read budget (1.2×10⁸ reads) | 23 | 3,360 / 11,550 / 34,650 | deterministic: a property of the position and the budget |
| 300 s cap | 10 | 2,940 / 5,880 / 34,650 | wall, load-dependent (FAIR CLOCKS): **not** a property of the position |
| memory refusal | 2 | 4,200 / – / 4,410 | the machine's free memory, set by other sessions |

The two memory refusals were caused by external pressure, not by the solver:
- **t-abs** (09:54): its RSS was not recorded.
- **t-abx** (10:10): a sampled peak RSS of 21 MB.

**Coverage.**
- Of all 76 attempted: 41/76 = 53.9% [42.8, 64.7].
- Against the deterministic budget alone (solved over solved plus read-budget refusals):
  41/64 = 64.1% [51.8, 74.7].
- Every solved fiber is at most 3,920 worlds. The read budget refused fibers as small as 3,360.

**G1 holds** at three parameter classes of all 41 solved decisions. **G6 on T4 was not run**
(the stop came first): a gap, not a pass.

**The solved positions** (zeb only; median / max):
- course choices 2.7×10⁵ / 5.0×10⁶;
- instructions retired (`time -l`, recount and controller included) 5.2×10¹¹ / 5.1×10¹²;
- peak RSS 0.72 GB / 5.4 GB;
- process wall 27 s / 292 s (descriptive);
- decided at the root 7/41.

**The family's shape at trick 4** (34 undecided):
- course thresholds: median 50, p90 554, max 1,963;
- root set constant in p: 28/34 = 82% [66, 92];
- act at ½ differs from an endpoint's: 5/34 = 15% [6, 30].

**The controller**, where computed (15 of 34; 19 had more than 64 classes):
- Reach Plan raw: median 498, max 2,137;
- Reach Every raw: median 2,293, max 10,410;
- fork: max 27 (Plan) and 84 (Every).

**An infrastructure finding.** `scripts/cpu-slot`'s timeout terminates only its direct child.
T4 ran the solver under `/usr/bin/time -l`, so at the cap `time` died and the solver was
orphaned. It kept running past the cap, outside any slot. That was seen once, at t-acx, which
was killed by PID at the stop. It applies to every T4 cap refusal, and the orphans' memory
may have added to the pressure behind the memory refusals. The S0 and P shards ran the solver
directly and are not affected (#253).

**Against the predictions.**
- **Coverage.** Predicted 45% ± 8; measured 54% [43, 65] of the attempted, on the zeb pour only.
  The predicted mechanism missed: refusals begin near 3,000 worlds, not 12,000.
- **Among the solved.** Predicted median 5×10⁶ choices, 60 s and 6 GB; measured 2.7×10⁵, 27 s
  and 0.7 GB. The solved set is the small fibers.
- **Root set constant in p.** Predicted 70%; measured 82% [66, 92].
- **G6.** Not run.

## Data

[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
at `exact/s1-2026-10-22/`: batch P, its G6, T4 as it stands and the drivers (1,367 files, count and SHA verified).
