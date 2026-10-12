# Empirical report: C1 and C2 measured on real 42 positions (research night 2026-10-12)

Assembled by the orchestrator from the empirical role's messages (the subagent
report-file guard refused its own writes). Code, JSON results and logs are in this
directory. Every run used `python -u` and `PYTHONPATH=/Users/jason/code/mk5-main`, and
finished under 5 minutes. Finished 02:05 CDT.

The player is deployed Zeb throughout: `k0,n512,F=margin_d250k_h128x64,f=16`. The
utility is make, the bid is 30, and seat 0 bids.

## The answers first

1. **With a deterministic F, the coherent posterior is usually empty in Zeb self-play.**
   - The field net picks Zeb's actual tile on only 58% of discretionary plays, against
     the wiki's held-out 53%.
   - The true deal goes F-inconsistent at the first modeled play: median play index 1,
     in 40 of 40 deals.
   - After play 0, the true deal is F-consistent at only 88 of 674 positions.
   - From play 12 on, most 512-world samples hold zero consistent worlds.
   - So O1 with a 0/1 likelihood is not defined at a self-play table. F must be
     softened, or the field must really be F.
2. **When the field really is F, uniform-over-fiber Zeb is measurably not an O1 best
   response.** Zeb sat in one seat and the net played the other three, over 2,120
   paired deals.
   - Planning over worlds drawn from beta^F gains **+0.028 marks per hand** (SE 0.006).
   - It flips **120 hands in Zeb's favor and 61 against**, sign test p = 1.4e-5.
   - The gain is the same bidding (+0.027) or defending (+0.028).
   - An independent uniform redraw on the same code path is null: 18 flips for and 19
     against.
   - This is C1, with real 42, the real net and the real planner.
3. **The repair has to sample from beta^F.** Reweighting the deployed 512 worlds does
   not work. Scored on an independent beta^F draw:
   - a fresh beta^F draw improves the chosen move by +0.53 ± 0.13 pmake points per
     decision;
   - keeping only the consistent worlds of the same 512 gives +0.00 ± 0.15.
   - The consistent fraction at plays 8 to 11 is about 4%, roughly 20 worlds. This is
     the wiki's effective-sample-size lesson at its exact 0/1 limit.
4. **The note-1 certificate Delta > 2·delta was never violated, but it almost never
   fires.** It certifies 8 of 1,230 decisions after play 0. The median 2·delta is 1.8,
   while the median Delta is 0.4 pmake points.
5. **Exact planning bounds execution, as T1 predicts, and the floor breaks that
   bound.** This is E2, on 300 late positions whose root sample is the whole fiber.
   - At floor 1, the plan's value was at least the replanning player's value at all
     300. It was strictly above at 11, by at most 2 pmake points.
   - At floor 16, execution beat the plan at 16 positions, because the floored plan
     optimizes over a restricted policy class.
6. **Sampling noise moves the move but not the value.** A fresh 512-world uniform draw
   changes the opening lead in 66 of 160 hands, and 238 of 1,390 decisions overall.
   The value cost is −0.09 ± 0.06 pmake points per decision, which is null.
7. **The floor changes the root move in 101 of 714 positions, including 25 of 40
   openings.** This is E3. Without the floor, the opening plan forecasts 15 pmake
   points higher.
8. **The softened posterior is null in self-play, where deployed Zeb lives.** One seat
   importance-resampled 512 worlds from 8,000 uniform draws under F at eps 0.5, against
   three deployed Zebs, over 1,200 paired deals. It gained +0.010 ± 0.008 marks per
   hand, with 53 flips for and 41 against, p = 0.26. Its effective sample stayed
   healthy, about 1,400 worlds at plays 8 to 11. It neither hurt, as the old
   tilt-reweighting did, nor helped.

## Definitions and method

- **Consistent world.** Replay the record from the world's dealt hands, which are its remaining hands plus the tiles each seat has played. A world is consistent when, at every non-viewer discretionary play, `FieldNet.move` would pick the tile actually played, deciding from that seat's own information. That is the same `features` function the planner's modeled seats use. The viewer's own plays are never filtered, and forced plays (one legal tile) always pass. Code: `consistency.py`.
- **c and delta.** c is the number of consistent worlds among the deployed 512. delta = 1 − c/512, which is exactly TV(uniform on the 512, uniform on its consistent subset).
- **Planning over a given world list.** The core accepts weights but not a world list. `subset.py` subclasses the referee `Walt` and replaces the live rung's sample, keeping weights at 1 and the same tapes, floor and tie rule. Checked first: given the core's own 512 worlds, it reproduced the core's tile and every q exactly at 4 positions (`probe2.py`). It takes about 1 s per decision.
- **The 0/1 reweight R** goes through the core's belief seam, `CompiledWalt(belief=...)`. It is not the same plan as the subset solve, because the floor counts zero-weight worlds as live. Both are reported.

### How the beta^F worlds were drawn (setting B, item 2)

Two regimes, set by the size of the void-consistent fiber at the decision:

1. **The fiber has at most 30,000 completions.** Enumerate it (`scenario.fiber.enumerate_fiber`) and filter every completion through the net. This gives the exact posterior. If more than 512 completions are consistent, keep a uniform random 512, drawn with `random.Random(seed*1000 + plays)`; otherwise keep them all. This regime covered 706 of the 1,390 analyzed decisions, nearly all at play 12 or later. 47 were subsampled down to 512. The median kept is 3 worlds, because the exact posterior is tiny late in the hand.
2. **The fiber is larger.** Draw batches of 2,000 exact-uniform completions (`scenario.fiber.sample_fiber`, the counting-DP sampler) and filter each batch. Stop at 512 consistent worlds or 40,000 draws, and keep the first 512 found. When fewer than 512 were consistent after 40,000 draws, plan over all of those found. That happened at 71 of 684 such decisions; the fewest kept was 9, and the median number of draws was 4,000.

Either way, the kept worlds are uniform over the F-consistent fiber, because rejection from a uniform pool is exact. No decision ever found zero consistent worlds: the true deal is always consistent in setting B, and the fallback to the deployed core fired 0 times across all F lines. Below the root, the planner floors cells thinner than 16 worlds, the deployed semantics. The root is never floored.

### Commands

All commands run from the repo root with `PYTHONPATH=/Users/jason/code/mk5-main`. E stands for `scratch/research-night/empirical`. The pour directory named in the brief, `scratch/viz/pour_viz_5/`, does not exist in this checkout, so I re-poured. The five-deal pour at seed 424242 (`E/pour5`, `E/e1a_pour5.json`, 10 s) shows the same picture as the 40-deal one below.

## Setting A: Zeb self-play pour, F as a model of Zeb

```
python -u -m scenario.atlas pour --deals 40 --spec k0,n512,F=scratch/field_net/v3/margin_d250k_h128x64.npz --seed 424242 --out E/pour40   # 4.6 s
python -u E/e1a.py E/pour40 E/e1a_pour40.json      # 78.6 s
python -u E/agree.py E/pour40                      # under 5 s
```

714 planned positions from 40 deals:

| plays | positions | mean c/512 | median c/512 | c = 0 | true deal consistent | certified |
|---|---|---|---|---|---|---|
| 0 | 40 | 1.000 | 1.000 | 0 | 40 | 39 |
| 1-3 | 71 | 0.259 | 0.209 | 2 | 31 | 0 |
| 4-7 | 117 | 0.117 | 0.035 | 27 | 32 | 1 |
| 8-11 | 115 | 0.045 | 0.002 | 56 | 17 | 0 |
| 12-15 | 115 | 0.009 | 0.000 | 87 | 4 | 0 |
| 16-19 | 126 | 0.003 | 0.000 | 117 | 3 | 0 |
| 20-27 | 130 | 0.002 | 0.000 | 128 | 1 | 0 |

How often the net picks Zeb's actual tile, on the true hands' discretionary plays:

| plays | 0 | 1-3 | 4-7 | 8-11 | 12-15 | 16-19 | 20-27 | all (714) |
|---|---|---|---|---|---|---|---|---|
| agree | 0.53 | 0.70 | 0.62 | 0.55 | 0.54 | 0.52 | 0.63 | 0.58 |

At the 257 positions after play 0 with at least one consistent world, here is how often re-planning over those worlds changes the move:

| c | positions | changed, subset re-plan | changed, 0/1 reweight |
|---|---|---|---|
| 1-7 | 74 | 35 | 33 |
| 8-31 | 68 | 39 | 38 |
| 32-127 | 58 | 24 | 26 |
| 128-512 | 57 | 9 | 10 |

In self-play the deterministic-F posterior works as a misspecification detector, not as a belief. Each of these changes is a plan over a handful of worlds, and the real deal is usually not among them (see E1d).

## Setting B: a well-specified field (Zeb in one seat, the net in three)

Deals are `scenario.atlas.deal(seed)`, declared by the arena's `_declared_trump`, bid 30, with seat 0 bidding. Zeb is seat 0 (the bidder) on even seeds and seat 1 (a defender) on odd seeds. Because the true deal is always F-consistent here, beta^F is the exact O1 posterior.

The three lines of `e1b.py` (usage: first_seed n_deals out.json workers):
- **U** is deployed Zeb on the compiled core, planning over the uniform 512.
- **V** is the referee on an independent uniform 512 draw, from the same sampler with no filter. It is the noise control on F's code path.
- **F** is the referee on the beta^F worlds described above.

The U line also opens up every discretionary decision when the analysis flags are set.

```
E1LINES=U EXTRA=1 python -u E/e1b.py 900000 320 E/e1b_analysis_320.json 16   # 226.8 s
E1LINES=UVF       python -u E/e1b.py 900000 320 E/e1b_lines_a.json 16        # 215.1 s
E1LINES=UVF       python -u E/e1b.py 900320 320 E/e1b_lines_b.json 16        # 179.7 s
E1LINES=UF LEAN=1 python -u E/e1b.py 900640 480 E/e1b_lines_c.json 16        # 79.0 s
E1LINES=UF LEAN=1 python -u E/e1b.py 901120 1000 E/e1b_lines_d.json 16       # 286.2 s
python -u E/e1b_report2.py E/e1b_analysis_320.json
python -u E/lines_report.py E/e1b_lines_a.json E/e1b_lines_b.json E/e1b_lines_c.json E/e1b_lines_d.json
```

The runs are deterministic. An earlier U/F run on seeds 900000-900319 (`E/e1b_320.json`, 112 s) reproduces every play of the U and F lines in batch a.

One earlier attempt, 1,400 deals from seed 901120, hit my 295 s timeout before writing its JSON. I discarded it and kept its log as `E/e1b_lines_d_TIMEDOUT.log`. The script now writes a JSONL line per deal, and the 1,000-deal rerun is batch d.

### E1 (a) and (d): consistency and the certificate (1,390 Zeb decisions, U line, 320 deals)

| plays | decisions | mean c/512 | median c/512 | c = 0 | certified (Delta_U > 2·delta) | beta^F enumerated / rejection | mean beta^F worlds | beta^F worlds < 16 |
|---|---|---|---|---|---|---|---|---|
| 0 | 160 | 1.000 | 1.000 | 0 | 152 | 0/160 | 512 | 0 |
| 1-3 | 97 | 0.374 | 0.340 | 0 | 0 | 0/97 | 512 | 0 |
| 4-7 | 230 | 0.249 | 0.137 | 6 | 7 | 1/229 | 480 | 0 |
| 8-11 | 226 | 0.086 | 0.043 | 9 | 0 | 49/177 | 409 | 5 |
| 12-15 | 226 | 0.054 | 0.021 | 19 | 0 | 205/21 | 163 | 69 |
| 16-19 | 220 | 0.071 | 0.033 | 3 | 0 | 220/0 | 17 | 178 |
| 20-27 | 231 | 0.235 | 0.167 | 0 | 1 | 231/0 | 2 | 228 |

The certificate's inputs:

| quantity | value |
|---|---|
| median Delta_U (best minus runner-up, mover's pmake) | 0.0039 |
| median 2·delta | 1.81 |
| decisions with Delta_U = 0 | 557 of 1,390 |
| of those, every move ties | 483 |
| of those, the contract is already decided in every world | 447 |

**Soundness.** There were 8 certified decisions after play 0. Neither the 0/1 reweight nor the subset re-plan changed the move at any of them. The 152 certified openings are trivial, because nothing has been played and delta = 0. The 8 uncertified openings are exact ties.

**Usefulness.** The certificate certifies 8 of 1,230 decisions after play 0, which makes it vacuous in practice. Two-thirds of the uncertified decisions do not change under the repair either, so the TV bound over the whole sample is far from tight.

### E1 (b): does the move change? Changes against deployed Zeb's move aU

| plays | V: fresh uniform 512 (noise) | C: consistent subset of the same 512 | R: 0/1 reweight of the same 512 | F: fresh beta^F draw |
|---|---|---|---|---|
| 0 | 66/160 | 0/160 | 0/160 | 65/160 |
| 1-3 | 29/97 | 21/97 | 21/97 | 30/97 |
| 4-7 | 68/230 | 85/224 | 82/224 | 94/230 |
| 8-11 | 41/226 | 85/217 | 84/217 | 82/226 |
| 12-15 | 29/226 | 55/207 | 52/207 | 61/226 |
| 16-19 | 5/220 | 40/217 | 40/217 | 41/220 |
| 20-27 | 0/231 | 8/231 | 8/231 | 8/231 |
| all | 238/1390 | 294/1353 | 287/1353 | 381/1390 |

C and R have no row for decisions where c = 0, which is why their totals are 1,353.

### E1 (c): which change is right? O1 regret, scored on an independent beta^F draw

q2 is the action value from planning over a second, independent beta^F draw. It is drawn the same way as F but from a different random stream. Scoring on q2 avoids the winner's curse of scoring a move on the sample that chose it. Values are in the mover's units, as pmake points ×100, ± one SE.

| plays | decisions | q2(aF) − q2(aU) | q2(aV) − q2(aU), noise | q2(aC) − q2(aU), subset |
|---|---|---|---|---|
| 0 | 160 | −0.10 ± 0.10 | −0.14 ± 0.10 | 0.00 |
| 1-3 | 97 | +0.01 ± 0.08 | −0.08 ± 0.12 | −0.35 ± 0.21 |
| 4-7 | 230 | +0.39 ± 0.11 | −0.05 ± 0.09 | −0.26 ± 0.19 |
| 8-11 | 226 | +0.42 ± 0.15 | +0.06 ± 0.10 | −0.97 ± 0.36 |
| 12-15 | 226 | +0.84 ± 0.23 | −0.39 ± 0.31 | −0.06 ± 0.30 |
| 16-19 | 220 | +1.72 ± 0.74 | −0.01 ± 0.01 | +1.47 ± 0.77 |
| 20-27 | 231 | 0.00 | 0.00 | 0.00 |
| all | 1390 | **+0.53 ± 0.13** | −0.09 ± 0.06 | +0.00 ± 0.15 |

**Reading.** The beta^F draw picks better moves from play 4 on. Re-planning over the consistent worlds of the same 512 does not, because it is planning over about 20 worlds at plays 8 to 11. The repair must sample from beta^F, not reweight the uniform sample.

### E1 (c), continued: the chosen move's forecast with the plan held fixed and the belief repaired

These are the 1,353 decisions with c > 0. The forecast is aU's splat over all 512 worlds, against the same splat restricted to the consistent subset.

| plays | decisions | pmake, uniform 512 | pmake, consistent subset | mean absolute shift | margin, uniform 512 | margin, consistent subset |
|---|---|---|---|---|---|---|
| 0 | 160 | 0.424 | 0.424 | 0.000 | +7.43 | +7.43 |
| 1-3 | 97 | 0.334 | 0.323 | 0.063 | +4.48 | +3.88 |
| 4-7 | 224 | 0.418 | 0.431 | 0.110 | +8.34 | +9.31 |
| 8-11 | 217 | 0.423 | 0.433 | 0.129 | +8.16 | +8.67 |
| 12-15 | 207 | 0.446 | 0.467 | 0.115 | +9.34 | +9.59 |
| 16-19 | 217 | 0.491 | 0.494 | 0.118 | +11.02 | +10.82 |
| 20-27 | 231 | 0.476 | 0.468 | 0.059 | +9.45 | +9.38 |
| all | 1353 | 0.439 | 0.445 | 0.090 | +8.70 | +8.89 |

Pmake is bidder-team. The shifts run in both directions and average out to almost nothing.

## E1d: does the real deal survive the net filter? (Setting A)

This uses the 40-deal self-play pour, where all four seats are deployed Zeb. At each of the 714 planned positions, the record was replayed from the real hands. The real deal survives when every other seat's non-forced play equals the net's choice. Here m is the number of other-seat non-forced plays so far.

```
python -u scratch/research-night/empirical/e1d.py scratch/research-night/empirical/pour40 scratch/research-night/empirical/e1a_pour40.json   # under 1 s (reuses e1a's 78.6 s run)
```

| m | positions | real deal survives | 0.53^m | 0.58^m |
|---|---|---|---|---|
| 0 | 40 | 1.000 | 1.000 | 1.000 |
| 1 | 56 | 0.589 | 0.530 | 0.580 |
| 2 | 51 | 0.294 | 0.281 | 0.336 |
| 3 | 46 | 0.217 | 0.149 | 0.195 |
| 4-5 | 107 | 0.178 | 0.062 | 0.091 |
| 6-8 | 170 | 0.059 | 0.012 | 0.023 |
| 9-11 | 131 | 0.008 | 0.002 | 0.005 |
| 12-21 | 113 | 0.000 | 0.000 | 0.001 |

| plays so far | 0 | 1-3 | 4-7 | 8-11 | 12-15 | 16-19 | 20-27 |
|---|---|---|---|---|---|---|---|
| real deal survives | 1.000 | 0.437 | 0.274 | 0.148 | 0.035 | 0.024 | 0.008 |
| mean m | 0 | 1.6 | 3.2 | 5.3 | 7.5 | 9.8 | 12.3 |

- **The theory's prediction holds in shape but runs low.** Survival decays geometrically, but slower than 0.53^m from m = 3 on: 18% against 6% at m = 4-5, and 6% against 1% at m = 6-8. The net's agreements with Zeb are correlated within a deal, so survival is not independent per play.
- **By mid-hand the real deal is essentially always excluded.** It is gone at all 113 positions with m of 12 or more, and at 99% from play 20.
- **The filter is undefined at most positions.** 417 of the 674 positions after play 0 have no consistent world among the 512. The real deal survives at none of those 417.

### The filter's confidence against the truth

The forecast below is the bidder-team pmake of the move Zeb actually played, at the 257 positions after play 0 with at least one consistent world. The truth is whether the bidder made the contract. Lower Brier is better.

| real deal | positions | real make rate | uniform forecast | filtered forecast | Brier, uniform | Brier, filtered | filtered says 0 or 1 | of those, wrong |
|---|---|---|---|---|---|---|---|---|
| survives | 88 | 0.386 | 0.343 | 0.333 | 0.175 | **0.149** | 24 | 1 |
| dies | 169 | 0.414 | 0.378 | 0.389 | **0.155** | 0.163 | 85 | 9 |

- **When the real deal survives, the filter helps.** Its Brier score is 0.149 against uniform's 0.175, and only 1 of its 24 certain forecasts is wrong.
- **When the real deal dies, the filter is a trap.** It is worse calibrated than uniform, at 0.163 against 0.155. It calls the hand certain at 85 positions and is wrong at 9 of them, about 11%.
- **In self-play the filter is mostly a trap.** The real deal has died at 169 of the 257 positions where the filter is defined at all. The repair only works when the field really is F, which is setting B, where the gain was +0.028 marks per hand.

### Realized outcomes: 2,120 paired deals (the headline)

Each deal is played by every line. Marks are Zeb's side's marks per hand: +1 when a hand flips Zeb's way, meaning made instead of set when Zeb bids, or set instead of made when Zeb defends. The sign test counts the flipped hands.

| comparison | Zeb's role | deals | bidder made (first line vs second) | Zeb marks per hand | flips for / against | sign p | points per hand |
|---|---|---|---|---|---|---|---|
| F vs U | bids | 1060 | 0.452 vs 0.425 | +0.027 ± 0.009 | 63 / 34 | 0.004 | +0.41 ± 0.39 |
| F vs U | defends | 1060 | 0.351 vs 0.379 | +0.028 ± 0.009 | 57 / 27 | 0.001 | +0.42 ± 0.30 |
| F vs U | both | 2120 | | **+0.028 ± 0.006** | **120 / 61** | **1.4e-5** | +0.42 ± 0.25 |
| V vs U (noise) | both | 640 | | −0.002 ± 0.010 | 18 / 19 | 1.0 | +0.23 ± 0.40 |
| F vs V (same code path) | both | 640 | | +0.028 ± 0.011 | 34 / 16 | 0.015 | +0.50 ± 0.46 |

The effect varies by batch. Batches a, b, c and d give 21/5, 15/14, 29/14 and 55/28 flips, all positive or flat.

In Jason's units: against a field that really plays like the net, coherent beliefs turn about 3 hands in 100 Zeb's way, from either chair.

### The softened F_eps posterior in self-play (E1c, where deployed Zeb lives)

One seat builds its belief like this:
1. It takes 8,000 uniform fiber draws, or the whole fiber when that is no larger.
2. It weights each world by the product, over the other seats' discretionary plays, of eps/n + (1 − eps)·[the net picks the played tile], with eps = 0.5.
3. It draws 512 systematic resamples by those weights and plans over them on the referee.

The other three seats are deployed Zeb. The comparison line puts deployed Zeb in all four seats, on the same deals.

```
python -u E/e1c.py 910000 600 0.5 8000 E/e1c_eps05.json 16     # 160.9 s
python -u E/e1c.py 910600 600 0.5 8000 E/e1c_eps05_b.json 16   # 167.1 s
python -u E/lines_report.py E/e1c_eps05.json E/e1c_eps05_b.json
```

| Zeb's role | deals | Zeb marks per hand | flips for / against | sign p |
|---|---|---|---|---|
| bids | 600 | +0.008 ± 0.013 | 32 / 27 | 0.60 |
| defends | 600 | +0.012 ± 0.010 | 21 / 14 | 0.31 |
| both | 1200 | +0.010 ± 0.008 | 53 / 41 | 0.26 |

The importance weights keep a healthy sample through mid-hand. Median effective sample size, from the first 600 deals:

| plays | 1-3 | 4-7 | 8-11 | 12-15 | 16-19 | 20-27 |
|---|---|---|---|---|---|---|
| ESS / draws | 0.48 | 0.33 | 0.19 | 0.15 | 0.20 | 0.40 |
| ESS | 3,852 | 2,645 | 1,428 | 443 | 25 | 4 |

**Reading.** Sampled properly, the softened posterior neither hurts, as the old tilt-reweighting of 160 draws did (−0.65 marks/game in the wiki), nor helps measurably. The net agrees with the deployed player only 58% of the time, so the likelihood carries little usable signal in self-play. The setting-B gain comes from the model being right. This run was powered to see a setting-B-sized effect at about 2.3 SE.

## E2: the execution gap on late positions where the root sample is the whole fiber

The positions are Zeb's discretionary decisions at play 12 or later on setting B's U line (Zeb in one seat, the net in three). Only positions whose root fiber has at most 512 completions qualify, so Zeb's root sample is the whole fiber and its root belief is exactly uniform.

- **V_plan** is the root plan's q for its chosen move, as bidder-team pmake.
- **V_execute** plays every fiber world out from the same root move. Zeb replans each turn on the deployed core, with a fresh sample and the same floor, and the net plays the other three seats. The make indicator is averaged uniformly over the fiber.
- **D** = V_plan − V_execute, in the mover's units.

```
python -u E/e2.py E/e1b_320.json 16 12 300 E/e2_f16.json   # 12.2 s
python -u E/e2.py E/e1b_320.json 1  12 300 E/e2_f1.json    # 9.7 s
python -u E/e2_report.py E/e2_f16.json E/e2_f1.json
```

"Undecided" means 0 < V_plan < 1. Mean D is in pmake points ×100.

| floor | plays | positions | undecided | mean D | D > 0 | D < 0 | largest abs D |
|---|---|---|---|---|---|---|---|
| 1 | 12-15 | 24 | 11 | +0.16 ± 0.09 | 4 | 0 | 0.020 |
| 1 | 16-19 | 122 | 39 | +0.05 ± 0.02 | 7 | 0 | 0.017 |
| 1 | 20-27 | 154 | 25 | 0 | 0 | 0 | 0 |
| 1 | all | 300 | 75 | **+0.03 ± 0.01** | 11 | **0** | 0.020 |
| 16 | 12-15 | 24 | 11 | −0.30 ± 0.19 | 1 | 4 | 0.035 |
| 16 | 16-19 | 122 | 39 | −0.12 ± 0.05 | 2 | 12 | 0.034 |
| 16 | 20-27 | 154 | 25 | 0 | 0 | 0 | 0 |
| 16 | all | 300 | 75 | **−0.07 ± 0.03** | 3 | **16** | 0.035 |

**Reading.**
- **Floor 1 matches theory.** The plan is an exact max over the viewer's policies under the root belief against the net. The replanning policy is one of those policies, so V_plan ≥ V_execute, and that held at all 300 positions.
- **The strict gaps are note 1's Delta_execution.** They appeared at 11 positions, at most 2 pmake points. Later replans use a fresh uniform belief over the later fiber, not the root belief updated by the net's plays. That time inconsistency gives up value the root plan had found.
- **Floor 16 breaks the bound.** The floored plan hands thin cells to the net, which is a restricted policy class, so V_plan is no longer an upper bound. Execution beat the plan at 16 positions, by at most 3.5 pmake points.
- **Late in the hand both gaps are tiny.**
- **Margin is not compared.** Once the contract is decided, the planner's splat plays the rest of the hand from its random tape, so a margin comparison against execution would not be meaningful.

## E3: does the floor change the root move? (f = 16 against f = 1, same sample)

```
python -u E/e3.py E/pour40   # 20.3 s
```

The script asserts that the two configurations draw identical samples, and that the f = 16 tile equals the pour's chosen tile at every position.

| plays | positions | root move differs | f1 plan's preference for its own move (mover, pmake ×100) | f16 forecast minus f1 forecast (mover, pmake ×100) |
|---|---|---|---|---|
| 0 | 40 | 25 | +2.27 | −15.02 |
| 1-3 | 71 | 13 | +0.37 | −8.77 |
| 4-7 | 117 | 33 | +0.36 | −6.45 |
| 8-11 | 115 | 18 | +0.18 | −3.23 |
| 12-15 | 115 | 9 | +0.08 | −1.11 |
| 16-19 | 126 | 3 | +0.02 | −0.16 |
| 20-27 | 130 | 0 | 0 | 0 |
| all | 714 | **101** | +0.27 | −3.50 |

The floor changes the root move at 14% of positions and at 25 of 40 openings. The floorless plan is 15 pmake points more optimistic at the opening, which is the clairvoyance the floor exists to guard against.

## What this means for the proof

- **T1 needs a full-support F to say anything about deployed Zeb.** With the deterministic net, beta^F is empty in self-play: the real deal is excluded by mid-hand (E1d).
- **C1 holds in real 42.** Uniform-over-fiber is not an O1 best response against the very field it models. The cost is about 0.5 pmake points per decision, or 0.028 marks per hand, concentrated at plays 4 to 19 (setting B).
- **The affordable repair is sampling from beta^F, not reweighting.** Rejection from the exact-uniform fiber cost 2 to 6 s per decision on the Python referee, and would be cheap in the core.
- **In self-play the repair buys nothing measurable yet.** The softened version is null over 1,200 deals (E1c).
- **The certificate needs a sharper delta to be useful.** Delta > 2·delta with TV over the whole sample is sound, with no violations, but certifies 8 of 1,230 decisions after play 0. A usable one would bound belief error only on the cells that decide the move.
- **Sampling is a C2 gap in move identity but not in value.** A fresh 512 draw changes the opening lead in 41% of hands, at −0.09 ± 0.06 pmake points per decision.
- **The floor is a C2 gap in both.** It restricts the policy class, so V_plan stops bounding execution (E2), and it changes the root move at 14% of positions (E3).

## What I could not do

- **The core takes weights, not a world list.** Every plan over a chosen world list ran on the Python referee. It is bit-identical given the same worlds, but I checked that at 4 positions, not all of them.
- **E2 covers late positions only.** It used positions at play 12 or later with a root fiber of at most 512. I did not measure the execution gap earlier in the hand, where fibers are too large to enumerate.
- **I tested only one softened F.** E1c used eps = 0.5 with 8,000 draws. I tried no other eps or pool size.
- **Coverage is narrow.** Everything used bid 30, seat 0 bidding and the arena's trump rule, with Zeb only in seat 0 or seat 1.
- **The stated five-deal pour directory was missing.** I re-poured it from seed 424242 instead.
- **I could not write REPORT.md myself.** The harness blocks report files from subagents, so the lead is assembling it from these messages.

EMPIRICAL DONE (assembled 02:10 CDT).
