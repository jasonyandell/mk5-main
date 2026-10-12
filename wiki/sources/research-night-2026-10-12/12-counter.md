# Counter: the counterexample lab

*Research night 2026-10-12, COUNTER role. Every number is exact unless marked. Each item is one command from `scratch/research-night/counter/` (all in `run_all.sh`), each under 5 minutes wall; outputs in `out/`. Assembled by the orchestrator from the role's numbered messages (the subagent report-file guard refused its own writes).*

## Verdict in six lines

1. **T1 holds in the lab.** Exact Walt with the coherent belief matched brute-force O1 on every root action at 50 random information sets. Coherent replanning realized its plan exactly: execution gap 0 on 150 of 150 positions.
2. **C1 is real and can be total.** In an 8-tile game Walt-uniform puts the count on a trick the field will overtrump: it makes 0%, the O1 move makes 100%. A 12-tile instance with a 420-world posterior loses 0.174 P(make).
3. **C2a: near-ties flip at deployed N.** A 0.011 gap flips with exact probability 0.234 at N = 512; a 0.05 gap with 0.0018. Expected cost is gap × flip probability, about 0.003 P(make) per close decision.
4. **C2b: the floor is a trade.** With exact beliefs it can only lose, and rarely does: 0.0009 mean P(make) at the root for f = 16. Under small-N sampling it cuts the clairvoyance loss (0.0965 to 0.0548 at N = 3 on the best position for it).
5. **T2: fixed beliefs stabilize; policy-dependent beliefs can cycle.** Every fixed-belief ladder settled well inside the R−1 bound. With coherent or ε-tilt beliefs at every rung, the mini-42 ladder has certified cycles. Some are strict: every change forced, no ties.
6. **C3: the gap is real in the abstract but did not appear in mini-42.** Note 1's tables check out (gap 0.97 at m = 99). In mini-42, bad mutual best responses exist, but the Walt partnership map reached O2 in all 1,840 priors tried. I could not construct an instance it lands on. That is a finding: one-seat-at-a-time Walt with off-path replanning escaped every bad fixed point the lab could build.

## The lab

`lab.py` (~600 lines) is exact enumeration, Python standard library only.

**Mini-42.** Dominoes over a small pip range; tile id = index in the sorted list, so the tie rule "lowest tile id" matches `scenario/planner.py`. Four seats, every tile dealt. A tile carrying the trump pip is trump; otherwise a led tile belongs to its higher pip. Follow suit if able. Highest trump wins, else highest tile of the led suit; within a suit the double is high, then the other pip. A trick is worth 1 plus its count. Bidders are seats 0 and 2. Utility `make` (0/1) or `points` (bidder margin). Bidders maximize, defenders minimize (bidder-team units, as the planner).

| game | tiles | per seat | trump | count | total | bid |
|---|---|---|---|---|---|---|
| mini42-8 | 11 20 21 22 30 31 32 33 | 2 | 3 | 3-2 = 5 | 7 | 4 |
| mini42-12 | 11 21 22 30 31 32 33 40 41 42 43 44 | 3 | 4 | 4-1 = 5, 3-2 = 5 | 13 | 7 |
| mini42-12-shown | as mini42-12; each seat publicly holds one of 44 / 43 / 33 / 22 | 3 | 4 | same | 13 | 7 |

Seat 0 leads. The deal prior is uniform over all deals unless stated; small-support priors are explicit deal lists, equal weight.

**The field F** (`heuristic_field`), a deterministic net-like map from information set to tile:
- lead: top trump if any, else highest double, else lowest tile;
- follow, partner winning: smear the highest-count legal tile;
- follow, partner not winning: win as cheaply as possible, else play the lowest-count legal tile.

**The Walt operator** (`WaltSolve`): exact over a weighted scenario list. Max (or min) at the viewer's turns with one action per cell; belief-weighted sum at other seats' turns, the cell split by the tile seen; exact leaves. Duplicate scenarios count separately (as sampled Walt does). Floor as in the planner: below the root, a viewer cell with fewer than f scenarios plays the hand-down move (F here; the net in Zeb).

**Beliefs.** *uniform*: prior × void-consistency, the fiber Zeb samples from. *coherent*: uniform × the product of F's probability of each observed play by another seat; with deterministic F a 0/1 filter on F-consistent worlds. *tilt*: the planner's `Tilt(field)` form, weight ε/n + (1−ε)·[F plays it], ε = 0.1.

**The ladder** (`ladder.py`): P^-1 = uniform random legal play (exact expectation, i.e. the dice integrated out); P^k = exact Walt against P^{k-1} at every other seat. Belief rule per rung: fixed (uniform fiber), coherent (likelihood under P^{k-1}; fallback to the fiber when that mass is 0, i.e. off path), or tilt.

**Ground truth.** `o1_enumerate`: every legal contingent viewer policy from the information set, evaluated exactly. `extract_plan` + `eval_policy`: a Walt plan read off the memo and evaluated as a legal policy. O2: full pair enumeration, or every seat-0 policy × seat 2's exact best response.

**T1 spot check** (`python -u sanity.py`, 1.8 s): random mid-hand information sets, field F or uniform random.

| game | information sets | root actions where Walt-coherent Q ≠ brute-force O1 |
|---|---|---|
| mini42-8 | 25 | 0 |
| mini42-12 | 25 | 0 |

## C1: uniform belief is not O1; coherent is

### Instance A, total loss (`python -u c1_show.py`, under 1 s)

Game: mini42-8, uniform deal, field F at seats 0, 1, 3. Viewer: **seat 2 (bidder) holding 3-1 3-2**. Record: partner (seat 0) led the 1-1; LHO (seat 1) played the 2-0, showing out of 1s.

| | worlds |
|---|---|
| void-consistent fiber (what Zeb samples) | 9 |
| F-consistent | **1**: s0{11 21} s1{20 22} s2{31 32} s3{30 33} |

The filter reads two plays. Partner led a non-trump double, so under F it holds no trump and no 2-2. LHO sluffed instead of trumping, so under F it holds no trump. Hence RHO holds both remaining trumps, including the 3-3.

| root move | Walt-uniform Q | brute-force max, uniform belief | Walt-coherent Q | **O1 (brute force, true posterior)** |
|---|---|---|---|---|
| 3-1 | 4/9 | 4/9 | **1** | **1** |
| 3-2 | **7/9** | 7/9 | 0 | 0 |

Policies. Walt-uniform: play the 3-2. O1: play the 3-1; everything after is forced (one contingent policy per root move). With the 3-2 on trick 1, RHO overtrumps with the 3-3 and takes 6 of 7: set. With the 3-1, RHO's 3-3 takes 1 point, RHO leads the 3-0, the 3-2 overtrumps and wins the count: 6 of 7, made.

**Reading.** The hand makes 100% with the right play and 0% with Zeb's belief. The information was on the table in two field plays: a lead that was not a trump and a sluff that was not a trump-in. Both plays were legal from every hand, so the void filter keeps the bad worlds; only "the field would have played it" removes them.

### Instance B, a big posterior (`c1_show.py`)

Game: mini42-12, uniform deal, field F. Viewer: **seat 1 (defender) holding 2-2 4-0 4-3**. Seat 0 led the 4-2. Under F seat 0 leads its top trump, so it lacks the 4-4: 140 of 560 fiber worlds removed, 420 remain.

Brute force is infeasible (1.8·10^72 policies). Each Walt-coherent plan was extracted and evaluated as a legal policy: both realized their Q to 6 decimals, over 490 and 510 viewer information sets. With T1 (theory role's proof) they are O1.

| root move | Walt-uniform Q | Walt-coherent Q = O1 |
|---|---|---|
| 4-0 (duck) | 11/28 = 0.3929 | **4/21 = 0.1905** |
| 4-3 (cover) | **173/560 = 0.3089** | 51/140 = 0.3643 |

Values are P(bidders make); the defender minimizes. Walt-uniform covers with the 4-3; coherent Walt ducks, because the 4-4 is behind it. Uniform Walt's whole plan, executed against the true posterior, makes 0.3643 where it believed 0.3089.

**Reading.** Covering costs **73/420 = 0.174 P(make)**: the defense sets 64% of the hands instead of 81%. The forecast was wrong as well as the move.

### How often, and what it costs ex ante (`python -u c1.py g8`, 0.6 s; `python -u c1.py g12 300`, 99 s)

One seat plays Walt-uniform or Walt-coherent at every decision; the other three play F. Values are P(bidders make); lower is better for defenders.

| game | seat | all F | Walt-uniform | Walt-coherent (O1 BR) | decision sets | sets where moves differ |
|---|---|---|---|---|---|---|
| mini42-8, all 2,520 deals | 0 bidder | 0.4421 | 0.5317 | 0.5317 | 28 | 0 |
| | 1 defender | 0.4421 | 0.4317 | **0.4302** | 72 | 6 |
| | 2 bidder | 0.4421 | 0.4750 | **0.4758** | 266 | 7 |
| | 3 defender | 0.4421 | 0.4337 | 0.4337 | 521 | 8 |
| mini42-12, 300 sampled deals | 0 | 0.4800 | 0.5200 | 0.5200 | 354 | 8 |
| | 1 | 0.4800 | 0.4700 | **0.4667** | 351 | 22 |
| | 2 | 0.4800 | 0.4900 | **0.4933** | 366 | 21 |
| | 3 | 0.4800 | 0.4633 | **0.4600** | 370 | 25 |

**Reading.** Coherence never hurts, since it is the best response. In 2- and 3-trick games it moves the whole-hand make rate by 1 to 3 hands per 1,000. The cost is concentrated in a few percent of decisions, some of them total losses. These games are too short to size C1 in real 42; that is the empirical role's measurement with the real net.

### Note 1 §7: the execution gap (`python -u delta_exec.py 150`, 55 s)

Δ_execution = V_plan − V_execute, both under the root belief and field F. V_execute replans with fresh Walt (same belief rule) at every later decision. Positions: 150 mini42-12 roots in trick 1, after 0 to 3 field plays.

| belief rule | mean Δ (P(make)) | positions with Δ > 0 | max | min |
|---|---|---|---|---|
| uniform (later replans forget what F revealed) | 0.0013 | 33 / 150 | 0.0208 | 0 |
| coherent | **0** | **0 / 150** | 0 | 0 |

**Reading.** Note 1's prediction, measured. Uniform replanning sometimes fails to realize its own plan, by up to 2 hands in 100, and never by a negative amount. Coherent replanning realizes its plan exactly.

## C2a: finite sampling flips near-ties (`python -u c2a.py`, 7 s)

Sampling model = `scenario/fiber.py`: N iid uniform draws when the fiber has more than N worlds; otherwise the fiber is enumerated and the flip probability is exactly 0. The chosen decisions have no later discretionary viewer choice, so Q̂(a) is a mean over the draws and the sampled argmax depends only on the counts of outcome types: an exact multinomial. Ties go to the lowest id. Game: mini42-12, field F, uniform belief.

Probability that the sampled move differs from the exact move:

| instance (exact Q, bold = exact choice) | N=1 | 4 | 16 | 32 | 64 | M−1 | M |
|---|---|---|---|---|---|---|---|
| seat 0 (bidder), 2-1 3-0 left, after 4-3 4-0 4-1 4-2; M = 90; 2-1 = 4/9, **3-0 = 41/90**; gap 0.0111; types (u21,u30): (0,0) 45, (1,1) 36, (0,1) 5, (1,0) 4 | 0.944 | 0.821 | 0.619 | 0.538 | 0.468 | 0.436 | **0** |
| seat 1 (defender), 4-1 4-4 left, after 2-1 4-2 2-2 1-1; M = 60; 4-1 = 0.15, **4-4 = 0.10**; gap 0.05 | 0.900 | 0.702 | 0.429 | 0.307 | n/a | 0.202 | **0** |
| seat 1 (defender), 3-0 3-3 left, after 4-3 4-4 4-0 4-2; M = 90; 3-0 = 0.589, **3-3 = 0.389**; gap 0.20 | 0.800 | 0.410 | 0.028 | 0.0008 | 0.0000 | 0 | **0** |

Real 42 fibers at trick 1 hold about 10^9 worlds, so Zeb always samples. Same per-world outcome laws in a fiber far larger than N, exact trinomial over the per-world difference:

| N | gap 0.0111 | gap 0.05 | gap 0.20 |
|---|---|---|---|
| 16 | 0.619 | 0.429 | 0.028 |
| 64 | 0.468 | 0.188 | 0 |
| 160 | 0.375 | 0.060 | 0 |
| **512** | **0.234** | **0.0018** | 0 |
| 1024 | 0.141 | 0 | 0 |
| 2048 | 0.060 | 0 | 0 |
| 4096 | 0.013 | 0 | 0 |

**Certificate.** Note 1's Δ > 2δ, applied to the paired per-world difference d in [−1, 1] with Hoeffding at 95%, needs N ≥ 2·ln 20 / Δ².

| gap | N for the certificate | normal-approx N for a 5% flip |
|---|---|---|
| 0.0111 | 48,531 | 2,190 |
| 0.05 | 2,397 | 160 |
| 0.20 | 150 | 11 |

That is for a decision with no later choice. For a contingent plan the certificate must cover every candidate policy (note 1), which adds the log of the policy count.

**Reading.** At N = 512 Zeb takes the wrong side of a one-hand-in-90 call about a quarter of the time. A flip costs at most the gap, so the expected loss is gap × flip probability: 0.0026 P(make) here. Under the normal approximation (not exact) the worst case over gaps is about 0.17·σ/√N, roughly 0.003 P(make) per close decision at N = 512 with σ ≈ 0.35. Sampling error is cheap per decision and lives where the call is close anyway. Note the discontinuity: once N reaches the fiber size the fiber is enumerated and the flip probability drops to exactly 0.

## C2b: the floor

### Exact beliefs: the floor can only lose, and rarely does (`python -u c2b.py 1`, 126 s)

Walt over the whole coherent posterior; interior viewer cells with fewer than f worlds play F's move. Positions: 300 mini42-12 roots in trick 1, field F. "Root loss" = O1 value minus the value of the floored root move with optimal continuation, which is what a replanning player pays. "Plan loss" = cost of executing the floored plan, handdowns included, to the end of the hand.

| floor | root move changed | mean root loss | positions with root loss | max root loss | mean plan loss |
|---|---|---|---|---|---|
| 4 | 12 / 300 | 0.0002 | 9 | 0.0167 | 0.0038 |
| 16 | 21 / 300 | **0.0009** | 14 | **0.1053** | 0.0081 |

Worst case at f = 16: **seat 2 (bidder) holding 4-0 4-1 4-3 after 1-1, 4-4**, posterior 19 worlds. The exact plan plays the 4-3 and makes 0.4737. The floored plan plays the 4-1, forecasts 0.3684 and makes 0.3684: **−0.105 P(make)**.

### Sampled beliefs: the clairvoyance the floor bans (`python -u c2b.py 2`, 2 s)

A one-world cell's max is perfect-information play for that world. Showcase: mini42-12, **seat 3 (defender) holding 4-0 4-1 4-4 after 3-3 3-0 2-2**, coherent posterior 6 worlds. Results are exact over every multiset of N draws.

| | 4-0 | 4-1 | 4-4 |
|---|---|---|---|
| exact O1 Q (P(make); defender minimizes) | 0.500 | **0.000** | 1.000 |
| mean one-world (clairvoyant) Q | **0.000** | 0.000 | 1.000 |

| N | floor | P(root ≠ O1 move) | E[O1 loss] |
|---|---|---|---|
| 1 | none | 1.000 | 0.500 |
| 1 | 2 | 0.500 | 0.250 |
| 2 | none or 2 | 0.500 | 0.250 |
| 4 | any | 0.125 | 0.0625 |

Clairvoyance turns a coin-flip guess after the 4-0 into a sure set.

Aggregate over 39 mini42-12 positions (posterior 5 to 12 worlds, a later viewer choice, non-flat exact Q), exact over all multisets (`python -u c2b_part2.py`, 9 s):

| N | floor | mean P(root ≠ O1 move) | mean O1 loss |
|---|---|---|---|
| 1 | none | 0.272 | 0.0839 |
| 1 | 2 | 0.263 | 0.0804 |
| 2 | none | 0.191 | 0.0512 |
| 2 | 2 | 0.187 | 0.0505 |
| 3 | none | 0.145 | 0.0345 |
| 3 | 2 | 0.142 | 0.0340 |
| 3 | 3 | 0.138 | 0.0335 |

Best position for the floor: seat 3 holding 1-1 4-1 4-4 after 2-2 2-1 3-2, at N = 3. The loss falls from 0.0965 (no floor) to 0.0756 (f = 2) and 0.0548 (f = 3). Its forecast is 0.000, an optimistic "sure set", in all three cases, where the truth is about 0.05 to 0.10.

**Reading.**
- With exact beliefs the floor is a pure tax: about 1 hand in 1,000 at f = 16, with rare positions costing 1 in 10.
- With thin samples it guards against clairvoyance: it halves the N = 1 loss on the showcase.
- Neither effect is large in these tiny posteriors. The hub's measured "floor buys marks" is a statement about the sampled regime, and these numbers point the same way.
- The floor's missing certificate would be "F's move is within ε of the cell's max". The floor never checks that.

## T2: fixed beliefs stabilize; policy-dependent beliefs can cycle

### Fixed beliefs: confirmed inside the bound

| game | command, wall | sets checked | last rung at which a move changed | bound R−1 | violations |
|---|---|---|---|---|---|
| mini42-8, every decision set of every deal | `python -u t2.py g8 8`, 3 s | 1,984 | 0 for all | 3 | 0 |
| mini42-12-shown, 19 sampled roots, rungs 0 to 9 | `python -u t2.py g12s 9 20`, 253 s | 19 | 0 (18), 2 (1) | 7 | 0 |

The 12-tile run uses a deterministic hashed base: an information-set-measurable stand-in for the tape. The T2 proof makes the limit independent of the base. In mini42-8 the Q values move between P^0 and P^1 at 253 sets (max |ΔQ| 0.333), but no argmax moves (`python -u t2_check.py`, 1 s), so the operator does read the rung below. In the same 12-tile run with coherent beliefs at every rung, the 19 roots also settled (last change at rung 7, within bound).

**Reading.** Real depth is far below the bound, as note 2 predicts: the deepest change was rung 2 against a bound of 7.

### Policy-dependent beliefs: certified cycles

Every rung's player weights its fiber by the rung below's likelihood: *coherent* (fallback to the fiber off path) or *tilt* (the deployed `Tilt(field)` form, ε = 0.1, which never needs a fallback). Certificate: P^{k+p} == P^k on every decision information set of the game; P^{k+1} depends only on P^k, so the ladder is periodic forever.

| game | belief | tie rule | result | command, wall |
|---|---|---|---|---|
| mini42-8, make | coherent | lowest id | **period 4** from rung 3; 15 sets move | `t2_cycle.py coherent 0.1 lowest`, 6 s |
| mini42-8, make | coherent | highest id | **period 12** from rung 3 | `t2_cycle.py coherent 0.1 highest`, ~15 s |
| mini42-8, make | coherent | sticky | settles at rung 3 | `t2_cycle.py coherent 0.1 sticky` |
| mini42-8, make | ε-tilt | lowest id | **period 4** from rung 0; 10 sets move; no fallback ever | `t2_cycle.py tilt 0.1 lowest`, 8 s |
| mini42-8, make | ε-tilt | highest / sticky | settles at rung 1 / 3 | `t2_cycle.py tilt 0.1 {highest,sticky}` |
| mini42-8, **points**, trump 3 | coherent | **sticky** | **period 4** from rung 4; 3 sets move; every change strict | `t2_strict.py 3 points coherent`, 20 s; `t2_strict2.py`, 3 s |
| mini42-8, make, **trump 0** | **ε-tilt** | **sticky** | **period 6** from rung 3; 24 sets move; every change strict; no fallback | `t2_strict.py 0 make tilt`, 30 s; `t2_strict3.py`, 5 s |

Sticky = on a tie keep the previous rung's move. A change then happens only when the incumbent becomes strictly worse, so no incumbent-keeping tie rule can stand these ladders still. The sticky sweep (`python -u t2_search.py {0..3}`, 50 to 70 s each) cycles in 8 of 16 distinct (trump × utility × belief) configurations; leader rotations are relabelings and give identical rows.

**Mechanism** (ε-tilt, lowest id; `python -u t2_trace.py`, 1 s). Note 1's hidden matching pennies, played as a lead whose meaning flips:
1. P^{k-1} leads the 3-1 only from 3-1 3-2. So a 3-1 lead tells seat 1 (holding 3-0 3-3) that the 3-2 is behind (weight 0.83), and seat 1 ducks with the 3-0.
2. Against a ducking seat 1, leading the 3-1 is better for every holding: 0.367 vs 0.333 from 2-0 3-1. So P^k leads the 3-1 from everything.
3. Now the lead says nothing (0.20 each), and seat 1 covers with the 3-3.
4. Leading the 3-1 drops back to a tie with the 2-0; the lowest id takes the 2-0. Back to step 1.

Policies on the cycle: seat 0 holding 2-0 3-1 leads 20 20 31 31 20 20 ...; seat 1 holding 3-0 3-3 after the 3-1 lead plays 33 30 30 33 33 30 ...; ex-ante P(make) of the all-seat profile cycles 0.4952, 0.4905, 0.5000, 0.4952.

The strict points/coherent cycle has the same shape. Seat 2 (bidder) holding 3-0 3-1 after 2-1 2-2 alternates which trump it plays, period 4, margins ±0.286 at every rung. Seat 3 holding 1-1 3-3 alternates overtrump and duck. When seat 2's play is off path under the rung below, seat 3 falls back to the fiber and misreads it: a bluff that works for one rung. The ex-ante bidder margin oscillates −0.0421 / −0.0429. The strict ε-tilt cycle (trump 0) has margins of 0.004 in P(make) at the moving sets and no fallback anywhere.

**Reading.**
- T2's finite stabilization needs the fixed-belief hypothesis. This is measured, not conjectured.
- Deployed Zeb is safe: its modeled minds use uniform beliefs, and the tilt, when on, touches only the live rung, which is never iterated.
- The repair for O1 against the net needs no ladder. F is the net and fixed, so one coherent Walt application is the best response (T1).
- Cycles appear only if coherent beliefs are pushed into every rung. If that is ever done, use the sticky tie rule and a rung cap; sticky alone does not prevent cycles.

## C3: the partnership gap

### Note 1's tables, checked (`python -u c3.py 2 5 150`, part A, under 1 s)

S/R table: (S,S) = ε, (R,R) = 1, mixed = 0. At (S,S) each unilateral switch drops ε to 0: an O1 fixed point for both partners; O2 = 1.

(m+1)-action family, c = 3/(m+1): action 0 with itself pays c, action 0 against any other pays c/2, (1,1) pays 1, all else 0.

| m | c | unique best response to uniform | unique best response to action 0 | fixed-point value | O2 | gap |
|---|---|---|---|---|---|---|
| 9 | 0.30 | action 0 (0.165 vs 0.115) | action 0 | 0.30 | 1 | **0.70** |
| 99 | 0.03 | action 0 (0.0151 vs 0.0101) | action 0 | 0.03 | 1 | **0.97** |

At m = 2 the formula gives c = 1, which is degenerate; the family needs m ≥ 3.

### Mini-42: bad fixed points exist, but Walt's map never stops at one

The Walt partnership map alternates exact Walt-coherent replanning for seat 0 then seat 2 (tie rule lowest id; off-path fallback to the prior fiber), with opponents (seats 1, 3) fixed at F, until the pair repeats.

| experiment | command, wall | priors | a mutual best-response pair below O2 | Walt map fixed point below O2 | Walt map cycled |
|---|---|---|---|---|---|
| mini42-8, 5-deal equal-weight priors, start (F, F) | `c3.py 2 5 150`, 13 s | 150 | 27 (worst gap 0.4) | **0** | 0 |
| same, random starting conventions | `c3.py 3 5 400 random`, 104 s | 400 | 82 (worst gap 0.6) | **0** | 0 |
| mini42-12 pooling-trap priors (≤ 3 deals; partner's hand fixed; leader's hands differ by one tile swapped with an opponent) | `c3_g12.py 1 3`, 260 s; `c3_g12.py 2 3 90`, 90 s | 947 + 343 | not enumerated | **0** | 0 |
| mini42-8, full uniform deal (2,520 deals) | `c3_full.py 4`, 7 s (+70 s two-lead search) | 1 | — | no improving one-lead or two-lead joint deviation (378 pairs); fixed point 101/180 = 0.5611, reached in one round from (F, F) at 0.4421 | 0 |

O2 is exact in the first three rows: full pair enumeration (mini42-8) or every seat-0 policy against seat 2's exact best response (mini42-12). In the full game the joint-deviation search gives only a lower bound on O2, and it found no gap. Bids 3 to 6 give the same full-game result, because in mini42-8 they all reduce to "capture the 3-2 trick". Sanity check on the 12-tile search: O2 beats (F, F) in 141 of 343 priors, and the Walt fixed point beats (F, F) in the same 141, reaching O2 every time.

Example of a bad pair (seen in an earlier c3.py version; gap 0.2, reasoned rather than re-run). Seat 0 holding 1-1 3-0 leads the 3-0; seat 2 holding 2-0 3-2, after 1-1 2-1, would trump in with the 3-2. Each move is a best response to the other only because the 1-1 lead never occurs, so seat 2's bad answer is never tested. Replanning seat 2 at that unreached set with the fallback belief favours the 2-0, after which leading the 1-1 pays. The measured rows of the table are the evidence.

**Reading.** This is a finding, not a failure. The bad mutual best responses in mini-42 all stand on a partner's move at an information set the pair never reaches. Walt replans at every information set, off path included through the fallback, and that sequential rationality removes them. In all 1,840 priors tried, one-seat-at-a-time Walt reached the partnership optimum, and I could not construct a mini-42 instance it lands on.

**The barrier.** In 42 actions are public and sequential, so a partner sees the lead before answering. A strict trap needs a convention: a hand-to-lead map whose informative version pays only if the partner's off-path reading changes at the same time (note 1's S/R structure). The pooling search targeted exactly that and found none in 1,290 priors. That does not show there are none in real 42: with 7 tricks, counts and real signalling, the structure is far more available than in 2 or 3 tricks.

## What I could not do

- **O3 (team maxmin):** not computed. Pure pair-versus-pair enumeration on the smallest priors is about 4,096 × 4,096 × 5 playouts, and correlated team mixing needs an LP I did not build.
- **A mini-42 C3 instance that Walt lands on:** none found (above).
- **C1 at real-42 scale:** the lab's games are 2 to 3 tricks; real-42 size against the real net is the empirical role's job.
- **Brute-force O1 for instance B:** 10^72 policies; verified as plan-realizes-Q plus T1 instead.
- **Tie-rule independence of the cycles:** the sticky rule certifies every change is forced; I did not prove that every tie rule cycles. About 60% of mini-42 decisions have tied Q values because the game is small.

## Files

- `lab.py`: game, fields, beliefs, `WaltSolve`, brute-force O1, plan extraction.
- `games.py`: the three instances. `ladder.py`: the rung ladder (fixed / coherent / tilt beliefs; lowest / highest / sticky tie rules; uniform or hashed base).
- Experiments: `sanity.py`, `c1.py`, `c1_show.py`, `delta_exec.py`, `c2a.py`, `c2b.py`, `c2b_common.py`, `c2b_part2.py`, `c3.py`, `c3_full.py`, `c3_g12.py`, `t2.py`, `t2_check.py`, `t2_cycle.py`, `t2_trace.py`, `t2_search.py`, `t2_strict.py`, `t2_strict2.py`, `t2_strict3.py`.
- `run_all.sh`: one command per item. `out/`: every output quoted above.

COUNTER DONE (assembled 02:16 CDT).
