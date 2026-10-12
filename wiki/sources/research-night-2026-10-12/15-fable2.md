---
title: Research night 2026-10-12 — Fable session 2 (verbatim)
kind: source
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

# FABLE2 — Zeb as a mathematical object: what it is, what it is not, what it would take

*Research night 2026-10-12. Independent attempt from the brief and the two notes.
All code in this directory; every experiment under 5 min wall; nothing outside
`scratch/research-night/fable2/` touched. Positions come from two pours of the shipped
player (`k0,n512,F=margin_d250k_h128x64`, floor 16, utility make, bid 30, seed 424242):
`pour5/` (88 planned positions, the brief's five deals) and `pour40/` (714 planned
positions). The Python planner is the referee throughout; it matched the pour's `q`
to the bit at every position checked, and the frozen-plan evaluator reproduces `q`
exactly when fed the plan's own worlds (the validation is in `lab.py`'s first run).*

## 0. The answer in six lines

1. **Zeb is a one-step rollout of its own field net.** On the pour's own worlds, the
   plan decides the root and then *0.12 of the live seat's 2.7 remaining discretionary
   decisions*; 81% of worlds never see a second planned decision (E2). A player with the
   floor set above the sample (every interior own decision handed to the net) plays
   the *same tile* as shipped Zeb at 82% of leads, 93% of trick-1/2 positions, 96% of
   mid-hand positions and 100% of late ones, and the q-loss when it differs averages
   0.0007 pmake (E7). Zeb ≈ argmax over its legal tiles of "play this, then I am the
   net, everyone else is the net", averaged over 512 uniform worlds. That is the object.
2. **It is a legal policy and a sample-average approximation (SAA).** The floored plan
   is a function of (own hand, public record) alone — proved from the code in §1. Its root
   value is the SAA of a two-stage-ish stochastic program; the in-sample value of the
   chosen move is optimistic by **+1.2 pmake points** (se 0.3) in the first two tricks
   (E1), the move ranks differently on a fresh sample **38%** of the time, and the
   out-of-sample regret of the chosen move is **0.5 pmake points** on average, 4.5 at
   worst (E1).
3. **The deployed move is not certifiable at 512 worlds, and will not be at 2–3 s.** The
   median lead has a gap of 1.1 pmake points to the runner-up, decided by 12% of the
   worlds; an empirical-Bernstein certificate at 95% needs ~10,000 worlds at the median
   lead; 512 certifies **0%** of leads (E8). "Finds the optimal move in finite time" is
   true for exact Walt (T1, theory's) and false as a certificate for sampled Zeb at any
   affordable N; what is true is that the uncertified move's regret is below a point.
4. **O1 with the deployed (deterministic) field is a degenerate target.** The fraction of
   uniform worlds from which the net would have produced the observed record is 17%
   after trick 1, 2.7% mid-hand, 0.2% late; **in 418 of 714 positions no sampled world
   is net-consistent**, and the *true deal* is net-inconsistent 66% / 91% / 98% of the
   time by phase (E3). β^F for a deterministic F is uniform on an almost-empty set. The
   belief repair of note 1 §2 needs a *stochastic* field before it is even defined; the
   coherent version is priced in §5.
5. **The floor is nearly free where the fiber is exact, and the plan is honest.** At 306
   positions whose whole fiber was enumerated (≤ 17,640 completions), floor 1 is the
   exact uniform best response (T1 instance) and floor 16 plays the same tile **99%** of
   the time, regret 0.0001 mean, 0.024 max; pooled with 43 more mid-size fibers, 349
   positions and 6 moves differ (E6). Forecast vs realized against the
   net itself: the in-sample q of the chosen move is **2.7 points *below*** what
   replanning Zeb realizes (E4) — note 1's Δ_execution is *negative* for Zeb, because
   the plan's policy class (net below the floor) is weaker than the replanning player.
6. **"Data stops at parity" is the ladder stabilizing.** Zeb beats its own net as the live
   seat by **+1.3 pmake points** (se 0.8; 42 vs 22 paired worlds) (E4). Expert iteration
   `net_{t+1} = imitate(R[net_t])` is approximate policy iteration; its fixed point is a
   profile P with P ≈ R_U[P] under the *uniform* belief — exactly note 1's "stabilized but
   not optimal" object, with the k→∞ ladder reached by imitation instead of nesting.
   More of the same data cannot move it; only a different operator can (§5).

The rest proves, measures or marks each line.

## 1. The object, exactly

Notation as in the notes: live seat i, information set I = (own hand, public record),
field F (the net; deterministic), belief β = uniform on the void-consistent fiber,
sample S of N = 512 worlds, floor f = 16.

**Definition (the Zeb plan).** For root action a, the planner solves

    Q̂_S(a) = max over ρ in Π^a_{S,f}  (1/N) Σ_{w in S} u(z(w; a, ρ, F))

where Π^a_{S,f} is the set of contingent plans of seat i that (i) start with a, (ii) choose
one tile per record (the cell), and (iii) at any record whose cell in S holds fewer than
f worlds, play the net's tile for i's own hand at that record. Zeb plays argmax_a Q̂_S(a),
ties to the lowest tile id, and re-solves from a fresh S at every turn.

**Proposition 1 (the floored plan is a legal policy of seat i).** Every plan in Π^a_{S,f}
is a function of (own hand, record) only. *Proof.* Along the plan tree the cell at a record
is determined by the record: the viewer's own tiles are in the record, and the other seats'
tiles partition S deterministically by the net's move from each world's hand
(`_partition`), so `alive = alive(rec)`. At a floored cell the move is
`field.move(rules, viewer, own_hand_at_rec, rec, bid, side)` — a function of the viewer's
hand and the record (`_handed_down`). At an unfloored cell the move is the argmax of
memoised values indexed by (rec, alive(rec)). Nothing reads a hidden hand. ∎

So the root value is the value of a *legal* policy on the sample, and the floor is a
**restriction of the policy class**, not a change of player: Π^a_{S,16} ⊂ Π^a_{S,1},
hence Q̂_{S,16}(a) ≤ Q̂_{S,1}(a) in-sample for every a (observed: E7's floor-1 node counts
are 9× floor-16's, and E6's exact-fiber values obey the inequality everywhere).
Answer to candidate angle (2): yes, the root value is the value of a legal policy of the
live seat; what the floor changes is *which* policies are allowed, and (E2/E7) it allows
almost only "the net after my first move".

**Proposition 2 (SAA optimism).** Let V(ρ) = E_{w~β}[u(z(w; ρ, F))] be a plan's true value
under the uniform belief and Q(a) = max_{ρ ∈ Π^a_f} V(ρ) (the floor-restricted true optimum;
the floor predicate has to be fixed to a sample-free rule for this, e.g. "the net after the
root", which E7 says is 93%+ of Zeb anyway). Then for every a,

    E_S[ Q̂_S(a) ] = E_S[ max_ρ Ŝ_S(ρ) ] ≥ max_ρ E_S[ Ŝ_S(ρ) ] = Q(a),

and the executed plan ρ*_S satisfies V(ρ*_S) ≤ Q(a) ≤ E_S Q̂_S(a). *Proof.* max of means ≤
mean of maxes (Jensen for the convex max). ∎ This is the optimizer's curse / the SAA bias
of stochastic programming; Mak–Morton–Wood (1999) add that E_S[max_a Q̂_S(a)] is
*non-increasing in N* and converges to max_a Q(a). That is the mathematics behind "512
beat 160" and behind the hub's "the forecast lies": a 1-world cell is an SAA with a policy
class that can see the world. The floor shrinks the class so the bias collapses from +20
margin points (hub, floor 1) to the +1.2 pmake points measured tonight at floor 16 (E1).

**Proposition 3 (Zeb is deterministic; the "mixture" is over seeds).** The sample seed is
`mix(config.seed, rung, viewer, own hand, record key, voids)` — a function of the
information set. For a fixed config seed Zeb is a *deterministic* legal policy; the
randomness candidate angle (1) worries about is the sensitivity of that policy to its seed.
Across 10 seeds the shipped move is unanimous at 18% of leads and 44% of trick-1/2
positions, with ≥3 distinct moves at 50% of leads; whenever the in-sample gap is ≥ 0.05
pmake the move is unanimous (E5). Against a fixed field, the value of the seed-mixture is
the mean of the seeds' values (a mixture over policies independent of the hidden hands),
so the instability costs nothing beyond the per-seed regret measured in E1. Finite-time
"finding the optimal move" is therefore the wrong frame for a sampled player: the right
statement is a *regret* statement (E1) or a *certificate* statement (E8), and only the
regret one is affordable.

## 2. Experiments (command, wall, result)

All run from this directory with `PYTHONPATH=/Users/jason/code/mk5-main`, `python -u`.

| id | script | wall | positions |
|---|---|---|---|
| pour | `python -u -m scenario.atlas pour --deals 40 --spec k0,n512,F=… --seed 424242 --out pour40` | 4 s | 714 planned (406 forced) |
| E1/E2 | `e1e2.py pour5 … 200` and `e1e2.py pour40 … 400 8` | 121 s; 283 s | 88; 85 (first two tricks) |
| E3 | `e3.py pour40 e3_pour40.json` | 7 s | 714 |
| E4 | `e4.py pour40 e4_pour40.json 64 280` | 306 s | 24 × 64 worlds × 2 live players |
| E5 | `e5.py pour40 e5_pour40.json` | 58 s | 714 × 10 seeds |
| E6 | `e6.py pour40 … 2000 240` + `… 20000 280` + `… 8000 270 2000` | 52 s; 308 s; 179 s | 306 + 43 exact fibers |
| E7 | `e7.py pour40 e7_pour40.json` | 41 s | 714 × floors {513, 64, 16, 1} |
| E8 | inline (log in session; json `e8_pour40.json`) | 5 s | 476 sampled-fiber positions |

### 2.1 E1 — the plan out of sample (winner's curse and regret)

Solve with the Python planner on S (512 worlds, seed 0: the pour's own decision), freeze the
plan (Prop. 1), evaluate every root action's plan on 512 fresh uniform worlds (seed 1).
`curse` = q_in(chosen) − q_out(chosen plan); `regret_out` = best out-of-sample plan −
chosen plan, both in the mover's pmake. Units: probability of making the contract.

| pour40, plays | n | curse mean | curse p90 | curse < 0 | rank flips | regret_out mean | p90 | max |
|---|---|---|---|---|---|---|---|---|
| lead | 40 | +0.016 | +0.049 | 30% | 45% | 0.007 | 0.023 | 0.045 |
| 1–4 | 45 | +0.009 | +0.049 | 40% | 31% | 0.003 | 0.011 | 0.021 |
| all 85 | | **+0.012 (se 0.003)** | | | **38%** | **0.005** | | 0.045 |

pour5 (all 88 positions): lead curse +0.040 (n=5), 1–7 +0.005, 8–15 +0.001, 16+ +0.002;
regret_out 0.010 / 0.003 / 0.001 / 0.000; flips 60% / 43% / 14% / 0%.

Reading: the optimism is real and the size Prop. 2 predicts for a nearly two-stage SAA;
the *move* is unstable (38% of early positions rank differently out of sample) but the
*cost* of the instability is half a pmake point. Worst case seen: pos 3 of pour40, lead,
q_in 0.381, q_out 0.301 (curse +0.080), regret 0.045.

### 2.2 E2 — how deep Zeb actually plans (pour40, first two tricks, the plan's own worlds)

Per world along the chosen root action: discretionary own decisions remaining, and how
many the plan's max decided (the rest were handed to the net by the floor).

| plays | own decisions left (mean) | decided by the plan (mean) | worlds with all decided |
|---|---|---|---|
| lead | 2.67 | 0.12 | 14% |
| 1–4 | 2.54 | 0.26 | 12% |

Histogram over 43,520 world-paths: 0 planned decisions after the root **80.8%**, 1: 19.1%,
2: 0.08%, 3+: none. pour5 later phases: 8–15 plays 1.12 left / 0.23 planned; 16+ 0.23 / 0.03.

### 2.3 E7 — the one-step rollout identity (pour40, 714 positions, compiled core, seed 0)

Same sample, four floors. f=513 > N: every interior own decision is the net's.

| plays | n | f=513 same tile as Zeb | f=64 same | f=1 same | q16 loss of the f=513 move, mean | max |
|---|---|---|---|---|---|---|
| lead | 40 | **82%** | 82% | 38% | 0.0007 | 0.008 |
| 1–7 | 188 | **93%** | 97% | 76% | 0.0008 | 0.033 |
| 8–15 | 230 | **96%** | 97% | 88% | 0.0005 | 0.042 |
| 16+ | 256 | **100%** | 100% | 99% | 0.0000 | 0.004 |

Nodes: f=513 7.9M, f=64 9.4M, f=16 11.5M, f=1 107.7M across the 714 decisions.
So shipped Zeb = one-step rollout of the net at 93–100% of positions after the lead, and
the second planned decision (the 18% at the lead) is worth 0.0007 pmake in its own units.

### 2.4 E3 — certificates read off the splat (pour40, 714 positions)

(a) Dominance: the chosen move's make indicator ≥ every alternative's in every sampled
world (a belief-free certificate: no reweighting of the sample can change the move).
(b) Net-consistency f: the fraction of the 512 uniform worlds from which the net, holding
each seat's implied hand, would have played every observed discretionary tile.
(c) The 2δ certificate of note 1 §3 with δ = TV(uniform, uniform-on-consistent) = 1 − f.

| plays | n | chosen dominates all | dominated by another | mean f | 2δ certificate | positions with f = 0 |
|---|---|---|---|---|---|---|
| lead | 40 | 0% | 0% | 1.000 | 98% (trivial: nothing observed) | 0 |
| 1–7 | 188 | 7% | 0% | 0.172 | 1% | — |
| 8–15 | 230 | 34% | 0% | 0.027 | 0% | — |
| 16+ | 256 | 86% | 0% | 0.002 | 0% | — |
| all | 714 | 44% | 0% | | 6% | **418** |

"Dominated by another: 0%" is a theorem, not a finding (a strictly dominating move has a
strictly higher mean, and Zeb takes the max). The true deal (whose other seats were Zeb,
not the net) is net-consistent 34% / 9% / 2% of the time by phase; the net agrees with
Zeb's own move at 58% of planned positions. Where f > 0, restricting to the consistent
worlds changes the move 36% of the time with mean regret 0.025 — on a handful of worlds,
so this is noise, not a belief result; it is what ESS collapse looks like.

### 2.5 E8 — what a certificate would cost (pour40, 476 sampled-fiber positions)

Chosen vs runner-up on the pour's own 512 worlds: gap, fraction of worlds where their
outcomes differ, and the N an empirical-Bernstein paired test (Maurer–Pontil, 95%,
Bonferroni over A−1 rivals) would need to certify the gap.

| plays | n | gap median | disagree median | N needed, median | N needed, p25 | certified at 512 |
|---|---|---|---|---|---|---|
| lead | 40 | 0.011 | 12.1% | **10,198** | 3,165 | **0%** |
| 1–7 | 188 | 0.016 | 9.0% | 3,602 | 654 | 23% |
| 8–15 | 248 | 0.008 | 3.9% | 7,848 | 1,308 | 16% |

Paired Hoeffding needs 2·ln((A−1)/δ)/gap² ≈ 80,000 worlds at the median lead. The compiled
core does 512 worlds at the lead in 44 ms (hub), so 10,000 would be ~1 s on the M5 and
several seconds on a phone — and that certifies only the *median* lead at 95%.

### 2.6 E6 — the floor where the fiber is exact (pour40)

At positions whose fiber has ≤ 17,640 completions the Python planner enumerates every
world; floor 1 is then the exact uniform best response to F = net (T1's instance: finite,
exact, no sampling; a one-world cell here is inference, not clairvoyance). Floor 16 on the
same enumeration restricts the class.

| | |
|---|---|
| positions | 306 (fiber 1 … 17,640; 107 undecided, 0 < V* < 1) |
| floor-16 tile = exact best response | **99%** (3 differ) |
| exact value of the floor's tile vs V*, mean regret | **0.0001** (max 0.024, at pos 507: V* 0.212 vs 0.188) |
| floored plan's own value vs V* | gap 0.0015 mean, 0.056 max (the plan under-reports because the net plays its tail) |
| nodes floor 1 / floor 16 | 1.21× |

E6m (2,000 < fiber ≤ 8,000): 43 positions, fiber 2,000–8,000 (179 s): floor-16 tile = exact best response in 93% (3 differ); mean regret of the floor's tile 0.0001, max 0.003; 31 undecided positions, 3 of them differ. Pooled with §2.6's 306: 349 exact-fiber positions, 6 moves differ (1.7%).

### 2.7 E4 — Zeb vs its own net as the live seat; the forecast vs the realized

24 positions of pour40 (plays 0/2/5/9/13), 64 fresh uniform worlds each, the other three
seats the net in both arms; the live seat the net (base policy) or shipped Zeb replanning
with the compiled core.

| | |
|---|---|
| mover's make rate, Zeb − net, pooled | **+0.013 (se 0.008)**; positions Zeb > net 13, < 6, = 5 |
| paired worlds | Zeb made and the net did not **42**; the reverse **22**; of 1,536 playouts |
| by phase | lead −0.016 (n=4), plays 1–7 +0.031 (n=9), 8–15 +0.009 (n=11) |
| forecast q_in(chosen) vs realized Zeb-vs-net | 0.299 vs **0.326**: forecast − realized = **−0.027** |
| net-vs-net at the same positions | 0.312 |

So the rollout improvement of the net by its own one-step planner is about a point of
pmake per position (consistent with the arena: the net-field Zeb at parity with the
two-rung champion, and the hub's regret-0.26-points imitation). And Δ_execution of note
1 §7 is *negative* here: V_plan < V_execute, because the plan values "net after my move"
while the executed player re-solves at every turn. Note 1's sign claim assumes the
executed policy lies in the plan's class; the floor puts the replanning player *outside*
it. The forecast is conservative, not optimistic, against its own model — the +0.3
margin-point optimism in the hub's calibration tables is Zeb-vs-Zeb, a different field.

## 3. What this says about the brief's claims

**T1.** Instantiated and checked at 306 real positions (E6): exact Walt over the whole
fiber with floor 1 is the uniform-β best response to F = net; the computation is finite:
52 s for the 300 positions with fiber ≤ 2,000, 308 s for 6 more up to 17,640, 179 s for
the 43 between 2,000 and 8,000 — all in pure Python. Proof is theory's; the data is here.

**C1 (uniform is not β^F).** Confirmed, and sharpened: for the *deployed* F the posterior
β^F has empty support on the sample in 58% of positions and is supported on ≤ 3% of the
sample after trick 2 (E3). The 2δ certificate of note 1 then certifies 6% of positions, all
but 1% of them opening leads where nothing has been observed. The brief's C1 experiment
("plan over consistent worlds only") is not an experiment one can run with a deterministic
net: it mostly has no worlds. What it needs is in §5.

**C2 (sampling, floor, tie rule).** Sampling: gap and certificate cost measured (E1, E8); the
move is uncertifiable at 512 worlds at the lead (0%) and the uncertified move's regret is
0.5 pmake points (E1). Floor: measured exactly at 306 positions (E6): 1% of moves change,
0.0001 mean regret; and at sampled fibers the floor is what makes the plan a legal one-step
rollout instead of a clairvoyant plan (E7, hub's +20 margin points). Tie rule: ties are
common under make (E8: the median lead has 88% of worlds where the top two moves agree),
and the lowest-tile-id rule is a deterministic selection among SAA-tied plans; it is not a
source of regret by itself (the tied plans have equal in-sample value) but it is why Zeb's
move is a *hash-like* function of the seed at near-ties (E5).

**C3 (O1 fixed points are not O2/O3).** Not attacked tonight beyond the observation that
the expert-iteration fixed point is a uniform-β stabilized profile (§4), which note 1 §4
already shows can leave an arbitrarily large partnership gap. Barrier stated in §5.

## 4. The fixed point (candidate angle 3)

Write R_U for Walt's one-step response operator under the uniform belief with floor
(E7: ≈ one-step rollout). Expert iteration is

    net_{t+1} = imitate( R_U[net_t] )       (the pour of the player whose field is net_t)

which is approximate policy iteration (API) on a self-play game where every seat is
modeled by the same net. At its fixed point net* ≈ R_U[net*] in value (E4: the live-seat
improvement is +0.013, and the arena interval includes zero against the two-rung
champion). Three consequences, each a restatement of the notes in Zeb's terms:

- The parity of "net at the bottom, one rung above" with "two rungs above dice" is the
  ladder of note 2 §2 having stabilized — not by nesting 23 rungs but by letting imitation
  carry the fixed point. "Five times the data bought a better model and the same player"
  is what P^k = P^⋆ for k ≥ h(c) looks like from the outside.
- The fixed point is note 1 §1's object: a profile in which every seat's move begins an
  optimal *one-step* continuation under a belief (uniform) that is **not** the belief its
  own policy induces. Note 1's family shows stabilization alone bounds nothing. Tonight's
  numbers say the loss is small *against this field*: Zeb's out-of-sample regret under its
  own belief is 0.5 pmake points (E1), and the tried belief repairs bought nothing in marks
  (hub). Whether the field-relative loss is small is exactly what no measurement so far
  can see, because the field is Zeb itself (and in Plunge, a family).
- API with function approximation has fixed points that are not optimal policies in
  general (standard; Bertsekas & Tsitsiklis 1996 for the perfect-information case). Here
  the operator itself is also not the Bayesian one. So two separate reasons the lever
  stops: the operator's belief, and the operator's one-step class. Neither is data.

**Conjecture (marked).** The uniform belief wins its nulls because, against a field that is
itself a one-step uniform-belief rollout, the discretionary plays carry little information
about *holdings* beyond the voids: the net's move is a function of the whole record and
58% of the time differs from the planner's, so a 0/1 likelihood is nearly orthogonal to
the hand. The test is the mutual information between the net's tile and the mover's hand
given the record, over the pour; not run tonight.

## 5. The ledger: what we would need, and the price

| target | what deployed Zeb lacks | the change | affordable? |
|---|---|---|---|
| O1, uniform β, exact | sampling (E1 regret 0.005, E8 0% certified at the lead); floor (E6: 0.0001) | N ≈ 10⁴ worlds *and* a paired empirical-Bernstein test on the splat (the certificate is computable from the per-world outcomes Zeb already keeps) | 1 s on the M5 per lead; not on a phone at 2–3 s. The regret without it is half a point. |
| O1 against the actual field F | β^F: with deterministic F the posterior has (near-)empty support (E3) | (1) soften F: a softmax or ε-mix over the net's integer scores so F_σ(a\|I) > 0; (2) sample the posterior by importance-resampling uniform draws with weight ∏_t F_σ(a_t\|I_t(w)) — at f ≈ 3% that is ~17,000 uniform draws and ~170,000 net forward passes per decision; (3) **model the future with the same F_σ**: per world, draw the modeled seats' tiles from F_σ on a record-keyed tape (the dice-tape construction, with the net's distribution instead of uniform), so worlds still partition and the plan stays one cell per record | (1)–(2) are cheap in the core (µs per forward pass); (3) is a one-line change to `_field_move`. Coherence requires all three; the hub's ε=0.1 tilt was (1)+(2) by reweighting without (3) and without resampling, and died of ESS. Marks value: unknown; the dice/auction nulls say expect little against Zeb-like fields. |
| O1 with the executed policy in the plan's class | the plan values "net after my move" while the player replans (E4: Δ_execution = −0.027) | nothing to fix for strength; to make the forecast match the player, replace the floor's hand-down with "the rung-0 player's move at that record from a fresh sample" — which is the two-rung champion's cost | not affordable at 2–3 s; the honest cheap forecast is the conservative one. |
| O2 (partnership) | one-seat operator (§4) | joint plan over (bidder, partner) policies with each seeing only its own hand: a Dec-POMDP step; the cell structure already supports "sum over partner's observations" but the max must range over *pairs* of plans | exponential in the partner's information sets; a small-case solver only. Not for Plunge. |
| O3 (team maxmin) | fixed field | double oracle with certified responses (note 1 §6); Zeb as the response oracle gives lower bounds only | research instrument, not a player. |

The affordable item that changes what Zeb *is*: the stochastic field with a tape (row 2,
item 3). It costs nothing at the table, makes β^F definable, and is the prerequisite for
any belief ever being measurable on the seam again. Whether to pay for the posterior
sampling on top of it is then an experiment with a non-degenerate answer.

## 6. What "optimal" should mean at the family table (candidate angle 4)

Plunge's field is three humans and one Walt partner; the pour's field is Zeb; the net is a
58%-faithful model of Zeb. O1 against the net is therefore a proxy twice removed from the
deployed objective, and O1 against a deterministic proxy is degenerate (E3). The target that
matches the purpose is **low regret under a belief that is robust to not knowing F**, and
uniform-over-the-fiber is the maximum-entropy such belief; its measured out-of-sample
regret under its own objective is half a pmake point (E1) and the rollout improvement over
the field it imitates is one point (E4). Both are below what a table notices. "Finds the
optimal move" is not the property Plunge needs; "plays within a point of its own best
response, in 44 ms, honestly forecast" is, and tonight's numbers say Zeb has it. The one
thing a human would notice is the seed-dependence at near-ties (E5: three or more
different leads from the same hand across seeds) — which is also what keeps Zeb from
being predictable, and may be a feature.

## 7. What I could not do

- No arena run (marks) tonight; everything is pmake at positions. The +0.013 of E4 is 24
  positions × 64 worlds and its interval touches zero.
- V_execute was measured against the net, not against Zeb-others or humans; Δ_execution's
  sign is established for the model the plan uses, which is the quantity note 1 defines.
- E6 covers fibers ≤ 17,640 (306 positions, plus E6m); exact best responses at early
  positions are out of reach (fiber 4·10⁸ at the lead).
- No stochastic-field prototype; the ledger prices it but does not run it.
- The mutual-information conjecture of §4 is unmeasured.
- Nothing about O2/O3 beyond restating the barrier.

## Files

`lab.py` (tools: Python solve, `FrozenPlan`, `consistent`, `playout`), `e1e2.py`, `e3.py`,
`e4.py`, `e5.py`, `e6.py`, `e7.py`; results `*_pour*.json` and `*.log`; pours `pour5/`,
`pour40/`.

FABLE2 DONE
