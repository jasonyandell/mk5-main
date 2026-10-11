---
title: The scenario player — Walt, rebuilt here from the mathematics
kind: entity
first_seen: 2026-10-09
last_updated: 2026-10-11
status: active
---

## What it is

`scenario/` is a from-scratch implementation of the player Jason's family
plays against every day in Plunge (`~/code/plunge`, the shipped mobile
game) and researches in `~/code/texas-42` — written in this repo from
*Walt: a mathematical account* (2026-10-09, `walt/math/` in texas-42) so
that this project's belief machinery can be run against it. The name
collides with the earlier [[walt]] on this wiki (the exact endgame
information-set solver of 2026-07-17); Jason ran out of names. This page
is the scenario player; the arena spec is `walt`.

The construction, in one line:

    Walt = R_live[ R_model[ U ] ]

A player holds a weighted set of scenarios (possible hidden situations),
chooses ONE action across every scenario it cannot tell apart, and lets
its plan branch only where an observation separates them:

    V(C) = max_a Σ_o V(C_{a,o})   at its own turn
         = Σ_o V(C_o)             at another seat's turn

The other seats are the same construction one rung down; random legal
play is the bottom rung, drawn from a tape keyed on the public record so
one scenario plays one tile at one record in every branch (worlds
partition, never multiply). A modeled seat's decision is a pure function
of (rung, seat, remaining hand, record) and is cached as such.

## Why it is not the wall

In the terms of [[belief-policy-value-algebra]]: [[expected-q-value|eq]]
is π deleted twice. The scenario player un-deletes both — the
continuation is one contingent plan over the cell (term 2 computed
natively, no [[strategy-fusion]] by construction), and the field is a
declared model with its own information. Value lives on information
states under a field. That is the object [[the-wall]] said was missing;
see the 2026-10-09 chapter of [[the-wall-biography]].

## The parts

| module | what | gated by |
|---|---|---|
| `scenario/rules.py` | per-declaration tables tabulated from `forge.oracle.tables` | parity with the zeb engine's legality and trick resolution |
| `scenario/record.py` | the public record a cell shares (played, trick, leader, banked, void-derived forbids); `advance` | replay of every engine history matches played/leader/trick/banked; voids never contradict the real hand |
| `scenario/fiber.py` | exact-uniform sampling of hidden-hand completions by a counting DP — no rejection, no invalid worlds | counts equal brute-force enumeration; empirical uniformity |
| `scenario/planner.py` | `Rung`, `Config`, `Walt`: the finite planning rule, the ladder, the dice tape, the `make`/`points` utilities | single-scenario value equals an independent best-plan search; the information inequality V(μ₁+μ₂) ≤ V(μ₁)+V(μ₂) on every split; scale invariance; determinism |
| `scenario/core/` + `scenario/compiled.py` | the compiled core (Rust, PyO3): the planner and the fiber sampler line for line; `decide_batch` plans an arena round in parallel on every core with the GIL released | **bit-identical** to the Python planner — same tile, same action values, same node count — on random states across rung-0, two-rung and three-rung configs, `make` and `points`, uniform and tilted beliefs, batched and single |
| `scenario/field.py` + `core/src/field.rs` + `scenario/field_train.py` | the field net: a learned integer policy as rung −1 (see "The net in the field") | forward pass bit-identical referee ↔ core; `decide`/`analyze` with the net field bit-identical |
| `arena/walt_play.py` | the [[arena]] `PlayPolicy`; spec `walt[:k<level>][,n<outer>][,m<inner>][,h<horizon>][,g<inner horizon>][,u=make\|points][,py]` (`py` = the pure Python referee) | an arena half completes |

Tests: `python -m pytest scenario/tests -q` (22, CPU, no checkpoint). Build the
core once with `python -m pip install ./scenario/core` (Rust toolchain).

The Python planner is the referee and the compiled core is the runtime. To
make that checkable, every random draw is defined: scenario sampling and the
dice tape run on SplitMix64 streams seeded from the invocation (config seed,
rung, viewer, hand, packed record, voids), and the record is keyed as one
integer, not a CPython hash. The core mirrors CPython 3.12's compensated
`sum()` so even tilted-belief values agree to the last bit. The belief
weights the live rung only; modeled minds are compatibility-only (g ≡ 0),
as the shipped player does.

## The levers (Θ = (G, I, u, β, S, k, H, L, ≺, Γ))

Every choice the mathematics names has a seat in the code: `β` is a
callable `(record, viewer, worlds) → weights` on `Walt(belief=…)`
(uniform by default — the g ≡ 0 limit); `k` is the number of rungs; `S`
is `n_worlds` per rung; `H` is `horizon` per rung (plays searched
explicitly; `None` = to the end); `L` is the tape playout at the
frontier; `u` is `make` or `points`; `≺` is lowest tile id; `Γ` is the
record-keyed tape. The belief seam is the point: it is where the
[[auction-decoder]], the [[gus]] belief head, and the tilt form
b = u·e^g plug in — the first time belief-weighting has had an honest
continuation to cash through.

## Measured (2026-10-09, M5 Max, pure Python)

Seconds per decision, three random states each, `make` utility, bid 30:

| plays so far | k0 n16 full | k0 n40 full | k1 n16 m4 g4 (default) |
|---|---|---|---|
| 0 | 0.21 | 0.52 | 6.2 |
| 4 | 0.04 | 0.07 | 1.3 |
| 8 | 0.01 | 0.03 | 0.4 |
| 12 | 0.00 | 0.01 | 0.1 |

Smoke: `heuristic+walt:k0,n24` vs `heuristic+random`, 8 games to 3 marks,
seed 777: 8/8, +2.25 marks/game [+1.75, +2.62], made 80% vs 20%.

First meeting with `lens:ev` — E[Q] n=10, undefeated at play against every
learned challenger — same heuristic bidders both sides, 32 games to 7 marks,
seed 7000000: rung 0 at 40 worlds (random-play field, uniform belief) wins
13/32, **−0.84 marks/game [−2.00, +0.34]**, point margin −5.8/hand, made
46.6% vs 54.3%, 293 s wall. Same protocol at Jason's shipped outer sample, rung 0 at **160 worlds**:
wins 18/32 (56.2%, both halves), **+0.34 marks/game [−0.91, +1.59]**, point
margin −3.2/hand, made 53.2% vs 50.0%, 766 s wall. Underpowered at n=32,
but a dice-field, belief-free scenario player in pure Python is at or
above the champion that beat every learned challenger — and every belief
lever above is untouched in this number.

For reference, the texas-42 repo's own `walt_bridge` binary (level-1,
bid-blind P(make 30), speaks rob's line protocol so it seats as
`rob:<path>`) on the identical block: 16/32, **−0.28 [−1.53, +0.94]**,
made 52.0% vs 54.7%. Same dead heat with E[Q] at this size.

The shipped two-response form is 160 outer / 8 inner worlds. In pure
Python its trick-1 decision cost 29 s after the 2026-10-09 speedups. The
compiled core (2026-10-10), one thread, M5 Max, seconds per decision:

| plays so far | k0 n160 full | k1 n160 m8 g4 | k1 n160 m8 gfull | k1 n640 m8 gfull |
|---|---|---|---|---|
| 0 | 0.056 | 1.12 | 10.5 | 41.9 |
| 4 | 0.011 | 0.30 | 2.0 | 7.9 |
| 8 | 0.004 | 0.10 | 0.36 | 1.5 |

Rung 0 is now 50× the Python; the two-response form with a 4-play inner
horizon 26×. The inner rung's horizon is the cost lever: a full-depth
inner mind (the bridge's shape) is ~10× a 4-play one at trick 1.

**Why the texas-42 bridge is slow** (`walt/walt/src/bin/walt_bridge.rs`,
read 2026-10-10): its values are exact rationals (`BigRational`), its
saturation-tie refinement resamples the tied options at 4× and 16× the
outer count (up to 2,560 worlds at 160), and the arena drives eight
single-threaded bridge processes in lockstep. Jason's 384-game receipt at
50/8 took 117 min; at 160/8 here one lockstep hand-round of 384 hands took
~30 min, i.e. ~6 h per block. Both 2026-10-09 384-game blocks (bridge
160/8 and Python rung 0 at 160) were killed at Jason's call after one and
three rounds respectively — too slow to be worth it; early reads were
bridge 208–176 marks after one round, Python rung 0 574–578 after three.
The powered replication ran on the compiled core instead (2026-10-10):

**Walt beats E[Q] here too.** Jason's dropped-30 protocol exactly (`bid30`
both sides, best-pip-trump declaration, 1 mark/hand, games to 7, mirrored
pairs, 384 games, seed 1) against `lens:ev` (E[Q] n=10, undefeated at play
against every learned challenger on this wiki):

| player | games won | marks/game (95% CI) | points/hand | made (bid 30) | wall |
|---|---|---|---|---|---|
| `walt:k0,n160` (rung 0: dice field, no partner model) | 203/384 (52.9%) | +0.18 [−0.12, +0.49] | −4.97 | 36.0% vs 34.4% | 124 s |
| `walt:k1,n160,m8` (two rungs, inner horizon 4 plays) | **233/384 (60.7%)** | **+0.74 [+0.42, +1.05]** | −4.51 | 38.8% vs 32.3% | 17.5 min |

Both halves positive for the two-rung form (65.1% / 56.3%). The signature is
the one in Jason's 2026-08-17 receipt (+0.38/game per seed with the texas-42
bridge at 50/8): marks up, points DOWN — the `make` utility cashes contracts,
not pips. Rung 0 alone is at or just above E[Q]; the second rung — the
modeled partner (and opponents) — is where the edge lives: "one more level
and its partner is participating." Uniform belief throughout; every belief
lever on this page is still untouched in these numbers.

## The belief program — night 1 (2026-10-10/11)

The first beliefs on the seam, each a likelihood factor on every live scenario
(the tilt form b = u·e^g), all bit-identical referee ↔ core:

- **dice** — the hidden seats' discretionary plays under random legal play: a
  play with n options weighs 1/n (forced-looking plays count more).
- **field** — the same plays under the rung-0 player: ε/n + (1−ε)·[it would
  have played that tile].
- **auction** — the declaration: ε/7 + (1−ε)·[the bidder's dealt hand in this
  scenario would have called this trump]; exact for the `bid30`/heuristic
  bidders, which declare `best_trump` deterministically.

Also landed: **exact endgame enumeration** — when the fiber is no larger than
the rung's sample (90 completions with two tricks left, 1,680 with three) it is
enumerated outright: the exact belief, no duplicate scenarios, faster.

**Measuring stick: uniform two-rung Walt, not E[Q]** (Jason: skip the oracle).
Same spec both sides (`k1,n160,m8`), mirrored deals, so deal luck cancels and
the margin is the belief alone. Dropped-30 protocol, 384 games, seed 1:

| belief vs uniform | marks/game (95% CI) | contracts made, belief vs uniform |
|---|---|---|
| auction | −0.05 [−0.37, +0.27] | 35.2% vs 35.7% |
| dice | −0.06 [−0.37, +0.26] | 35.8% vs 36.3% |
| auction+dice | −0.19 [−0.51, +0.13] | 34.4% vs 36.0% |
| field ε=0.5 | **−0.65 [−0.96, −0.34]** | 33.3% vs 39.0% |
| field ε=0.1 | **−0.96 [−1.26, −0.66]** | 32.0% vs 40.5% |

(48-game previews, all five run first, pointed the same way: auction +0.19,
dice +0.19, auction+dice −0.40, field −0.85, auction+field −1.02.)

**Two findings.** (1) Against this field, exact knowledge of the auction and
Occam inference over the observed plays buy *nothing measurable* at the table
— uniform over the void-consistent fiber is already a very good belief for 42,
as Jason kept finding against his own instinct. (2) The field tilt *hurts*,
and the mechanism is measured: **effective sample size**. The fraction of the
160 scenarios the weights effectively keep, by plays so far:

| belief | 4 plays | 8 | 12 | 16 |
|---|---|---|---|---|
| dice | 0.83 | 0.86 | 0.89 | 0.83 |
| auction | 0.54 | 0.84 | 0.85 | 1.00 |
| field ε=0.5 | 0.26 | 0.14 | 0.12 | 0.08 |
| field ε=0.1 | 0.08 | 0.02 | 0.02 | 0.03 |

A field-tilted player at ε=0.1 is planning over three to thirteen effective
worlds by mid-hand. The lesson is general and belongs to the seam itself:
**weighting a uniform sample is only sound when g is mild. A strong g must be
sampled from** — draw from b = u·e^g (importance-resample a larger uniform
draw, or sample the fiber under the tilt), not reweight 160 uniform draws. That
is the next build on the seam, and it is what makes any sharp belief (a learned
head, a convention model) usable at all.

## The atlas (2026-10-10)

Beliefs paused after night 1 (the taste said *uniform again*; a dozen other
low-hanging things come first). The new object is **the atlas**: one fully known
deal, the perfect-information game tree the oracle would have walked, and on
each of its positions Walt's honest evaluation — the distribution of outcomes
over the scenarios a seat that *cannot* see the other hands would entertain.
The oracle's mold, poured without peeking.

The idea arrived as "reverse DP fuzzed out from the end" and resolved in
conversation: a position's label never reads its children (they are values of
one frozen world, not of the viewer's scenarios), so **there is no induction
and no direction** — the backward walk only enumerated the positions. What
Jason wanted was *position → outcome distribution* on the oracle's position
set; what Walt already was is a backward induction over a *bundle* of
scenarios sharing one record (the cell), which is the only unit that can host
imperfect information. See [[strategy-fusion]] for why per-world backup is the
oracle's sin, and [[the-wall]].

**Two instruments landed (commit `332ccfe8`, referee and core bit-identical,
100 tests):**

- `Walt.analyze(state, bid) → Analysis(tile, q, worlds, weights, outcomes)` —
  `decide` opened up: per legal root action, the leaf value every live scenario
  reaches under the plan (a replay of the solved tree through the memo, zero
  measurable cost). Σ w·outcome / Σ w = q[a]. The **splat**: with the points
  utility it is the outcome distribution, and every make-probability is one of
  its tails (pmake(b) = mass at or above b) — the bid curve for free.
- `CompiledWalt.mold(hands, decl, leader) → positions per layer` — the shape of
  one deal's tree with transposition on the full record (128-bit keys, two
  layers resident, hash-sharded, all cores). `scenario/atlas.py` is the referee.

**The mold, seed-1 deal (decl sixes, bidder leads).** 374,004,900 positions in
1.9 s wall, 5.7 GB peak. It bows in the middle and tapers hard at both ends,
as the oracle era remembered:

| layer | 8 | 12 | 16 | 18 | 20 | 23 (peak) | 24 | 27 |
|---|---|---|---|---|---|---|---|---|
| positions | 5,081 | 167,405 | 2.66M | 14.3M | 18.3M | 87.4M | 30.4M | 27.3M |

**The pour price** (scratch `atlas1.py`, 192 sampled positions per layer, 18
cores): rung 0 labels from 3 positions/s at the lead to 500k/s at two tricks
left (the fiber is exact there); `k1,n160,m8` from 0.04/s to 490k/s. A whole
deal at rung 0 is under an hour; with the two-rung player, overnight. The
distilled net ([texas-42 PR #102](https://github.com/jasonyandell/texas-42/pull/102))
is the way to pour k1 quality at full-deal scale.

**The slice** (Walt at all four seats, every legal move labeled, 28 calls):
the first look at the object. Opening lead valued +30 margin; the hand went
down 19–23; the bidder's partner leading trick 3 saw every option negative;
one hinge (the 6-5 at trick 5) was worth 17 points over its alternatives to a
viewer who could not see the hands.

**The splat, seen (2026-10-10, `scratch/atlas/splat1.py`, seed 1).** Walt in
all four chairs; seat 0 bid 30 on sixes with 6-5 6-2 6-1 2-2 1-0 4-0 4-1; the
hand went down 19–23. At trick 5 the bidder held 4-1, 6-1, 6-5, with 8 banked
to 1. Sixty scenarios remained — the whole fiber, so this is exact:

```
play 41: mean  -2.3, pmake  0%   (53% "short", never makes)
play 61: mean  -2.9, pmake 37%   (37% set hard, 37% made: two-humped)
play 65: mean +14.7, pmake 43%   (chosen)
```

Look at the first two. Nearly the same mean. Completely different hands. The
4-1 is safe and hopeless, the 6-1 is a coin flip between a bad set and making
it. A scalar player sees them as equal. The splat sees a bidder who is already
behind choosing between resignation and a gamble. That is the whole argument
for the distribution over the number, in one position.

Jason: "seeing that is like seeing purest treasure. I've searched for an
object with this property a long time. blind to these options."

**The forecast lies, and we know why (2026-10-10, `scratch/atlas/marginal1.py`,
`marginal2.py`, `horizon1.py`, `cells1.py`).** Hold seat 0's hand, vary the
other 21 tiles over 64 deals, play each with Walt in every chair, merge the
endings, and compare with what the root splat forecast for the same lead:

| other seats | mean margin | made 30 |
|---|---|---|
| root forecast, 160 scenarios | +30.1 | 83% |
| root forecast, 1,024 scenarios | +27.5 | 77% |
| real endings, others = the modeled rung-0 player (n=8) | +12.8 | 39% |
| real endings, others = rung 0 at n=160 | +11.2 | 44% |
| real endings, others = full two-rung Walt | +9.6 | 45% |

Twenty points of optimism, the same against every opponent including the one
the forecast imagines — so it is inside the plan. **Mechanism, measured:** the
cells thin. Along the chosen plan, at the bidder's second decision the 160
root scenarios sit in cells averaging 3.1 worlds (27% already alone); from the
bidder's third decision on, **every world is alone in its cell**. A max in a
one-world cell is perfect-information play for that world: the plan is
fusion-free at the root and clairvoyant from trick 2. The horizon sweep agrees
(explicit search H plays then dice: H=16 +29.1, H=12 +24.3, H=8 +15.6, H=4
+2.6 — two lies crossing near the truth, not a fix). Live play is honest only
because it resamples every turn. The old E[Q] ([[expected-q-value]]) was this
with no cell at all: ten singleton worlds, each judged by the clairvoyant
oracle, averaged; it already had the per-world "PDF", the splat's grandparent.
**The fix (Jason: "big yes on resampling; ban singleton worlds"):** a cell-size
floor per rung — below m live worlds the viewer stops optimizing over the
survivors and plays as the rung below would at that record, with a fresh
sample (the modeled-move construction already used for the other seats).
Landed as `Rung(n, horizon, floor=m)`, arena `f=<m>` (commit `1406fdae`, 104
tests; floor 1 is the old player to the bit; the root is never floored).
**The arena default is the honest plan from 2026-10-10:** `walt:k1,…` means
`f=16`; `f=1` restores the clairvoyant tail (Jason: "perf with noisy to
similar outcomes are a win… the aesthetics are a factor too" — the math said
one action per cell, and a one-world cell was breaking that promise from
trick 2 on).

**The floor calibrates the forecast (2026-10-10, `floor_forecast.py`,
`marginal3.py`).** Same hand, same lead question, root forecast by floor, and
real endings when everyone plays the floored player over the same 64 deals:

| floor | root forecast | real endings (64 deals, everyone floored) | nodes | time |
|---|---|---|---|---|
| 1 | +30.1 / 83% | +9.6 / 45% | 297M | 26 s |
| 2 | +16.2 / 47% | — | 141M | 12 s |
| 4 | +12.8 / 41% | +9.4 / 38% | 127M | 11 s |
| 8 and above | +10.4 / 35% | +11.5 / 34% (at 16) | 119M | 10.5 s |

Floors 8, 16, 32, 161 are identical: the cells are already thinner than 8 at
the second own decision, so floor 8 hands every interior decision down. The
floored forecast matches the real endings within a point or two of margin
and a point or two of make share — **the splat is now a calibrated forecast**,
at less than half the nodes and time. The honest plan also changes the lead
(the 6-1 trump at +10.4 over the 2-2 at +10.0, where the clairvoyant plan
had the 2-2 four points clear). Whether honesty also *plays* better: **a null,
leaning positive** — `f=16` vs floor 1, mirrored deals, 384 games, seed 1:
**+0.13 marks/game [−0.20, +0.45]**, 195/384 (50.8%), halves 53.6% / 47.9%,
contracts made 36.8% vs 35.6%. Same play within noise, honest forecast, under
half the nodes per decision. The freed compute, spent at the root where the
cell is joint, **buys marks**: `k1,n512,m8,f=16` vs the old champion
`k1,n160,m8` (floor 1), mirrored, 384 games, seed 1: **+0.56 marks/game
[+0.26, +0.86]**, 217/384 (56.5%), halves 60.4% / 52.6%, contracts made
38.2% vs 33.1%, 26.6 min wall vs 19.4 for the floor-16/160 block. The honest
plan at three times the root worlds beats the clairvoyant plan at power, in
about 1.4× the wall. At the 1,024-world cap (`k1,n1024,m8,f=16` vs the same
old champion): **+0.49 [+0.18, +0.80]**, made 37.9% vs 33.6%, 38 min wall —
no further gain over 512 for twice the worlds; **512 is the working spec**.
Attribution (`k1,n512,m8,f=16` vs `k1,n160,m8,f=16`, both honest):
**+0.30 [−0.00, +0.61]**, made 37.3% vs 34.6%, 12.8 min wall. So the root
worlds carry about +0.3 and the floor about +0.1 (its own block was +0.13);
neither alone reaches power, together they are the +0.56. And the
both-honest block ran in half the wall of the mixed ones: the honest player
is the cheap one.

**What must be true (`whatmustbetrue.py`).** The outcomes are per world and
paired, so a move's success set has a description in holdings, read off
without re-solving. The trick-5 hinge at floor 1: the 6-1 makes in 22 of 60
worlds — in 22 of the 24 where the *partner* holds the 6-6, and in none of
the 36 where an opponent does; the 6-5 makes in 26 of 60 across several
holdings and beats the 6-1 world-for-world 48 to 10. Re-weighting under
"partner holds the 6-6" (24 worlds): 4-1 +8.0, 6-1 +14.5, 6-5 +28.8; under
the complement: −9.2, −14.6, +5.3. A testable belief per move, for free; the
root of adaptive sampling among near-ties (paired sign tests, sampling into
the disagreement set — best-arm identification / racing) and of
score-aware utilities read off the splat.

**The pour (commit `3ecf46db`, 117 tests).** `scenario/atlas.py` now holds the
atlas's third instrument: `play_waves` (Walt self-play in lockstep waves, every
planned decision of a wave in one batch on all cores — the "tape": the next
frontier is known before it is asked for) and `pour` (keep the whole splat
for every legal move at every planned position: actions, q, per-world
outcomes as int8, worlds, weights, plus each deal's real ending), saved as a
compressed `.npz` + manifest, round-trippable. Readers: `advantages` (q minus
the group mean, in the mover's units — the GRPO-shaped signal, group = legal
moves), `paired_advantages` (per world), `success_set` and
`holding_predictors` (what-must-be-true), `calibration` (forecast vs real).
CLI `python -u -m scenario.atlas pour --deals N --spec k1,n512,m8,u=points
--seed S --out DIR`. Smoke: 8 deals at `k1,n160,m8` in 3 s, 0.2 MB. **First real pour (seed 1, 200 deals, `k1,n512,m8,u=points`): 3,494 planned
positions (2,106 forced) in 75 s wall, 12.9 MB.** Calibration on the bidder's
opening lead over the 200 deals: forecast +7.0 margin / 37% make vs real
+8.2 / 38%. The honest splat is a calibrated forecast at scale, and the whole
dataset costs a minute. This is
the **advantage dataset** — the dense signal texas-42 PR #94's constructor
never had (one make/set bit per deal there; hundreds of paired outcomes per
move per position here).

**Published (2026-10-10).** The pour is a public dataset,
[jasonyandell/texas-42-walt-pour](https://huggingface.co/datasets/jasonyandell/texas-42-walt-pour),
**CC0 1.0**. Split `train` is a 5,000-deal pour (seeds 1001..6000, 87,403
positions, 31 min wall, opening forecast +6.1 / 36% vs real +5.6 / 36%);
split `validation` is the 200-deal seed-1 pour above; same spec
`k1,n512,m8,u=points`, code at `d94c3050`. Three Parquet configs read off the
pour's own offsets — `positions` (one row per position: record, legal moves,
`q`, `advantage`; 4 MB), `splats` (per-world outcomes, worlds, weights; 406 MB),
`deals` — plus the raw `pour.npz` per split for an exact `atlas.load`. One row
per position, not per action: the group of legal moves stays together, as a
group-relative trainer reads it. Repeatable from the repo (`scenario/hf.py`;
the repo is exactly its arguments, and the pour CLI now records its commit):

```
python -u -m scenario.hf publish --repo jasonyandell/texas-42-walt-pour \
  --split train=scratch/atlas/pour_s1001_n512_d5000 --split validation=scratch/atlas/pour_s1_n512 \
  --commit d94c3050
```

**Leaves keep the margin (2026-10-10, issue #107).** Jason's framing: *the
thing to maximize is pmake given options* — and the splat should let any
utility be read after the fact. Before, a pour under `make` stored leaves as
0/1 and threw the margin away, while a pour under `points` kept the margin
but had planned for points, not pmake. Now the trace's leaves are the
bidder-team final margin under every utility, in referee and core alike; the
utility shapes only the plan, and `q` stays in the plan's units
(`atlas.in_plan_units`: identity under `points`, `[margin ≥ 2·bid − 42]`
under `make`). Where the plan stops before the hand ends — the contract
decided under `make`, or the horizon — the tape plays the rest out, so beyond
that point the margin is the tape's, not the plan's; pmake at the planned bid
is exact, pmake at other bids is read off a partly tape-played tail. One pour
under the arena's own utility (`make`) therefore yields the bid curve, the
points splat and dominance together. The dataset is being re-poured under
`make` (same seeds) for the net-in-the-field run.

**Pruning (the plan, not yet built).** No induction means no table to fill:
a label is a function called when wanted, and the tail layers — most of the
374M — never need to exist. Expand the *plausible cone* (children of moves
within δ points of the best; δ=0 is the Walt-vs-Walt path), *hinges first*
(alternatives far apart), the *cheat map* (expand where the oracle's
perfect-information value and the honest label disagree — the one use of the
oracle: a map of where it was wrong), forced moves free, equivalent tiles
branched once, the exact tail at no cost. Work proceeds in **waves**: label a
frontier in one `decide_batch`, keep survivors, the next frontier is known
before it is asked for (the tape). Mac for the Walt labeler, GPU for the net.

## The net in the field (2026-10-10)

A learned policy as the modeled seats, **and nowhere else** (Jason's binding
rule: the net never enters a value or a leaf; the live seat still solves).
One knob: **rung −1 is the net instead of the dice.** `walt:k0,n512,F=<npz>`
is a live seat planning against modeled seats that play the net, with rung 0's
floored moves handed to it too (a floor at rung 0 is now legal iff there is a
net under it). The leaf playout stays dice. `k1,…,F=` also works (net under a
rung-0 Walt) but costs more than the champion, not less.

| choice | what |
|---|---|
| features (330 bits) | the modeled seat's own hand, tiles gone, open trick by who played it (left/partner/right), voids shown by each other seat, count-tile slots, side, trick position, who is winning the trick, thermometers of points the bidder still needs and the defenders still need — exactly the modeled-move cache key, nothing hidden |
| canon | tiles relabeled so trump is pip 6, other pips in order: one trump suit to learn, not seven. Pip-trump declarations only (the `bid30`/heuristic bidders call nothing else); others refuse |
| target | cross-entropy to softmax(sign·m/τ) over the legal moves, m = each move's mean margin read off the splat (points under every utility, #107), τ = 2 points: near-ties flat, hinges sharp |
| net | MLP 330→128→64→28, 52k params, numpy + Adam (no torch), every tenth deal held out |
| parity | integer inference: bit features, int16 weights at power-of-two scales, `max(acc >> s, 0)`, rounding folded into biases, ties to the lowest tile id. Referee and core agree by construction; tested on 300 positions' raw scores and on `decide`/`analyze` across four net-field configs |

**Smoke (the 200-deal pour, `pour_s1_n512`).** Training 30 epochs in under a
second: held-out regret 1.11 points/decision vs 1.94 for a random legal pick
(the int net picks what the float net picks on 100% of held-out rows).
`bid30+walt:k0,n512,F=pre200` vs `bid30+walt:k1,n512,m8`, 64 mirrored games,
seed 1: −0.70 [−1.44, +0.05], made 28.8% vs 35.1%, 225 s. The plumbing works;
the number is a smoke test, not a verdict.

**Ran before the scope change (points-utility 5,000-deal pour; preliminary,
to be redone on the #107 `make` re-pour):** trained in 9 s, held-out regret
0.60 points (random 2.31; hinges 0.99 vs 9.86); as a model of the points
teacher it beats the champion's own field (rung 0 n8 g4 `make`: 1.54 points,
1.9 pp of make mass vs the net's 1.2 pp). Arena `k0,n512,F=` vs the champion,
384 games, seed 1: **−0.23 [−0.54, +0.08]**, made 35.9% vs 38.0%, point margin
+0.03/hand — even on points, behind on marks, which fits a field (and floor
hand-down) imitating a *points* Walt under a *make* live seat. Self-play wall,
200 deals: 23 s with the net field vs 82 s for the champion (3.6×); opening
lead 62k vs 4.3M cells (70×). Calibration with the net field (fresh 200 deals,
seed 20001): forecast +6.8 / 38% vs real +9.3 / 42% (champion on the same
deals +7.0 / 38% vs +8.1 / 40%). A net trained on make mass read off the
points splat imitated worse in both units; cut.

**The make run (the #107 re-pour, `k1,n512,m8` under `make`, 5,000 deals,
seeds 1001.., 87,403 positions; opening forecast +6.3 / 38% vs real +6.1 / 37%).**
First attempt was a units bug worth keeping: the trainer divided the
advantage by τ = 2 *points* while the make plan's q is a probability, so every
target was flat (loss stuck at 1.19 = the entropy of a uniform guess, zero
hinges) and the arena faced a random field: **−1.57 [−1.85, −1.26]**, made
26.5% vs 40.5%. Not a verdict on anything but the units. Fixed by reading the
target off the margin splat (what #107 was for); retrained in 9 s, held-out
regret 0.67 points/decision (random 2.38), 437 hinges in the held-out tenth.

| `bid30+walt:k0,n512,F=make-net` vs champion `bid30+walt:k1,n512,m8` | |
|---|---|
| mark margin, 384 mirrored games, seed 1 | **−0.16 [−0.48, +0.14]** |
| contracts made | 36.4% vs 37.8% |
| point margin per hand | +0.04 |
| 200-deal self-play wall (seed 20001) | 14 s vs 53 s (3.8×) |
| calibration, same 200 deals, net field | forecast +5.9 / 39% vs real +7.6 / 38% |
| calibration, same 200 deals, champion | forecast +5.6 / 37% vs real +6.9 / 38% |

Same picture as the points-trained net (−0.23): **a cheaper equal, not a
stronger player.** The field net is the model of the other seats; a better
imitation of Walt cannot make the planner above it stronger, only cheaper, and
the champion's field was already a rung-0 Walt. The interval includes zero at
a quarter of the cost, and the calibration holds. Where it earns its keep is
the pour price: whole deals and 10× datasets for the same wall. It stays out
of the champion's lineup; the champion stays `k1,n512,m8`. Data scaling is
the lever that moved it (200 deals: −0.70; 5,000: −0.16/−0.23), so the next
net is trained on a bigger pour, poured by this one.

**Two targets, the bid, and a GPU path (2026-10-10, layout `field-v2`).**
Built for the 50,000-deal pour the net field is writing. *Target `pmake`* is
texas-42 PR #102's faithful recipe: per legal tile the belief mass in which
the mover's side gets its way at the pour's bid, read off the margin splat
(`atlas.makes`), trained by per-tile BCE on sigmoid outputs with the illegal
tiles masked out. The defender's sign is folded into the target (P(set) =
1 − P(make)), so the score is the mover's own success logit and the integer
selector, referee and core, is unchanged. *Target `margin`* is the one above.
*The bid* joins the features as a 13-row one-hot over 30..42 (a mark bid
reads as 42), referee and core alike (343 features; old nets refused by
layout). Every pour so far is bid-30, so the row is constant in training:
plumbing for a pour with varied contracts, not yet signal. *MLX*
(`--device mlx`) runs the same init, batches, masked loss and Adam step on
the GPU, numpy staying the reference; a test holds them to float32 rounding.
Held-out tenth of the 5,000-deal `make` pour (8,659 positions, 437 hinges;
random pick gives up 2.44 points and 3.9 pp of make probability):

| net | regret, points | regret, pmake | agrees with Walt's move | hinge regret | train wall numpy / MLX |
|---|---|---|---|---|---|
| `margin` 128×64 | **0.65** | **1.3 pp** | 49% | **1.2** | 5 s / 8 s |
| `margin` 512×256 | 0.70 | 1.4 pp | 48% | 1.7 | 19 s / 9 s |
| `pmake` 128×64 | 1.38 | 1.9 pp | 45% | 4.5 | 5 s / 6 s |
| `pmake` 512×256 | 1.25 | 1.7 pp | 46% | 4.0 | 18 s / 7 s |

The margin target is the better model of the `make` teacher *in its own
units*: under `make` most positions are already decided and their pmake
vectors are flat, so BCE has nothing to learn there, while the margin still
ranks the moves the way Walt's tie-broken play does. 512 wide overfits 87k
rows (validation loss bottoms by epoch 8–16). Arena, the hub's shape (384
mirrored games, seed 1, vs `k1,n512,m8`, run beside the 50k pour so walls
are inflated):

| field net | mark margin | made | wall |
|---|---|---|---|
| `pmake` 512×256 | **−0.46 [−0.76, −0.15]** | 34.6% vs 38.8% | 70 min |
| `margin` 128×64 | **−0.01 [−0.31, +0.30]** | 37.0% vs 36.9% | 19 min |

The margin net is the same cheaper equal as before (−0.16 with the old
layout; the same recipe plus a constant bid row, so the two are one result
seen twice, inside each other's noise). The faithful target loses at power
on this pour; #102's +.100 came from a
teacher whose pmake was not flat (a 64-world search with varied bids, 7.6M
rows). The 50k pour trains with the margin target first.

**The 50,000-deal run — and the name (2026-10-10, late).** Poured by the
cheap player itself (`k0,n512,F=make-net-v2`, make utility, seeds 100001..,
**864,604 planned positions**, 3.2 GB, 2 h 8 min; opening forecast +5.5 / 38%
vs real +5.8 / 37% — calibrated to a third of a point over fifty thousand
hands). Trained on it with the MLX path (about 70 s each), graded on the
held-out tenth and in the arena against the champion `k1,n512,m8`:

| net, 50,000 deals | regret (points) | hinge regret | agree | arena, 384 games, seed 1 | made | arena wall |
|---|---|---|---|---|---|---|
| margin 512×256 | **0.38** | **0.45** | 53% | **+0.11 [−0.20, +0.40]**, +0.38 pts/hand | 38.3% vs 37.4% | 2,083 s |
| margin 128×64 | 0.41 | 0.44 | 53% | **+0.06 [−0.25, +0.38]**, +0.31 pts/hand | 36.8% vs 36.4% | 727 s |
| pmake 512×256 | 0.97 | 2.52 | 48% | — | — | — |

Calibration with the wide net (fresh 200 deals, seed 20001): forecast +5.9 /
39% vs real +8.6 / 40%.

**Data is the lever, and it has not stopped working:**

| field net trained on | arena vs champion |
|---|---|
| 200 deals | −0.70 [−1.44, +0.05] (64 games) |
| 5,000 deals | −0.16 [−0.48, +0.14]; −0.01 [−0.31, +0.30] |
| 50,000 deals | +0.06 [−0.25, +0.38]; +0.11 [−0.20, +0.40] |
| 250,000 deals | +0.11 [−0.19, +0.41]; +0.13 [−0.17, +0.44] |

Every step is inside the noise of the last; the trend across three steps is
not. The point estimate crossed zero and the point margin per hand is now
positive. The cheap player — a one-rung Walt planning against a learned
field — is at parity with the two-rung champion. The wide net buys nothing
the small one does not and costs 2.9× the arena wall (the forward pass is now
the hot spot), so **the small net is the pour engine**. The pmake target is
confirmed worse at scale. The champion stays `k1,n512,m8` until a net field
beats it with an interval clear of zero; the next net trains on the next
pour, poured by this one — expert iteration, with the net kept in the field
(done the next morning: see the 250,000-deal pour below).

**The 250,000-deal pour (2026-10-11): the data lever stops at parity.**
Overnight, four more 50,000-deal chunks (seeds 200001, 250001, 300001,
350001; 869,355 / 870,209 / 869,807 / 869,451 positions; 90–92 min each,
never one long run) poured by the small 50k net itself — expert iteration's
second turn. Opening calibration on every chunk: forecast +5.6 / 39% vs
real +6.1 to +6.3 / 38%. With the first 50k that is 4.34M positions from
250,000 deals; `field_train` now takes several pours as one dataset
(7c31bbf9, deals renumbered so the held-out tenth stays whole deals).
Trained with MLX (dataset 3.7 min, training about 2.3 min each), graded on
the held-out tenth (434,535 positions) and in the arena against the champion:

| net, 250,000 deals | regret (points) | hinge regret | agree | arena, 384 games, seed 1 | made |
|---|---|---|---|---|---|
| margin 128×64 | 0.35 | 0.34 | 53% | **+0.11 [−0.19, +0.41]**, +0.15 pts/hand | 37.4% vs 36.5% |
| margin 512×256 | **0.26** | **0.19** | 55% | **+0.13 [−0.17, +0.44]**, +0.09 pts/hand | 37.5% vs 36.4% |

The imitation kept improving — the wide net finally has the rows to use its
capacity (regret 0.38 → 0.26, hinge regret 0.45 → 0.19 from 50k to 250k) —
and the arena did not move: +0.11 → +0.13, inside the noise of a flat line.
That is the "cheaper equal" argument arriving as a measurement. The field is
the model of the other seats; a perfect imitation of the cheap player's own
play can only make the cheap player's field more like itself, and the
planner above it is still a one-rung Walt. Five times the data bought a
better model and the same player. Parity with the two-rung champion at a
quarter of the cost is the result; a stronger player will not come from
more of this pour. The champion stays `k1,n512,m8`. The four chunks are
published as
[jasonyandell/texas-42-zeb-pour-200k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-200k)
(CC0; a separate repo because the chunks' pour spec names the 50k net, not
the 5k one).

Where strength could still come from, now that imitation is not it: the
live seat (racing among near-ties, a deeper plan priced by the cheap field —
`k1` over the net field costs more than the champion today), the utility
(the pours are all bid-30 `make`; a varied-contract pour would give the bid
row signal), or the modeled seats learning something other than Walt's
move — #94-style mining of the splats for what must be true. None of these
is more data of the same kind.

**Zeb.** Jason (2026-10-10): "can't keep calling this Walt. Walt is just an
ingredient." The family of things built on Walt — the atlas, the pour, the
field net, whatever comes next — is **Zeb**, the first project's name
reclaimed ([[zeb]] was the January AlphaZero-style self-play player, superseded
on fused labels). Not diverging from Walt, evolving; no self-play mandate
comes with the name. The 50,000-deal pour is published under it:
[jasonyandell/texas-42-zeb-pour](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour),
CC0 (a separate repo because `scenario/hf.py` refuses to mix specs in one;
the k1 pours stay in [texas-42-walt-pour](https://huggingface.co/datasets/jasonyandell/texas-42-walt-pour)).

## Links

[[the-wall]] · [[the-wall-biography]] · [[belief-policy-value-algebra]] ·
[[strategy-fusion]] · [[expected-q-value]] · [[arena]] · [[walt]] ·
[[auction-decoder]] · [[gus]] · [[consumption-ledger]]
