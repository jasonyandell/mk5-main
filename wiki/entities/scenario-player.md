---
title: The scenario player — Walt, rebuilt here from the mathematics
kind: entity
first_seen: 2026-10-09
last_updated: 2026-10-14
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

## Where it stands (2026-10-12)

- **The champion is `walt:k1,n512,m8`** (two rungs, 512 root worlds, floor
  16). It stays champion until a field-net player beats it with an interval
  clear of zero; none has (best: +0.13 [−0.17, +0.44], the 250,000-deal net).
- **Deployed Zeb** is `k0,n512,F=margin_d250k_h128x64,f=16` (one rung over
  the 250k field net, trained on
  [texas-42-zeb-pour-250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k);
  the package defaults to 160 worlds on a phone),
  published as `@jasonyandell/zeb` **0.2.0**
  (2026-10-12: the floor spares enumerated fibers, the seed is
  rung-independent; 35 parity positions). See "Zeb, the package" below.
- **Defaults after the foundation repair** (below): the cell floor acts only
  on a sampled fiber, an enumerated one is planned exactly (#110); a rung's
  sample is seeded by its coordinate, not the rung (#112); the field net is
  scored by regret under q, not agreement (#109); the field is deterministic
  unless `field_eps` > 0 (#108); the live worlds are uniform unless
  `draw="posterior"`, which beats uniform +0.026 ± 0.003 marks/hand against
  a table of nets (#111) and is not turned on in Zeb or the champion.

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
| `scenario/reason.py` | the reason report (#124): one decision, Zeb's tile against a named one — the line, the split, the separating facts in [[silas]] (one fact or two, with coverage, against a threshold and a chance level set before looking), counterexamples, a noise bar (see "The reason report") | the book's 84 finds its own set condition at 33/95 exact, as one fact; a split built from two facts is found as those two; a real price the space cannot carry says *no reason found*; a tie is indifferent; a price inside the bar is indistinguishable; the referee writes the same report |
| `arena/walt_play.py` | the [[arena]] `PlayPolicy`; spec `walt[:k<level>][,n<outer>][,m<inner>][,h<horizon>][,g<inner horizon>][,u=make\|points][,b=<belief>][,e=<eps>][,f=<floor>][,F=<net>][,E=<field eps>][,d=uniform\|posterior][,D=<budget>][,t=lowest\|margin][,py]` (`py` = the pure Python referee; `t=` the tie rule, [[tie-margin-2026-10-14]]) | an arena half completes |

Tests: `python -m pytest scenario/tests -q` (CPU, no checkpoint). Build the
core once with `python -m pip install ./scenario/core` (Rust toolchain).

The Python planner is the referee and the compiled core is the runtime. To
make that checkable, every random draw is defined: scenario sampling and the
dice tape run on SplitMix64 streams seeded from the invocation's coordinate
(config seed, viewer, hand, packed record, voids; not the rung since
2026-10-12, #112, so every rung at a coordinate shares one sample), and the
record is keyed as one
integer, not a CPython hash. The core mirrors CPython 3.12's compensated
`sum()` so even tilted-belief values agree to the last bit. The belief
weights the live rung only; modeled minds are compatibility-only (g ≡ 0),
as the shipped player does.

## The levers (Θ = (G, I, u, β, S, k, H, L, ≺, Γ))

Every choice the mathematics names has a seat in the code: `β` is a
callable `(record, viewer, worlds) → weights` on `Walt(belief=…)`
(uniform by default — the g ≡ 0 limit; since 2026-10-12 the live worlds can
instead be drawn from the field's posterior, `draw="posterior"`, #111); `k` is the number of rungs; `S`
is `n_worlds` per rung; `H` is `horizon` per rung (plays searched
explicitly; `None` = to the end); `L` is the tape playout at the
frontier; `u` is `make` or `points`; `≺` is the tie rule, `Config.tie`: lowest tile id by default, or make then the plan's mean margin ([[tie-margin-2026-10-14]], #120); `Γ` is the
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

- **dice** (retired 2026-10-12) — the hidden seats' discretionary plays under random legal play: a
  play with n options weighs 1/n (forced-looking plays count more).
- **field** (retired 2026-10-12) — the same plays under the rung-0 player: ε/n + (1−ε)·[it would
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
is what makes any sharp belief (a learned head, a convention model) usable at
all; built 2026-10-12 for the field's posterior (#111, "Foundation repair"
below).

## The atlas (2026-10-10)

Beliefs paused after night 1 (the taste said *uniform again*; a dozen other
low-hanging things come first; resumed 2026-10-12 as the field's posterior,
[[research-night-2026-10-12]]). The new object is **the atlas**: one fully known
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
Since 2026-10-12 (#110) the floor acts only on a sampled fiber; an enumerated
one is planned exactly, floor 1 ("Foundation repair" below).
**The arena default is the honest plan from 2026-10-10:** `walt:k1,…` means
`f=16`; `f=1` restores the clairvoyant tail (Jason: "perf with noisy to
similar outcomes are a win… the aesthetics are a factor too" — the math said
one action per cell, and a one-world cell was breaking that promise from
trick 2 on).

**The floor calibrates the forecast (2026-10-10, `floor_forecast.py`,
`marginal3.py`; every number in this paragraph is the 2026-10-10
rule, the floor on every thin cell, with the rung-mixed seed).** Same hand, same lead question, root forecast by floor, and
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
branched once (own-hand only; hidden tiles never compress, see
[[endgame-equivalence-census]]), the exact tail at no cost. Work proceeds in **waves**: label a
frontier in one `decide_batch`, keep survivors, the next frontier is known
before it is asked for (the tape). Mac for the Walt labeler, GPU for the net.

## The net in the field (2026-10-10)

A learned policy as the modeled seats, **and nowhere else** (Jason's binding
rule: the net never enters a value or a leaf; the live seat still solves).
One knob: **rung −1 is the net instead of the dice** (a deterministic field;
since 2026-10-12 `field_eps` = ε > 0, arena `E=`, makes it stochastic, F_σ,
default 0, #108). `walt:k0,n512,F=<npz>`
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
of the champion's lineup. Data scaling is
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

| net | regret, points | regret, pmake | agrees with Walt's move (retired 2026-10-12, #109: exact ties make it a coin flip) | hinge regret | train wall numpy / MLX |
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

| net, 50,000 deals | regret (points) | hinge regret | agree (retired) | arena, 384 games, seed 1 | made | arena wall |
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
confirmed worse at scale. The champion rule ("Where it stands") dates from
here; the next net trains on the next
pour, poured by this one — expert iteration, with the net kept in the field
(done the next morning: see the 250,000-deal pour below).

**Zeb.** Jason (2026-10-10): "can't keep calling this Walt. Walt is just an
ingredient." The family of things built on Walt — the atlas, the pour, the
field net, whatever comes next — is **Zeb**, the first project's name
reclaimed ([[zeb]] was the January AlphaZero-style self-play player, superseded
on fused labels). Not diverging from Walt, evolving; no self-play mandate
comes with the name. The 50,000-deal pour is published under it:
[jasonyandell/texas-42-zeb-pour](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour),
CC0 (a separate repo because `scenario/hf.py` refuses to mix specs in one;
the k1 pours stay in [texas-42-walt-pour](https://huggingface.co/datasets/jasonyandell/texas-42-walt-pour)).

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

| net, 250,000 deals | regret (points) | hinge regret | agree (retired) | arena, 384 games, seed 1 | made |
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
more of this pour. The four chunks are
published as
[jasonyandell/texas-42-zeb-pour-200k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-200k)
(CC0; a separate repo because the chunks' pour spec names the 50k net, not
the 5k one). Republished whole (2026-10-12) as
[jasonyandell/texas-42-zeb-pour-250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k):
five splits, the three field nets in `nets/` with sha256, the repo to cite (the
two earlier cards describe the planner as `k1,m8,u=points`, which it is not);
[[publish-data-to-huggingface]].

Where strength could still come from, now that imitation is not it: the
live seat (racing among near-ties, a deeper plan priced by the cheap field —
`k1` over the net field costs more than the champion today), the utility
(the pours are all bid-30 `make`; a varied-contract pour would give the bid
row signal), or the modeled seats learning something other than Walt's
move — #94-style mining of the splats for what must be true. None of these
is more data of the same kind.

**A live bidder, graded against Plunge's book (2026-10-11).** Plunge's
deployed Walt does not bid: it looks up a frozen book — 130 catalogue deals,
520 hands × 9 declarations, complete-game score tails from 318,472 bid-30
games — and takes the highest target some declaration clears in 4 games of 5.
Hence the repeats Jason's players see. A live bid is the same quantity
computed instead of stored: one root analysis per declaration, the bidder's
seven tiles only, pmake(b) the chosen lead's tail mass at margin ≥ 2b − 42
(#107's splat; `scratch/bid/`). Cost on the M5, seven pip declarations:
cheap player `k0,n512,F=` 0.8–1.0 s, `k0,n160,F=` 0.2–0.25 s, champion 4.7 s.
Graded on every book panel (4,680; the book's hand and declaration ids are
the same triangular ids; doubles and no-trump forecast by `k1,n160,m8` since
the net is pip-only):

| | |
|---|---|
| mean \|ours − book\| at target 30, games-weighted | 0.052 (the book's own binomial sd at its sample sizes: 0.066) |
| correlation across panels, 30 / 36 / 42 | 0.87 / 0.85 / 0.75 |
| calibration, bins of our pmake30 → book's make rate | 0.16→0.16, 0.24→0.24, 0.35→0.35, 0.45→0.46, 0.55→0.54, 0.64→0.63, 0.75→0.72, 0.85→0.82, 0.95→0.90 |
| best declaration at 30 agrees | 386 / 520 hands |
| bid decision under the book's rule | both pass 442; same target 22 (same declaration 21); ours a notch higher 36, lower 3; ours bids where the book passes 15, the reverse 2 |

The forecast disagrees with the book by less than the book disagrees with
itself, and is calibrated through the middle. Two honest edges: on the
8-game screened panels the book reads 0.08 where we read 0.21 — the
campaign stopped those hands early *because* they scored low, so that bias
is the book's (40-game panels: 0.291 vs 0.291; 160–640: 0.579 vs 0.585); and
at the top the live forecast runs 3–5 pp bold (0.85 → 0.82, 0.95 → 0.90), so
under the 4/5 rule it bids one target higher than the book on about 7% of
hands. The book's tails are the *deployed* Walt's own play; ours forecast
the cheap player's, which the pour calibration shows honest against itself.
A cutoff of 0.85, or more worlds, would close the gap if one wanted the
book's temperament. Worth knowing: Plunge once had a live Walt bidder and
switched it off — its field was the dice ("the price curve saturates against
walt's level-0 field model, so even θ = 1 overbids at the table",
`src/ai/walt/requests.ts`), so the book was the workaround. The net in the
field is what makes the live price honest; that is the whole difference. Nothing here is a deployment: Plunge runs texas-42's Walt
in WASM, and `scenario/` is this repo's core — data crosses, code does not.

**Zeb, the package (2026-10-11).** The cheap player as an npm package,
[`@jasonyandell/zeb`](https://www.npmjs.com/package/@jasonyandell/zeb)
(`zeb/`; MIT): the planner and the 250k field net compiled to WebAssembly
with a JavaScript surface of five calls — `forecast` (one plan per
declaration from the bidder's chair, pmake at every target 30..42),
`chooseBid` (the book's rule as a pure function: highest target some
declaration clears at the cutoff, default 0.8), `play`, `analyze` (each
legal tile's tails, for hints), and the receipts `version` / `netId`. Built
by an Opus agent, reviewed by a second one cold; both reports and the fixes
are in the session. What made it possible: `scenario/core` is now a plain
Rust library — the rules tables derived in Rust and held equal to
`forge.oracle.tables` for all ten declarations by a test, the canon a
function of the declaration, the net one blob (`FieldNet.to_bytes`, magic
`FNET`, layout `field-v2`, int16 weights, int64 biases) — with pyo3 behind a
`python` feature, so the WASM and the Python referee compile the same
planner. Facts a reader can check, as of 0.1.0 (0.2.0 adds a 35th parity
position and changes the wasm bytes; "Foundation repair" below):

| | |
|---|---|
| parity | 34 positions across all nine straight declarations, bit-identical to `scenario/planner.py` on tiles, q, weights and every scenario's margin; the fixture regenerates byte-identically from the pure-Python referee |
| wasm | 207,640 bytes (137 KB gzipped), 110 KB of it the net; the build reproduces byte-for-byte from a clean target (toolchain 1.95.0 and wasm-pack 0.15.0 pinned, paths remapped) |
| latency, Node 24, M5 | 7 pip declarations 0.30 s at 160 worlds / 1.2 s at 512; doubles 0.30 s; no-trump 0.23 s; an opening lead 44 ms |
| input | every play replayed and checked, then the fiber's completion count must be nonzero; the reviewer fuzzed ~1,800 games and ~9,000 mutated records through the WASM with no reachable panic |
| plans | pip trumps `k0,n<worlds>,F=net` (floor 16, on sampled fibers only from 0.2.0); doubles and no-trump `k1,n<worlds>,m8`, since the net is pip-only; declaration 8 (nello's algebra) refused |
| default | 160 worlds is a phone budget, not the graded setting (512 for pips); the README says so and names the lean it buys |

Publishing (0.2.0 is current, 2026-10-12): 0.1.0 went out by hand from this
machine on 2026-10-11 (npm
requires 2FA; a one-time link each for login and publish). The registry
has it: 10 files, 247,827 bytes unpacked, shasum `30aee968…`, and a clean
`npm install` runs the forecast. From here
`.github/workflows/publish-zeb.yml` publishes every later `zeb-v*` tag with
provenance through npm's trusted publishing, no token anywhere (connected
2026-10-11 with "allow npm publish" ticked, so a tag goes live without a
visit to npmjs.com; npm's newer default stages for a human to promote, a
knob to flip if the package ever gains maintainers); the old
Pages deploy, which had failed on every push for months, is gone. Plunge
adopting it is Plunge's work: swap the book auction for `forecast` +
`chooseBid`, keep its partner-pass and forced-30 rules.

**The splats, drawn (2026-10-11).** `scenario/viz/` draws a pour in the
browser — every legal move's splat at every planned play, the means alone
beside them, and the deal's real ending as one line through every forecast:
[[splat-visualizer]].

## Foundation repair (2026-10-12)

**Research night (2026-10-12): does Zeb find the optimal move in finite time?**
Seven agents, one question, see [[research-night-2026-10-12]] (lab outputs, with
the day's arenas and endgames: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10),
published 2026-10-12). Exact Walt does,
against a fixed field, when its belief is that field's posterior (proved; Lean at
`scenario/lean/`, 15 theorems, no sorry). Deployed Zeb does not, in four measured
ways: the belief (uniform vs the posterior: **+0.028 ± 0.006 marks/hand** when the
other seats really are the net, p = 1.4e-5; sample it, never reweight; null at power
in self-play, +0.003 ± 0.004 on 2,400 deals, so the binding constraint is the model of
the other seats), sampling (moves flip, value does not), the floor
(a pure tax on exact fibers; floor-off-when-exact built and null in marks), and the
referent (the net reproduces every decided move already, which is why data stopped).
The fixed-belief ladder settles by the 24th application to a unique fixed point;
coherent beliefs cycle. The sharpest line: with 512 worlds and floor 16, **the plan
decides the root and then 0.19 of 2.6 own decisions; Zeb is one policy-iteration step
of its own net, evaluated by rollout**, and the +0.56 was the worlds, not the floor.
What was missing mathematically that night: a stochastic field (the prerequisite for
any belief on the seam), regret instead of agreement as the net's metric, a per-cell
certificate, and an O2-vs-O1 number on endgames. All but the certificate landed the
same day, below ([[endgames-o2-vs-o1-2026-10-12]] for the last).

**The net is scored by regret (2026-10-12, #109).** `field_train` grades the
field net by what its move gives up against the best legal move under the
splat's own q (`regret_q`, in the plan's units: make probability under `make`)
and by the share of decisions where its move is a best one, any member of an
exact tie (`at_best`); points regret stays beside them. Agreement with the
teacher's tile is retired: 285 of 714 `make` decisions are exact ties, where the
teacher's tile is the lowest id of a seed-dependent set, and the teacher agrees
with itself across seeds only 0.89 of the time (critic §3.2). The shipped
`margin_d250k_h128x64` on the 40-deal research-night pour (n = 714 decisions):
**regret 0.0064 make probability per decision** (a uniform legal pick 0.044),
at a best move 77.5%, 0.29 points; identical to the referee's own pick.
`--grade NET.npz` scores any saved net on any pour.

**The floor guards only a sample (2026-10-12, #110, Zeb 0.2.0).** When the live
fiber fits in the rung's worlds it is enumerated outright, its cells are the exact
posterior under the model, and a max over a thin one is a best response, not
clairvoyance; so an enumerated invocation is never floored. This is now the only
behavior, in both engines (the night's opt-in `Rung.sampled_floor` / arena `x` is
gone), bit-identical referee to core. At n=512 it acts from the third play of trick 6
(sooner with voids) and with three or more tricks left: 26 of 2,771 late decisions
changed, and the arena is null (−0.02 [−0.35, +0.29] marks/game at k0,
−0.01 [−0.33, +0.29] at k1; [[research-night-2026-10-12]]). It is correctness, not
strength. `@jasonyandell/zeb` 0.2.0 carries it; its parity fixture gains a position
where the rule moves q (35 positions).

**The seed no longer mixes the rung (2026-10-12, #112).** A rung's worlds and dice
tapes are seeded by its coordinate (viewer, own hand, record) and never by k, in
both engines, bit-identical: every rung at a coordinate plans over the same sample
(a smaller rung's is the larger's prefix), so the sampled ladder is the iteration of
one operator and the stabilization theorem in `scenario/lean/` describes the code
(the rung-mixed seed was its `rung_seeded_no_settle` witness). Bytes moved, the
distribution did not; Zeb 0.2.0's parity fixture is regenerated. Re-graded on the 0.2.0 bytes (2026-10-12, main `4bd19fc9`): deployed Zeb `k0,n512,F=margin_d250k_h128x64` vs the champion `k1,n512,m8`, the hub's block (384 mirrored games, seed 1), **+0.04 [−0.27, +0.33] marks/game**, 194/384, made 36.9% vs 36.6%, 785 s; the standing block was +0.11 [−0.19, +0.41], so parity holds and the champion stays `k1,n512,m8`.

**A stochastic field, so the posterior is a real object (2026-10-12, #108).** With the
deterministic net as F, β^F(world | record) puts zero mass on the real deal at 586 of
674 self-play positions, so every belief on the seam was degenerate. `Config.field_eps`
= ε (arena `E=`) makes the net's field F_σ: in each scenario a modeled seat explores
with probability ε, playing the dice's tile from the record-keyed tape, else the net's
move, so P_σ(t | hand, record, world) = ε/n + (1 − ε)·[t = the net's move] over the n
legal tiles. The coin is one more SplitMix step past the dice's draw, compared with
⌈ε·2^53⌉, integer-exact, so referee = core bit-identical; worlds still partition; floored
moves stay the net's. Default 0 is the deterministic net byte for byte (Zeb unchanged).
`Walt.field_likelihood` is the exact likelihood of that field (referee only; the core's is
#111). On the research-night pour (674 positions after play 0) at ε = 0.05 / 0.1 / 0.2:
the real deal ranks at the 71st percentile of the 512-world sample at every ε (median
rank 0.74 at plays 1–3, 0.84 at 12–15; ε = 0 reproduces the 586 exclusions); the weights' median ESS is 7.5 / 9.8 / 16.1 of
512, falling from 118–139 at plays 1–3 to 8–16 at plays 12–15 (1.6–3.2% of the sample).
Reweighting 512 uniform draws leaves about ten worlds mid-hand, so #111 must sample from
β^F, not reweight. `scratch/stochastic-field/measure.py` (12 s) and `measure.json`.
2026-10-12: the old tilt (`Tilt`: `b=dice`, random play, and `b=field`, the rung-0 solver,
both reweighting a uniform sample) is gone from both engines and the spec, so
`field_likelihood` is the one likelihood of the plays, applied once, by #111's draw.

**The posterior, sampled in the core (2026-10-12, #111).** `Config.draw = "posterior"`
(arena `d=posterior`, budget `D=`, default 2^22; a one-rung player with a field net)
draws the live seat's worlds from β^F ∝ uniform × `field_likelihood`, bit-identical
referee ↔ core. An enumerated fiber is weighted by L normalized (the exact posterior,
never floored); a sampled one is n draws by rejection from the uniform stream itself
(a proposal is kept when a coin on its own stream falls below L ≤ 1; at ε = 0 L is the
net's 0/1 filter, no coin, and the uniform sample's consistent worlds are the
posterior's first), equally weighted; no explained world, or n not found within the
budget, plans over the uniform sample and is counted. Default `uniform`: every
deployed byte unchanged (Zeb's fixture untouched). Cost is small because the product
short-circuits at the first miss and the net's moves are cached: at k0,n512 the
posterior decision costs 135 / 162 / 91 / 31 / 5 ms at plays 1–3 / 4–7 / 8–11 / 12–15
/ 16–19 against the uniform's 120 / 125 / 70 / 26 / 4 (median proposals 1.7k / 4.5k /
17k / 22k; p99 ≤ 0.47 s, max 1.3 s), so E1b's 2–6 s Python referee becomes
milliseconds. **Arena, E1b's design** (Zeb in one seat, the deterministic net in three,
seat 0 bids 30, Zeb bidding on even seeds and defending on odd, each deal played by
both lines): posterior vs uniform Zeb **+0.026 ± 0.003 marks/hand [+0.023, +0.029],
n = 22,000 paired deals**, 962 flips for / 386 against (sign p ≈ 5e-57), +0.025
bidding, +0.027 defending; on E1b's own 2,120 seeds +0.022 ± 0.011. Fallbacks 95 of
98,060 decisions (0.10%; 0.74% at 2^20). So against the net the repair is real and
the full E1b size. Whether Plunge's Walt partner is "net enough" is the other half:
in Zeb self-play the deterministic posterior is empty (17 of 36 probe positions fall
back at 2^20, 10 of the 12 at plays 16+), and the softened one was null at power
(E1c), so the filter pays only for seats that play like the net, and no seat that
is a Walt yet does. `scratch/posterior-arena/` (`table.py`, `probe.py`, `report.py`,
`arena.log`, `report.json`).

**Reading the plays by the net's set (2026-10-14).** `Config.reading` (arena
`d=posterior,r=consistent|uniform:<margin>`, default off, both engines bit-identical) makes the
posterior's likelihood the field net's *set* — the legal tiles it scores within a margin of its
best — so a play weighs [tile ∈ A] or [tile ∈ A]/|A| (restricted choice), and a fiber of at most
2^20 is enumerated and drawn from exactly. Against plain Zeb the reader is a null under both
likelihoods (−0.09 and +0.01 marks/game, restricted choice +0.10 [−0.15, +0.36]); it changes
10–11% of exact-fiber moves, on the 5-5 and 6-4 first, but the net's set fits a Zeb's record so
poorly late in the hand that half of trick 6 falls back to the uniform: [[reads-plays-2026-10-13]].
**Softened, it pays (2026-10-14, #145):** `Reading.eps` (arena `r=<likelihood>:<margin>:<eps>`,
default 0, both engines bit-identical) weighs a play ε/n + (1 − ε)·[tile ∈ A]·w, F_σ's ε over the
set, so the posterior never empties. At ε = 0.28 (fitted to Zeb's choices) the fallback rate is 0 at
every trick. The reader gains +0.23 [+0.05, +0.41] marks/game at a table of seats that play the
model. Against plain Zeb it first measured +0.30 / +0.29 (consistent / uniform; A/A control −0.03). The
registered replication on fresh deals shrank that. The consistent primary came in at **+0.12
[−0.02, +0.27]**, which **failed** its bar. Uniform held at +0.23 [+0.08, +0.37]; A/A −0.04.
The reading costs up to 1.7× plain's median per decision at tricks 2–3 and nothing measurable
later. Restricted choice stays null at both tables, so the gain is the soft posterior's, not the
set size's. Not turned on in Zeb, and the zeb package cannot reach it:
[[reads-plays-soft-2026-10-14]].

**A convention overlay, default off (2026-10-12).** The book's conventions
(`scratch/book/CANDIDATES.md`) are joint deviations, so the overlay has two halves
in both engines, bit-identical referee ↔ core (`scenario/convention.py`, core
`convention.rs`; `Config.convention = Convention(name, send, read)`, arena
`c=<name>:S|R|SR`). A rule is a gate in the mover's chair (seat role, own hand,
record) naming the allowed tiles. *Send*: the plan runs and its q is reported, but
at a gate the seat plays the plan's best allowed tile (`sent` counts it). *Read*:
the live seat draws its worlds from the rule-only posterior, #111's rejection from
the uniform stream under the 0/1 likelihood that the partner's gated decisions
obeyed the rule in that world (enumerated fiber: the filter normalized; none
survives or the budget runs out: the uniform, counted in `fallbacks`). A reader
composes with `draw="uniform"` only. Four rules: `handoff` (C2, the table's
high-trump handoff), `handoff-singleton` (the same sender, read the book's way:
a top trump from partner was his only one), `biggest` (C3) and `donation` (C1).
The decision request now carries the bidder seat, not the team; with no
convention every byte is main's (Zeb's fixture untouched). The senders' price on
250,000 Zeb self-play deals (the research archive's `book/wave1/`): the handoff
fires in 19% of bid hands at −0.039 pmake per firing (55% ties), biggest count
in 5% at −0.020, the donation in 17% at −0.007 (85% ties). Graded the same day in
S/R/SR arms, two Zebs vs two nets, 22,000 paired deals per arm
([[book-study-2026-10-12]]): the donation is null in every arm; the handoff's sender
costs −0.013 [−0.016, −0.010] marks/hand and its reading recovers 0.000 ± 0.002 on
top; the book's singleton reading of the same sender costs −0.031, because a reading
that is not the sender's gate inverted plans on false worlds. No convention confirmed;
nothing adopted as a rule; the overlay stays default off. Its gates are hand-written Python; the book's rules move to [[silas]] policies, text that is the code.

**O2 beats O1 on real endgames (2026-10-12, #113).** On 566 three-trick endgames
from 142 deals (V's hand common knowledge, P's private, opponents the deployed net,
margin utility), the common-information coordinator O2 beats the Walt partnership map's
fixed point by +0.027 [+0.022, +0.033] margin per endgame (49% of endgames), and plain
best-response dynamics by +0.115. Of the Walt gaps diagnosed, 78 of 171 close only when
*both* partners change moves they actually make: a convention, the first measured here
(reverse of mini-42, where every gap stood on an unreached set). Cycles of the Walt map
(149 of 508) are all period 2, off path, value constant. Small in points; decisive about
the referent: the partnership operator is where the next mathematics is.
[[endgames-o2-vs-o1-2026-10-12]].

**An exploiter against Zeb on the same endgames (2026-10-14, #77).** The opponents and
the partner are frozen at a Zeb proxy: the planner's rung 0 over the net, with enumerated
fibers, under `make`. V's exact information-state best response to that table beats V's
best response to the net, played against the same table, by **+0.50 [+0.42, +0.58] margin
per endgame**. That fails the pre-registered sturdy bar (< 0.10) and straddles the
exploitable one (> 0.50).
- Half of it comes from endgames whose contract is already decided, where `make` ties every
  tile and Zeb plays the lowest id: points, never marks.
- In contract terms, knowing Zeb is worth +0.0040 [+0.0028, +0.0054] per endgame, and
  +0.009 where the contract is open. That is under half of what the same seat takes from
  the net, and best-responding to the net takes nothing from Zeb.

[[exploiter-endgames-2026-10-14]].

**The book player against Zeb (2026-10-13).** The book's 18 rules as a [[silas]] policy
stack (`arena/book_play.py`, `book:` in the arena) lose to deployed Zeb by −1.88 [−2.09,
−1.66] marks/game when Zeb plays wherever no rule fires; every departure from Zeb's tile,
rolled out with Zeb continuing, sums exactly to that loss, −0.051 marks a departure, and
Zeb's forecast of each departure's price is honest in sign
([[book-head-to-head-2026-10-13]]). The arena's `WaltPlay` now plans a marks bid as its
42 points. Investigated ([[book-investigation-2026-10-14]]): the arena gained the book's game
(`ArenaConfig.points_to_win`, points to 250 by the book's settlement), where the book loses
too; breaking `make`'s near-ties (within 0.02) by the book's rules wins it for Zeb, 53.2%
[51.1, 55.2] of games, replicated, with no marks moved (#120).

**Make, then margin (2026-10-14, #120).** The tie rule ≺ is now a lever: `Config.tie`
(arena `t=margin`; default `lowest`, every byte unchanged). It breaks a decision's exact ties
of q, meaning equal make counts in its own sample with no bar, by the plan's mean margin for
the mover's side, then the lowest id. Both engines are bit-identical. Re-planned on the
face-off's rows, Zeb has an exact tie for best at 25% (18% dead). The book's tile is the
margin-best tie at 59.7% against a chance of 51.8%, all of it at decided contracts. Book
agreement rises from 52.2% to 54.3%. At the table, against deployed Zeb, it is +0.01
[−0.04, +0.07] marks/game but +4.55 points a hand, and 55% of games to 250. Against the
champion it is +0.18 [−0.01, +0.36]. Nothing ships under the pre-registered marks bar; a
points-scoring zeb version is proposed ([[tie-margin-2026-10-14]]).

## The reason report (2026-10-13, #124)

Zeb priced moves but could not say why. **The plan's line** is now exported:
`analyze(state, bid, line=True)` adds, per root tile, a `Step` for every viewer
cell of the plan — the plays that reach it (the other seats' tiles among them
are the observations it branched on), the tile played, the cell's scenarios, and
whether the floor handed the move down. It is read off the same replay as the
outcomes, in referee and core alike (`core::planner::analyze_line`), and
`test_compiled` holds the two step for step. It is in both because the report
runs at deployed sizes (n512, k1) where only the core is fast; asking for it moves
no other byte, and the default path, Zeb's fixture included, is unchanged.

`scenario/reason.py` (`reason(state, bid, other, config)`, `render`, CLI
`python -u -m scenario.reason SPEC --decl --bid --hands --plays --other`) reads
one decision for a pair of tiles: where they part (A: Zeb's tile succeeds for the
mover and the other fails; B: the reverse; counts and weight shares), the
**separating facts** — a declared space of [[silas]] queries (a seat or "an
opponent" holds t; holds t and another of suit q; is void in q), each evaluated by
Silas on Zeb's own scenarios and weights labeled `sampled(n, seed)`, ranked by
|P(fact | A) − P(fact | B)|, with the fact's exact prior over the fiber
(`uniform_resolving`) — a counterexample world on each side with the plan's path
through it, and a **noise bar**: the decision re-planned at the next seed and one
rung stronger; Silas's three-valued compare of P(Zeb's tile beats the other, ties
halved) against 1/2, at half the bar. Inside the bar: *indistinguishable*, no
reason; an exact tie: *indifferent* (#120). The estimate comes through a new
binding, `SilasBelief.condition_likelihood` (Silas's own conditioning, its evidence
an `Estimate`); `silas/` is unchanged.

**The 84, found by search** (`test_reason.py`, k1 n64 m4, dice field): Zeb's lead
against the book's double-five, and the top fact of the whole space is the book's
own set condition, *an opponent holds the 6-5 and another trump*: in every world
where Zeb's lead makes and the 5-5 does not, and in 2 of 49 of the rest, exactly
33/95 over the fiber; the book's query file reads the same mass on those worlds. Under the
deployed plan (`k0,n512,F=margin_d250k`) Zeb leads the 5-5 itself; against the
5-4 the reason is *an opponent holds the 6-5* (0.75 vs 0), price +0.18 against a
bar of 0.08 — the double ahead of the off.

**The grown space (2026-10-13).** On the face-off's C3 and C21 positions the first
space found no one holding that carried the split, so the space grew, still
declared up front (`reason.space`): to the holdings, a holding with another of a
suit and the voids it adds **position facts** about the seats still to play to the
open trick (holds the led suit; void in it and holds trump; can beat the trick as
it stands, for the partner *can still win this trick*; can take the trick after
the viewer plays t — leading, can beat t led, an opponent can beat t led), **count
at risk** (a count tile the viewer could give is capturable by the next seat; the
tile-relative facts for each legal count tile), and **the book prelude's nouns as
holdings** (a count tile, a ten, a trump, the boss trump, a trump none of mine
beats, a double, the top live tile of a suit). The prelude's frame-only concepts
(the safe trick, can-take-it, may-yet-trump, the dead double) are constant over one
decision's worlds and cannot separate them; they enter as their world forms. On
top, the **conjunction and disjunction of every two** single facts, each one Silas
query (each fact in its own `some` scope), kept where the pair separates more than
either fact. Each fact's truth in every scenario is read by Silas in one pass (a
belief weighted 2^i: the probability's numerator is the truth mask), pairs are
screened in floats and settled exactly, and every reported fact is re-read by
Silas from its own text and must agree. Each fact carries its **coverage**: the
share of each side it explains, and the other facts that read alike ("named
otherwise"). The threshold was set before any position was run: a reason
separates the sides by at least **1/2** and by more than the space's best on **20
random relabelings** of the same scenarios into sides of the same sizes (its
chance level, single against single, pair against pair); two facts are the reason
only when they separate strictly more than a single fact that passes. Below it the
report says **no reason found** and shows the best candidates anyway. Tests: the
84 still finds 33/95 as one fact; on the 84's frame, 128 fixed deals split by *S1
holds the 6-6 and S3 the 6-5* are found as exactly that pair (separation 1,
coverage 100% / 100%, exact prior 7/60); a seeded state whose price clears the bar
says no reason found (best single 0.498, best pair 0.67 under a chance of 0.87).

**On the book study's disagreements** (published at `jasonyandell/texas-42-zeb-research-2026-10` under `zeb/reason-v2/`; the pour player at
current code, `k0,n512`, `make`, field `margin_d50k_h128x64`, seed 0; ten
positions per rule from chunks s200001 up where Zeb's exact-best set is outside
the book set at a price below −0.05, Zeb's tile against the book's best; the six
worked reasons of [[book-study-2026-10-12]]):

| rule | priced outside the bar | one fact | two facts | no reason found | median best single / pair | median chance, one / two |
|---|---|---|---|---|---|---|
| C3 biggest count first | 10 of 10 | 1 | 6 | 3 | 0.79 / 0.82 | 0.41 / 0.52 |
| C7 pounce | 9 | 0 | 6 | 3 | 0.47 / 0.66 | 0.39 / 0.57 |
| C10 trick-one off lead | 9 | 1 | 7 | 1 | 0.48 / 0.59 | 0.37 / 0.48 |
| C12B double-up, lead low | 7 | 0 | 6 | 1 | 0.45 / 0.71 | 0.27 / 0.35 |
| C21 six-five bait | 9 | 0 | 5 | 4 | 0.34 / 0.51 | 0.31 / 0.42 |
| worked: feed the trump holder | 3 of 3 | 0 | 1 | 2 | 0.69 / 0.85 | 0.81 / 0.92 |
| worked: throw a dead double | 3 of 3 | 0 | 2 | 1 | 0.55 / 0.77 | 0.35 / 0.43 |

Of the 44 face-off prices outside the bar, a reason is found for 32 (2 one fact,
30 two) and none for 12; of the six worked reasons, for 3. The first report's two
positions, re-planned under this net: C3 deal 3194 (+0.38, bar 0.025) is found as
two facts (0.81 against a chance of 0.76; the best single, 0.60, is under its
chance of 0.65); C21 deal 3195's opening lead (+0.12) still has no reason (best
single 0.33, pair 0.41). Two facts dominate because a single fact rarely clears
1/2 (18 of 44), and many found pairs are two named holdings, which say where the
split lies but not why. What recurs, by rule:

- **C12B: the guard of the missing trump.** In 4 of 7 the explanation is *the
  partner holds the missing top trump, or an opponent holds it with another
  trump*, true in every world where the book's low lead makes and the double does
  not: the low lead wins only there, and Zeb's double where that trump is bare
  with an opponent. The book's rule reads only the
  bidder's own trumps; the guard, the 84's 33/95 condition again, is one it omits.
- **C10: the partner's ruff.** In 3 of 9, *the partner is void in the off's suit
  and holds trump* (the best single in two, 0.52 and 0.66): the book's off lead
  wins where the partner can ruff it, Zeb's trump lead elsewhere. The book's
  permission does not name it.
- **C3: the safe trick.** In 6 of 10 the opponent still to play holds trump or can
  beat the trick as it stands (one found as *void in fives*, which reads alike):
  the world form of the book's own safe trick,
  which the C3 gate omits and `donate-when-safe` states. Not a condition the book
  omits.
- **C7: whether the bidder's side can overtake** — specific holdings (7 of 9),
  named otherwise as the boss trump or a double (2 each): the world truth of the
  pounce's own guard. **C21:** nothing recurs but guarded holdings of assorted
  suits; four of nine have no reason. **The worked six:** the feed's late-hand
  samples (12 and 30 scenarios) leave a chance level of 0.81–1.00, so its perfect
  separators are not reasons; the dead double's partner holds the boss trump (2 of 3).

## Constellations as the late-hand table (2026-10-14)

texas-42's constellations, the canonical form of a position over the live tiles, are now in
the core (`constellation.rs`, about 0.5 µs a form) with a late-hand ladder over a structural
field keyed by them (`scenario/constellation.py`). Equal constellations plan to equal values
(0 differences on 2,322 substituted pairs); deployed Zeb does not (3.4–20%), because the field
net reads pips and dead tiles. The ladder settles by level 3 with two tricks left, and the 250k
net already plays its best move 96% of the time; a warm fine-tune toward it is inside the noise.
[[constellations-2026-10-14]].

## Links

[[exploiter-endgames-2026-10-14]] (an exploiter against Zeb on three-trick endgames: not sturdy in points, small in contracts) ·
[[constellations-2026-10-14]] (constellations as Zeb's late-hand table) ·
[[splat-search-bounds]] (searching over splats: what is sound) ·
[[lumps-voi-2026-10-13]] (lumps by a hidden tile: hedge or robust, the value of knowing it, restricted choice) ·
[[reads-plays-2026-10-13]] (Zeb reads the other seats' plays: both likelihoods null in marks) ·
[[reads-plays-soft-2026-10-14]] (the same set softened: pays at a net table; against Zeb the registered consistent replication failed, +0.12 [−0.02, +0.27], uniform +0.23; restricted choice null) ·
[[the-wall]] · [[the-wall-biography]] · [[belief-policy-value-algebra]] ·
[[strategy-fusion]] · [[expected-q-value]] · [[arena]] · [[walt]] ·
[[auction-decoder]] · [[gus]] · [[consumption-ledger]]
