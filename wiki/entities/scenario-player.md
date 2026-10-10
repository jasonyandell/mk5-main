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
tests; floor 1 is today's player to the bit; the root is never floored).

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
had the 2-2 four points clear). Whether honesty also *plays* better is the
arena's question (`f=16` vs floor 1, mirrored, 384 games, in flight).

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

## Links

[[the-wall]] · [[the-wall-biography]] · [[belief-policy-value-algebra]] ·
[[strategy-fusion]] · [[expected-q-value]] · [[arena]] · [[walt]] ·
[[auction-decoder]] · [[gus]] · [[consumption-ledger]]
