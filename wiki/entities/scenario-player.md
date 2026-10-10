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

## Links

[[the-wall]] · [[the-wall-biography]] · [[belief-policy-value-algebra]] ·
[[strategy-fusion]] · [[expected-q-value]] · [[arena]] · [[walt]] ·
[[auction-decoder]] · [[gus]] · [[consumption-ledger]]
