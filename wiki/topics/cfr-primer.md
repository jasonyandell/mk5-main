---
title: CFR primer — regret at the table, in this project's units
kind: topic
first_seen: 2026-07-20
last_updated: 2026-07-20
status: active
---

What CFR (counterfactual regret minimization) is and why [[hoyt]] runs it,
written in tricks, worlds, and points rather than in the literature's
notation. The code this describes is `hoyt/cfr.py` (its docstring is the
honesty contract; this page is the intuition). Companion decoder for perf
vocabulary: [[hoyt-perf-primer]].

## The problem it solves

With all four hands face-up, backward induction settles everything — that
is [[the-oracle]], and it is why perfect-information 42 is "solved." Face
down, there is no single state to induct over: when you decide what to
lead, you are really deciding *at an information set* — every arrangement
of the hidden tiles consistent with what you have seen. Your one choice
plays out differently in every world, and the other seats are learning
about the worlds from your choice at the same time. CFR is the standard
way to find strong play in this setting without ever "solving for beliefs"
explicitly.

## The objects, at the table

- **History**: one world (who holds what) plus the public record of plays
  so far. All histories live in ONE shared tree — this is the wave
  engine's (public node × surviving world) structure ([[walt-spec]] §4).
- **Information set**: what a seat actually knows — (its own hand, the
  public record). One info set bundles every history it cannot tell
  apart. In code: the profile domain `(seat, hand_mask, node)`. An info
  set is a *perspective*; the tree is the *table*. Perspectives cannot
  fork the table: the record shows which tile hit it
  ([[endgame-equivalence-census]] — split pairs fight, co-held pairs
  speak).
- **Strategy (σ)**: at each info set, a probability over the legal moves.
  All four seats' strategies together are a *profile*.
- **Reach**: how likely a history is, given everyone's strategies and the
  world distribution. When my regrets are computed, my own choices are
  factored OUT (that is the "counterfactual"): each of my info sets is
  weighted by chance × *the other seats'* tendencies to bring that moment
  about — never by how often I choose to go there myself.
- **Counterfactual value**: at my info set, the expected points of a move
  = sum over the histories inside the info set of (others' reach × the
  subtree's value). This cross-seat product is the coupling that makes it
  a game — the one genuinely global computation, and the reason
  per-perspective tree compression is structurally impossible.
- **Regret**: after an iteration, "how much you wish you'd led the other
  tile," in points: regret(move) += value(move) − value(what σ did). It
  accumulates across iterations.
- **Regret matching (RM+)**: next iteration's σ at that info set is just
  the positive regrets, normalized. Never wished you'd played it → zero
  probability. RM+ additionally clips accumulated regret at zero (no
  hoarding of negative regret), which is why FORCED info sets (one legal
  move, ~83% at H4) are *exactly* inert — σ ≡ 1, regret ≡ 0 forever —
  and could be compressed out bitwise ([[perf-log]] 18l).
- **Average strategy**: the thing that converges is not the last iterate
  (iterates cycle) but the reach-weighted average of all iterates. That
  average is the exported profile.

## One trick, by hand

Last trick, fives are trump. You (declaring) hold 5-1; the defender holds
either 5-3 (world A) or 4-4 (world B), equally likely, and leads first…
skip the setup: suppose your two lines are worth, per world,

|        | world A | world B |
|--------|--------:|--------:|
| line X |     +2  |     −4  |
| line Y |     −2  |     +6  |

Uniform σ starts at 50/50: value = ½(½·2−½·2) + ½(−½·4+½·6) = +0.5 per
world-pair. Counterfactual values: X → ½(2−4) = −1, Y → ½(−2+6) = +2.
Regrets after iteration 1: X gets −1−0.5 = −1.5 → clipped to 0 (RM+);
Y gets +2−0.5 = +1.5. Next σ: all mass on Y. If the defender's strategy
were also adapting, their shift would change these numbers next round —
that feedback, iterated with averaging, is the whole algorithm. Nothing
else is happening at scale: `engine="loop"` is literally this recursion;
`wave`/`fused` are the same arithmetic vectorized over millions of (node,
world) slots, parity-gated bitwise.

## What convergence means here — the honesty line

The classic theorem (average regret → 0 ⟹ average profile → Nash) is a
TWO-PLAYER zero-sum theorem. 42 is two-TEAM zero-sum with private hands
inside teams, so hoyt claims no equilibrium — the deliverable is a
**low-exploitability reference priced by exact best response**: after (or
during) the run, `br_solve` computes, exactly, the most points any single
seat could steal by deviating while everyone else stays on the average
profile. That number is the **gap**, and *the measured gap is the claim*
(`hoyt/cfr.py` docstring; [[hoyt]]). Measured at H4: gap ≤ 0.05 points in
≤ 40 iterations at every root tried, scale-invariant in worlds.

Two things CFR is NOT doing, which surprise people:

- **No beliefs are ever computed.** Bayesian updating is implicit in the
  reach weights — worlds inconsistent with observed play get zero reach
  through legality, and "suspicious" play patterns get down-weighted
  through the opponents' σ. [[pimc]]-style explicit world-sampling and
  CFR's reach-weighting are two renderings of the same integral.
- **The last iterate is not the answer.** Iterates chase each other in
  circles; only the average settles. (Related measured fact: σ *among
  value-tied moves* is noise — RM+'s clip amplifies fp dust between
  provably tied actions, 4.5e-04 drift on an exact tie — so per-move
  VALUES, not σ mass, are the observable for ties;
  [[endgame-equivalence-census]].)

## Map to code

| concept | code |
|---|---|
| shared history tree | `hoyt/subgame.py` waves: (public node, world) slots |
| chance | the world distribution `build_subgame(root, worlds, weights)` |
| info set | `(seat, hand_mask, node)` — profile domain (`hoyt/CONTRACTS.md`) |
| payoffs | `payoff43[final declaring points]` — points/make/marks all one shape |
| RM+ iteration | `hoyt/cfr.py` engines: loop (definition), wave (numpy mirror), fused (numba, default) |
| forced-iset inertness | forced-slot compression, #82, bitwise no-op |
| the gap | in-struct exact BR ([[perf-log]] 18m), `br_solve` as pricing oracle |

## Reading

Neller & Lanctot, *An Introduction to Counterfactual Regret Minimization*
(2013) — the practical walkthrough; then Zinkevich et al., *Regret
Minimization in Games with Incomplete Information* (2007) — the original.
Adjacent landmarks this project has touched: Gilpin & Sandholm's
*GameShrink* (lossless abstraction — what `hoyt/equivcensus.py` computes
for 42, with a near-empty answer where poker gets orders of magnitude:
poker's suits are exchangeable, 42's are welded by the lead-max rule and
counts); DeepStack/ReBeL's public-belief-state factorization (the wave
engine's shape).

## Links

[[hoyt]] · [[walt]] · [[walt-spec]] · [[endgame-equivalence-census]] ·
[[the-oracle]] · [[pimc]] · [[perf-log]] · [[hoyt-perf-primer]]
