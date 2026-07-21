---
title: hoyt — the game's own referee
kind: entity
first_seen: 2026-07-18
last_updated: 2026-07-21
status: active
---

## What it is

The net-free exact-game instrument: a zero-torch solve substrate plus a
verified CFR+ layer that together price any strategy **against Texas 42
itself** rather than against whichever net the project trained last.
Code at `hoyt/` (promoted out of `walt/kernel/` the morning after it was
built — 2026-07-18, [[perf-log]] entries a–e). Named for *"according to
Hoyle"* — the family-table phrase for settling an argument by appeal to
the authority on games. Hoyle's authority was a book; hoyt's is computed.

Born as [[walt]]'s throughput program and promoted the moment its real
identity emerged: walt is a *player* (best response vs a modeled field);
hoyt is a *referee* (values and exploitability with no model in the
loop). The distinction is the same one [[walt-spec]] §7 drew in advance —
"equilibrium solver as instrument, never player" — built.

## Why it exists: the stable eval

Every walt number is denominated in "vs jud-v1's argmax" and dies when
[[jud]] retrains. hoyt's numbers are invariants of the game given a
utility config: the first play-quality yardstick that survives the
project's own model generations. "Exploitability in points/hand vs the H4
reference, on the frozen root set" means the same thing in 2027. The
frozen anchor is `hoyt/evalset_h4_v1.jsonl` — 200 stratified H4 roots,
never to be edited (v2 may be added; v1 is forever). Its reference line
is **built**: `hoyt/reference_h4_v1_cap256.jsonl` — all 200 roots at gap
≤ 0.05, produced by the `hoyt/refsweep.py` cascade ([[perf-log]] 18h–j).

## The machine

One generic SoA wave engine (forked from the [[walt-spec]] §4 wavefront)
with the σ-step behind pluggable providers: net / compiled table / rule /
uniform / dict-profile / full-width. Zero torch anywhere except
`compile_sigma`, which captures a net field's decision at every reachable
info set in about the cost of one ordinary solve — after which:

- **Exact best response** (`br_solve`): backward-pass-only for
  deterministic profiles — **0.9 ms p50** per H4 re-solve (45× the net
  wavefront, 24.7 ns/node), with the payoff table AND belief weights
  swappable per re-solve. Payoffs are `float64[43]` leaf tables on final
  declaring points — points, make/set, and marks-race utilities all
  compile to one shape.
- **CFR+ references** (`cfr_solve`): regret-matching+, alternating
  updates, linear averaging, over all four seats; gap = max single-seat
  exact-BR gain vs the average, priced by `br_solve`. Three engines,
  bitwise-pinned: "fused" (default — numba edge kernels over int32
  indices with the forced-slot-compressed strategy space, #82,
  [[perf-log]] 18l), "wave" (the pure-numpy mirror), "loop" (the
  verified recursive oracle). Forced info sets (~83% at H4) are exactly
  inert in RM+ (σ ≡ 1.0, regret ≡ 0), so compressing them out is a
  bitwise no-op. Gap pricing is IN-STRUCT ([[perf-log]] 18m): exact BR
  as a forward-reach + backward-argmax pass over the resident wave
  structure — no per-measurement export or re-walk; br_solve stays the
  pricing oracle via the loop engine and gate P4 (≤1e-9). The build
  keeps the walk's parent-slot/actor arrays resident instead of
  re-deriving them by key search, and the fused lane emits legal moves
  via `buildkernel.py` ([[perf-log]] 18n, output-identical; the numpy
  provider is the pinned mirror). Verified on
  toys against scipy-LP / full-strategy enumeration.
- **Exploitability meter**: BR gain vs any frozen profile — counter-walt
  ([#77](https://github.com/jasonyandell/mk5-main/issues/77)) is this
  meter pointed at walt.

## Measured (2026-07-18, M5 Max; receipts in [[perf-log]])

| quantity | value |
|---|---|
| H4 BR re-solve after compile | p50 0.9 ms (45× net wavefront) |
| H5-cap512 BR | p50 3.5 ms; compile 0.35 s |
| CFR gap ≤0.05 pts | ≤40 iterations at EVERY root tried — **scale-invariant in worlds** (10 → 33,740) |
| CFR wall/root, cap-256 | anchor **7.2 s** after the fused iterate + in-struct gap pricing + resident build + walk elisions + threaded iterate ([[perf-log]] 18l–19a: iterate 7.1× then 2.67–3.12× more at 4–8 threads (bitwise at any count — P13's disjoint-output structure), br 16×, build 2.7×; was 88.6 s at the start of 2026-07-18, ~115 s before that); build is the top bucket now (48% fleet-level at threads=4) |
| stochastic-field tree blowup | p50 526×, max 5219× vs deterministic σ (measured, was argued ×10²–10⁴) |
| jud rent, ALL 200 evalset roots | median **+1.53 pts/root** (mean +2.02, p90 +5.22, range **−8.65…+15.49**; the 12-root "never negative" died — 18 roots < −0.5, the population term cuts both ways) |
| refsweep cascade velocity | rung-0 **224 evals/hour** pre-kernel; post 18l–19a (threads=4 default): same-seed paired sample **23.7× less worker-time per usable eval** (11.7 worker-s), 19/20 converge at rung 0 and the 555090 wedge now cashes at rung 2 (473 s full-cascade root-wall; banked: 1,799 s) → **~3,400–4,650/hour projected** ([[perf-log]] 19a); deepening 87% → 99% → 100% over three rungs pre-kernel |
| mixing | ~500 toy configurations, zero mixed equilibria — vs deterministic fields, late 42 is **pure**; mixing must earn via concealment, not value |
| exact equivalence ([[endgame-equivalence-census]]) | world/root compression exactly 1.000× (co-occurrence theorem — dead at every horizon); within-hand interchangeable pairs in 25/200 roots, 29/29 bitwise value-tie receipts via `hoyt/equivcensus.py` |

Habitat: **H ≤ 4 comfortable** (H3 nearly free); H5 needs a measured
population baseline plus a new execution lane; H6+ is a different build.
[[full-metal-hoyt]] states the active structural question: build a fast,
device-resident, root-batched Metal lane that makes H5 and H6 reference lines
practical with calibrated error bars. Bit replication is not part of that
goal. Per tile of horizon: BR pays ~1 order of magnitude uncapped and ~nothing
capped; CFR pays ~1.5 orders in both time and memory.

## The honesty line (registered before building)

Texas 42 is two-team zero-sum with private hands; CFR's convergence
guarantee is a two-player theorem and the partnership seam (TMECor-style
correlation) is real. hoyt therefore never claims "equilibrium": it
claims **a low-exploitability reference priced by exact single-seat best
response** (deviator alone; partner stays on profile). Team-pair
deviation is explicitly out of scope for v1. Equilibrium is also the
wrong *personality* for the table — hoyt is an instrument, never a seat.

## Consumers (per the distill-for-what rule)

Exploitability pricing for [#77](https://github.com/jasonyandell/mk5-main/issues/77)
(counter-walt, ε-tie mixing reference — the measured pure-equilibrium
fact says concealment is mixing's only paycheck); the stable eval for
every future player generation; reference values for the
[[count-fate-ledger]] referees; rent-decomposition of walt's edge (the
per-root form of [#72](https://github.com/jasonyandell/mk5-main/issues/72)).

## Links

[[walt]] · [[walt-spec]] · [[perf]] · [[perf-log]] · [[jud]] ·
[[belief-policy-value-algebra]] · [[count-fate-ledger]] · [[the-wall]] ·
[[cfr-primer]] · [[endgame-equivalence-census]] · [[full-metal-hoyt]]
