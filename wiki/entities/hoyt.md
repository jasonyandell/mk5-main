---
title: hoyt — the game's own referee
kind: entity
first_seen: 2026-07-18
last_updated: 2026-07-18
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
never to be edited (v2 may be added; v1 is forever).

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
  exact-BR gain vs the average, priced by `br_solve`. Two engines,
  bitwise-pinned (the verified slow loop is kept as the parity oracle for
  the vectorized wave engine). Verified on toys against scipy-LP /
  full-strategy enumeration.
- **Exploitability meter**: BR gain vs any frozen profile — counter-walt
  ([#77](https://github.com/jasonyandell/mk5-main/issues/77)) is this
  meter pointed at walt.

## Measured (2026-07-18, M5 Max; receipts in [[perf-log]])

| quantity | value |
|---|---|
| H4 BR re-solve after compile | p50 0.9 ms (45× net wavefront) |
| H5-cap512 BR | p50 3.5 ms; compile 0.35 s |
| CFR gap ≤0.05 pts | ≤40 iterations at EVERY root tried — **scale-invariant in worlds** (10 → 33,740) |
| CFR wall/root, cap-256 | median ~92 s, peak RSS 6.7 GiB after the columnar-profile perf day ([[perf-log]] 18g; was ~115 s / 7.4+ GiB) |
| stochastic-field tree blowup | p50 526×, max 5219× vs deterministic σ (measured, was argued ×10²–10⁴) |
| jud rent, 12 evalset roots | walt-BR-vs-jud beats the reference line by median **+1.9 pts/root** (range −0.08…+5.63) |
| mixing | ~500 toy configurations, zero mixed equilibria — vs deterministic fields, late 42 is **pure**; mixing must earn via concealment, not value |

Habitat: **H ≤ 4 comfortable** (H3 nearly free); H5 needs the queued
int32 narrowing + small caps; H6+ is a different build (streaming /
Metal / sampling — parked). Per tile of horizon: BR pays ~1 order of
magnitude uncapped and ~nothing capped; CFR pays ~1.5 orders in both
time and memory.

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
[[belief-policy-value-algebra]] · [[count-fate-ledger]] · [[the-wall]]
