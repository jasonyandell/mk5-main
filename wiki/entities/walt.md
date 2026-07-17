---
title: walt — the exact endgame information-set solver
kind: entity
first_seen: 2026-07-17
last_updated: 2026-07-17
status: active
---

## What it is

The base layer of the endgame→up ladder from the 2026-07-17 brainstorm
(Jason + Fable session): **exact best response in the information-set game**
at ≤4 tiles per seat, against a fixed known field (jud argmax), over
exactly-enumerated worlds. Code in `walt/` (tracked), issue
[#71](https://github.com/jasonyandell/mk5-main/issues/71) with registered
predictions. The name is walker→walt (renameable): the solver whose north-star
behavior is *seeing* [[texas-42]]'s walkers — and, if the ladder gets
licensed, learning to *create* them.

**A walker** (definition corrected at the table, 2026-07-17): a tile that is
**unbeatable when led** — a good lead in any world that matters — while
ranking as trash by the human beat-count heuristic. Its tactical value is
harvest (up to 25 riding count + the trick) and its emotional value is
surprise; it can be right to burn a good lead now to *create* a walker two
tricks later. This supersedes the "promoted trash" gloss on [[table42]].

## Where it sits in the algebra

Per [[belief-policy-value-algebra]]: eq = π deleted twice (CAN-#5). walt
un-deletes π from the **continuation** — one action per information set
across worlds, [[strategy-fusion]] killed by construction — and, in its W1
config, from the **belief**: with a deterministic field, e^g degenerates to
a 0/1 consistency filter, so exact B(σ) is computable with no learned head.
W0 (uniform u) vs W1 (filter) under the same solver is the gap
decomposition run as an ablation — term 2 alone vs terms 1+2 — the first
direct test of MIGHT-#3 (the play edge lives almost entirely in term 2).
Prior art: [[champion-design-review]] named exactly this door "the summit…
optional"; [[lamir1-ceiling]] is the scar that gates the ladder's next rung
(a distilled leaf must preserve argmax before any climbing — follow-up
issue after grading).

## The build (2026-07-17, all gates green)

- `walt/tables.py` + `walt/worlds.py` — rule LUTs **tabulated from
  `forge.oracle.tables`** (never reimplemented; parity-tested against zeb on
  2,200+ random states) and exact void-constrained world enumeration
  (brute-force cross-checked two independent ways; 34,650-world worst case
  enumerates in 6.4 ms).
- `walt/field.py` — jud as the field: vectorized featurizer **bit-identical**
  to `champion.jud_net.featurize_state` (2,240 pairs, 0 mismatch), 100%
  decision parity vs the real `JudPlay` (621 states), exact B(σ) filter,
  lazy incremental POV feature blocks, functional bit-identical net forward.
- `walt/solver.py` — the recursion, exactness-gated: **T2** equality vs
  explicit enumeration of *all* of the seat's pure info-set strategies, and
  **T6** claim-vs-cash closure — root value == realized playout mean at
  1e-9, the [[count-fate-ledger]] honesty loop closed internally.
- `walt/grade.py` + `walt/bench.py` — arena-paired grading (identical deal
  seeds, halves swap teams, bootstrap CI via `arena.match`) with per-decision
  records and walker-event instrumentation.

Perf (M5 Max, after a 3.6× surgery pass — 103.6 s → 29.1 s at 14,700
worlds): W1@H4 filter+solve p50 ≈ 1.0 s, p95 ≈ 10 s; W0@H3 p50 0.36 s; the
σ-filter costs ~90 ms and cuts u-worlds p50 14,700 → 376. Remaining wall is
the intrinsic node count (cross-node net batching is the next lever if ever
needed). Empirical note: σ-consistency barely discriminates on early
forced-ish moves and bites late — the coupling theorem's "forced actions
contribute zero," observed in filter survival curves.

## Grading

Pilot protocol + registered predictions (frozen before any run) are on
[#71](https://github.com/jasonyandell/mk5-main/issues/71): Run A =
walt(W1, H=4) vs jud at n=512 paired (P1 marks, P2 window points); Runs
B/C = W0 vs W1 at H=3 (P3, the term-decomposition bet); P4 walker rate.
Results land here when the runs finish.

## Links

[[belief-policy-value-algebra]] · [[strategy-fusion]] ·
[[expected-q-value]] · [[jud]] · [[table42]] · [[count-fate-ledger]] ·
[[champion-design-review]] · [[lamir1-ceiling]] · [[the-wall]]
