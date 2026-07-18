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

## Graded (2026-07-17 night, n=512 paired each, protocol pre-registered on #71)

All three arms vs JudPlay, `margin:wp`(r8) bidder both teams, base_seed=0:

| arm | config | marks/game [95% CI] | game win | wall |
|---|---|---|---|---|
| A | W1 exact B(σ), H=4 | **+3.002 [+2.769, +3.219]** | 88.1% | 48 min |
| B | W0 uniform u, H=3 | +1.594 [+1.334, +1.861] | 70.1% | 4.3 min |
| C | W1 exact B(σ), H=3 | +1.877 [+1.623, +2.131] | 75.0% | 90 s |

- **Mechanism**: same bidder, same deals — make rate 66.3% when walt's team
  bids vs 37.5% when jud's (Run A). The ≤4-tile endgame flips ~29 points of
  contract percentage, both roles (halves symmetric; edge on offense AND
  defense). walt diverges from jud on 33.6% of endgame decisions.
- **P3 / MIGHT-#3 graded**: paired W1−W0 at H=3 = **+0.283 [+0.184,
  +0.391]** — term 2 carries **84.9%** of the edge. Synthesis with the
  champion-ladder #24/#25 nulls: belief-reweighting over *clairvoyant*
  values was marks-neutral, but the same beliefs acting through an *honest
  continuation* are worth +0.28 — **term 1 is only cashable through
  term 2**.
- **Prediction grades** (registered before the runs): P1 badly under-called
  (+0.05..+0.35 predicted, +3.00 actual — the "endgames are mostly forced"
  prior over-suppressed); P4 pass (walker led 0.191/hand vs ≥0.15 floor);
  perf target missed (in-run p95 1.9 s vs ≤300 ms). Jason's slots unfilled.
- **Scope caveat (the honest flag)**: opponents in these runs ARE the field
  walt models — the field-model term contributes its maximum. The served
  champion plays `lens:ev`, a different and stronger opponent. Licensed
  claim: exact term-2 + exact-belief play beats the 1-ply greedy head by
  ~3 marks/game at H=4 *when the field model is perfect*. The transfer test
  (walt vs `lens:ev`) measures how much was field-model rent — follow-up
  issue filed from #71.
- **Tail economics** (drives the throughput program): solve time mean
  624 ms vs median 5 ms; 7% of solves (>1 s) hold 91.5% of all solve time.
  The long solves are exactly the uninformative-history nodes (σ never
  bit), so a world-cap subsample hurts where beliefs are flattest.
  σ-filter survival is the coupling theorem observed: forced early moves
  discriminate nothing; the filter bites late.
- The B/C runs' decision logs carry full serialized roots + exact
  values/argmaxes — the seed corpus for the [[lamir1-ceiling]] scar probe
  (distilled-leaf argmax preservation), which gates the ladder.

## Links

[[belief-policy-value-algebra]] · [[strategy-fusion]] ·
[[expected-q-value]] · [[jud]] · [[table42]] · [[count-fate-ledger]] ·
[[champion-design-review]] · [[lamir1-ceiling]] · [[the-wall]]
