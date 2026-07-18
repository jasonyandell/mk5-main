---
title: The walt spec — exact endgame info-set solving, and what it becomes
kind: topic
first_seen: 2026-07-17
last_updated: 2026-07-18
status: active
---

Dense technical spec of [[walt]], written 2026-07-17 the night it was built and
graded. The entity page carries the story; this page carries the machine.

## 1. The object

At an information set s of seat m (own hand x, public history h) with ≤ H
tiles per seat remaining, walt computes the **exact best response in the
information-set game against a fixed deterministic field σ**:

```
ρ* = argmax_ρ  E_{ω~b(·|s)} [ U( terminal(ω; ρ, σ) ) ]
```

ρ ranges over strategies that assign ONE action per information set of m —
consistency across worlds is structural, so [[strategy-fusion]] is dead by
construction, not by penalty. The other three seats (partner included) play
σ. U configs: declaring-team points (JudPlay-comparable), make/set step at
the bid line; marks-race utility ([[jud]]'s `MarksToSeven`) is the planned
third (see §7).

## 2. Location in the algebra

Per [[belief-policy-value-algebra]]: eq = E_u[Q°] is the honest object with
π deleted twice (CAN-#5) — from the belief (B(π) → u) and from the
continuation (π → clairvoyant Q°). walt un-deletes both:

- **Continuation**: honest info-set-consistent play vs σ — term 2 of the
  gap decomposition, the insurance economy, computed natively for the
  first time in this project.
- **Belief**: for a *deterministic* field, the tilt b = u·e^g degenerates —
  g(ω) ∈ {0, −∞} — so exact B(σ) is a **0/1 consistency filter**: keep
  worlds where σ reproduces every observed non-m action. No learned head.
  Sound only when observed seats actually play σ; under field
  misspecification the filter mass-extincts and falls back to (capped) u.

W0 (uniform u) vs W1 (filter) under the same solver is the gap
decomposition run as an ablation. **Graded 2026-07-17**: term 2 carries
84.9% of the H3 edge; the belief kicker is +0.283 [+0.184, +0.391]
marks/game — the first belief effect in project history whose CI excludes
zero at the table. Synthesis with the [[champion-ladder]] #24/#25 nulls:
**term 1 is only cashable through term 2** — beliefs over clairvoyant
values were marks-neutral; the same beliefs through honest continuations
pay.

Empirical corollary of the coupling theorem, observed in filter survival
curves: forced actions discriminate nothing; σ-consistency bites late.
Median worlds after filtering in live play: **3** (from u-p50 14,700).

## 3. Design

Modules (`walt/`, all gates green before any number was believed):

- `tables.py` — rule LUTs **tabulated from `forge.oracle.tables`**, never
  reimplemented (led-suit, follow, rank, count, beat-count; parity vs zeb
  on 2,200+ random states). Hands are 28-bit masks; trick resolution is
  LUT gathers.
- `worlds.py` — exact enumeration of hidden-tile assignments: multinomial
  (3H)!/(H!)³ worst case (H=4: 34,650; enumerates in 6.4 ms), cut by
  played-tile conservation and void constraints (a failure-to-follow
  constrains the hand *at that moment*). Brute-force cross-checked two
  independent ways.
- `field.py` — σ = jud argmax: vectorized featurizer **bit-identical** to
  `champion.jud_net.featurize_state` (torch.equal on 2,240 pairs), 100%
  decision parity vs `arena.jud_play.JudPlay` (621 states), functional
  forward (same ops, no Module dispatch), `NodeCtx` (lazy incremental
  4-POV play blocks, O(1) advance, materialize-on-miss), `decisions_at`
  (per-node batching, unique-hand dedup, shared memo), `sigma_consistent`
  (the e^g filter).
- `solver.py` — the recursion: at m's nodes, argmax sign·value over legal
  moves (sign fixed at root); at σ nodes, one batched decision call over
  alive worlds, partition by returned move (observation branches carry the
  belief update — partner-reading is the partition). Memo on
  (my_hand_mask, history). Leaves score via LUTs to zeb-exact points.
- `grade.py`/`bench.py` — arena-paired grading (identical deal seeds,
  halves swap teams, bootstrap CI via `arena.match`), per-decision records
  with **full serialized roots** (the distillation corpus accumulates as a
  side effect of grading), walker instrumentation, hand-stream prelims.
  Corpus caveat: root serialization landed AFTER pilot arm A launched —
  **A (H4, the +3.00 arm) banked no roots**; only B/C (H3) did. The H4
  corpus for the scar probe (#73) must come from fresh grading (cheap now).

**Correctness gates**: T1 known-world degeneracy; T2 exactness vs explicit
enumeration of ALL pure info-set strategies; T3 dominance; T4 weight-split
invariance; T5 rules parity; T6 claim-vs-cash closure — root value equals
realized playout mean at 1e-9, the [[count-fate-ledger]] honesty loop
closed inside the solver.

**The load-bearing structural fact: determinism of σ is the compression.**
One path per world (partition), vs full-width opponent branching (tree).
Stochastic or unmodeled fields cost ×10²–10⁴ at H=4 and change the problem
class (equilibrium, CFR-shaped, all seats' info sets live). Field
*accuracy* is iterable (field ← distilled student, re-solve); field
*determinism* is not optional. Beliefs, by contrast, are droppable at a
measured cost and purchasable back (§6).

## 4. Performance (M5 Max, 2026-07-18: the wavefront engine)

2026-07-18 the recursion was replaced by the **wavefront engine**
([#74](https://github.com/jasonyandell/mk5-main/issues/74) stage 2, the
gomoku-derived wavefront-with-net design): level-synchronous waves, all
node/world-slot state struct-of-arrays numpy, per-wave dedup of σ
decisions, ONE chunked forward per wave (8k-row chunks; torch cpu gemm
~1M rows/s there vs ~65k rows/s at the old avg batch of 4.6), bincount /
segmented-reduceat backward pass. **Exact parity** with the recursion on
46 stratified golden fixtures (`walt/tests/fixtures_h4.jsonl`): identical
best_move, value to 1e-9 (bitwise where inspected), identical n_nodes AND
n_field_queries. Diagnosis that drove it: the old solve was ~all
overhead — 88 µs/node, ~2 worlds per oracle call — and its (hand,
history) memo provably never hit (history is unique per tree node;
measured 0/19,566). Subtlety to respect forever: torch's B=1 gemv kernel
is ~1-ulp off the gemm path, so `ev_rows` pads odd tails even; the
fixture tree-count equality is the regression signal for any rebatching.

| quantity | measured |
|---|---|
| 46-fixture bench (sigma@H4) | 125.4 s → **8.5 s** (14.8×; worst 34.6 s → 2.2 s) |
| in-game grading solves (n=32 smoke) | p50 4.8 ms, p95 **193 ms**, max 4.0 s (was p95 1.9 s, max 153 s) |
| end-to-end grading wall | ~1.14 s/game vs 5.6 (≈4.9×; solves no longer dominate) |
| world-cap K=512 on the engine | **6.1×** more on cap-affected (≥500-world) roots |
| remaining wall | ~55% GEMM (near-irreducible without breaking decision parity) |

**World-cap (stage 1), measured honestly** (2,485 capped-vs-exact solves):
the issue's "~10× at K=1–2k" was wrong — K=1–2k bought 1.7–2.3× on the
old engine. Error curve: p90 |ΔV| 0.45 pts @K=512, 0.58 @K=256, bias ~0.
Raw argmax flips run 12–25% at every K but are **near-ties**: material
flips (|ΔV|>0.5 pt) go 10.4% (K=64) → 1.3% (K=256) → **0% (K≥512)** —
raw flip rate is the wrong acceptance gate. `walt.grade --world-cap 512`
(deterministic per-root subsample, default off) is the endorsed knob;
K≈64–100 would buy ~10× alone at real decision damage. The breadth of the
near-tie band is independent evidence for #77's ε-tie mixing headroom.
Stage 3 (Metal/batch-VCT kernels) stays parked unless H5/H6 bulk needs it.

## 5. Graded (pilot, n=512 paired each, predictions pre-registered)

vs JudPlay, `margin:wp`(r8) bidder both teams:

| arm | config | marks/game [95%] | game win |
|---|---|---|---|
| A | W1, H=4 | **+3.002 [+2.769, +3.219]** | 88.1% |
| B | W0, H=3 | +1.594 [+1.334, +1.861] | 70.1% |
| C | W1, H=3 | +1.877 [+1.623, +2.131] | 75.0% |

Mechanism: make-rate flip 66.3% vs 37.5%, both roles, halves symmetric;
33.6% decision divergence from jud; walker led 0.191/hand. **Scope**: the
opponents ARE the modeled field — +3.00 includes maximal field-model rent.
The transfer test ([#72](https://github.com/jasonyandell/mk5-main/issues/72),
walt-under-lens:ev-everywhere-else) is the first direct head-to-head of
eq's two deletions vs zero deletions at a fixed horizon. Prior registered
in-conversation: the edge shrinks a lot, stays positive, defense-heavier.

## 6. The fitted belief net

The filter is exact only against the modeled deterministic field. Against
everything else — humans, lens:ev, future students — beliefs return as a
**fitted amortization of e^g**: a net trained supervised on (info set →
world weights) from self-play replays under the *current* field, playing
the role [[gus]]'s head was always meant to play, now with a consumer that
provably cashes belief (term 1 through term 2, §2). Design commitments
(registered 2026-07-17, [#75](https://github.com/jasonyandell/mk5-main/issues/75)):

- **Beliefs do NOT get the RL gradient.** Policy explores; beliefs track
  by supervised amortization; otherwise the loop can converge to the
  internally-consistent, externally-nonsense fixed point ([[jud]] #66).
- **u-replay anchors audit every rung** (CAN-#4: V-error is repairable by
  evaluation data alone; a replay probe costs seconds).
- Two-timescale scheduling against the measured tiger (MIGHT-#6).
- At low-history roots (openings) the net IS the belief — the filter has
  nothing to bite; "few important worlds" = importance sampling by the
  fitted e^g over the [[candlewax]] bumps.

## 7. Trajectory

Gated sequence, cheap probes first:

1. **Scar probe** ([#73](https://github.com/jasonyandell/mk5-main/issues/73))
   — the [[lamir1-ceiling]] warning (distilled leaves flip argmax) gates
   the ladder. walt's targets differ from LAMIR's in every suspected
   cause: honest info-set values (not clairvoyant marginals),
   belief-conditioned (not world-blind), rollout IS the solve. Corpus
   already banked from grading logs.
2. **The ladder**: distill rung k → leaf for a one-trick solve at rung
   k+1; per-rung cost ~constant instead of ×100/tile. H4 in-game now; H5
   in-game post world-cap; H6 analysis-only; trick 1 belongs to the
   ladder + fitted beliefs, never to exact enumeration (399M worlds).
3. **Utility, not personality**: marks-race U at the leaf makes
   "go-for-it" emerge where variance is genuinely free (down 5-1, the 30%
   game-winning line beats the 60% hand-winning line). An afternoon; do
   before any personality work.
4. **Exploits and countermeasures**
   ([#77](https://github.com/jasonyandell/mk5-main/issues/77)): walt is a
   deterministic best response — maximally readable (hand-bleed audit:
   point `sigma_consistent` at walt; hypothesis: optimality IS
   legibility, better decisions discriminate more) and maximally
   exploitable (counter-walt, cheap post-distill). Countermeasures:
   ε-tie mixing (CAN-#7 dual — concealment free on ties), the fixed-point
   loop, marks-race utility.
5. **Opening net + bidding loop-in**
   ([#75](https://github.com/jasonyandell/mk5-main/issues/75)): the solver
   is the new oracle in the right game; a bid head distilled from the
   player that cashes the bids makes claim-vs-cash structural. Openings
   are value-smooth but convention-dense (free channel): value comes from
   distillation, tie-breaks from RLAIF critics — the [[w42]] claim battery
   (64 detectors) and the family-idiom judge — MIGHT-#5's
   equilibrium-selection role for the book, the small-ε member of the
   (ε, init) family.
6. **Equilibrium solver as instrument, never player — BUILT 2026-07-18
   and promoted to its own entity: [[hoyt]]** (`hoyt/`, [[perf-log]]
   2026-07-18a–f). The net-free kernel + verified CFR+ references + the
   exploitability meter + the frozen eval anchor now live there; hoyt
   carries the numbers, the honesty line, and the habitat table. The
   division of labor: walt is a player (best response vs a modeled
   field); hoyt is the referee (values with no model in the loop).

## 8. Edges (measured or argued, so nobody re-derives them)

- No beliefs: ×~100 at H4, cappable, effect −0.283 marks at H3. Cheap.
- World-cap: a 3–6× knob, never a 10× one (§4). Choose K by error budget
  (material-flip rate, not raw flips); K=512 is free, K=256 near-free.
- No deterministic field: class change, not slowdown. Not repairable by
  throughput (×1600 kernels don't cover ×10⁴/tile compounding).
- W1 under misspecification: filter extincts → capped-u fallback (already
  wired). Corpus generation should use capped-u anyway (coverage control).
- Grading is lockstep-batched: no early outcome signal by design;
  hand-stream + chunked-lockstep are the prelim answers.

## Links

[[walt]] · [[belief-policy-value-algebra]] · [[strategy-fusion]] ·
[[expected-q-value]] · [[jud]] · [[gus]] · [[count-fate-ledger]] ·
[[lamir1-ceiling]] · [[champion-design-review]] · [[w42]] · [[the-wall]] ·
[[candlewax]] · [[the-wall-biography]] · [[perf]]
