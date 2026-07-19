---
title: hoyt perf primer — the shorthand decoded, start to current
kind: topic
first_seen: 2026-07-19
last_updated: 2026-07-19
status: active
---

Plain-language decoder for the [[hoyt]] perf campaign's vocabulary, with
the runtime characteristics at each milestone. Written because the
lineage ([[perf-log]] 18a–19i) developed dense shorthand that resists
cold reading. Receipts live in [[perf-log]]; laws live in [[perf]]; this
page only translates. If a term here drifts from the log, the log wins.

## The objective, in game units

A [[texas-42]] hand has 7 tricks. **H4 = horizon 4 = positions where
every player holds exactly 4 dominoes** (the last 4 tricks). Each
additional trick of horizon multiplies the tree ~100× ([[walt-spec]]).
Jason's directive (perf-log 18h): optimize **4th-play evals per
wall-clock hour** — how many 4-tricks-left positions get exactly solved
per hour. **One eval** = one root taken to a CFR+ reference strategy
*plus its certificate*.

## Glossary, in birth order

- **root** — one frozen H4 position (turn, play history, hero's tiles,
  score).
- **the anchor** — `hoyt/evalset_h4_v1.jsonl`: 200 stratified roots,
  frozen forever, so numbers stay comparable across years. The log also
  uses "the anchor solve" for the single paired-bench root 555006 — two
  senses, same word.
- **worlds** — consistent deals of the three hidden hands (10 to
  ~34,000 per root). **σ-consistent**: filtered to deals where the
  net's decisions match the actual play. **σ** is a strategy; the net's
  is captured once per root by `compile_sigma` (≈ one net solve), after
  which everything is net-free.
- **cap 256** — solve over ≤256 sampled worlds. Priced: cap 512 ≈ 2×
  cost; value errors >0.5 pts ("material flips") are 1.3% @256, 0%
  @512+. Raw argmax flips are near-ties and the wrong gate.
- **BR** — exact single-seat best response ("if this one player
  deviated perfectly, how many points would they gain").
- **gap** — the certificate: max any single seat could gain (pts/hand)
  vs the average strategy, priced by exact BR. **gap ≤ 0.05 =
  converged.** (Registered honesty: CFR's Nash theorem is 2-player;
  the claim is "low-exploitability reference," not equilibrium.)
- **rent** — walt-BR-vs-[[jud]] value minus reference value at the same
  root: the jud-specific share of [[walt]]'s edge. Median +1.5…+1.9
  pts/root; population −8.65…+15.49.
- **banked** — the first complete reference line (perf-log 18j):
  200/200 at gap ≤0.05 in 10.8 wall-hours ≈ **18.5 evals/hr** (rung-0
  stratum 224/hr). The permanent denominator for every later speedup.
- **rungs / cascade / verdicts** — `hoyt/refsweep.py`'s budget ladder
  (rung 0: 90 s wall + 32M slots → rung 1: 600 s + 64M → rung 2:
  unlimited + 128M). Every (root, rung) attempt is a ledger row with a
  first-class verdict: `converged` / `gap_capped` (usable at its
  measured gap) / `slot_capped` / `error`. **slots** = strategy-table
  entries, the memory currency.
- **the wedge** — root 555090: 589.6M tree slots (~5× the next
  biggest), needs rung 2 and ~27 GiB solo, converged at gap exactly
  0.0. Tree size ⊥ decision difficulty. Since 19g there is exactly one
  wedge (555212's wedge-hood dissolved under speed).
- **build / iterate / br / export** — the solve's internal anatomy:
  lay out the wave-tree arrays; CFR+ update sweeps; gap pricing;
  profile export.
- **w5 t4** — production fleet shape: 5 worker processes × 4 numba
  threads. **paired sweep** — same seeds re-run same-session and
  compared per-seed (big-root single runs swing ±25% with ambient
  state; unpaired cross-day numbers are meaningless).
- **priors / bars / quoted flat** — predictions registered with
  numeric pass bars *before* measuring; refutations stay in the log at
  full volume ("PASS by 0.9 s — quote it as 1.11×, not a triumph").
- **worker-s/eval** — worker-seconds per delivered eval; a pricing
  tool, explicitly NOT the objective (19c: width beats decontention;
  throughput is the objective).
- **churn / pressure** — macOS memory-compressor traffic on the
  48 GiB box; measured to be run-ordering-determined, not
  code-determined (19f).
- **the snake → the pull queue** — work assignment: static
  interleaved-and-rotated per-worker queues (18h) replaced by one
  shared biggest-first queue workers pull from via atomic claim files
  (19i). Balance comes from the pull, not from cost-model precision —
  exact cost ordering LOST to blunt size ordering by front-loading
  monsters into simultaneous memory pressure.

## Runtime characteristics, milestone by milestone

hoyt 1.0 (perf-log 18b–18e): σ-compile ≈ 1 net solve, then BR re-solves
**0.9 ms p50** (45× the net engine; payoff/belief weights swappable);
H5-cap512 BR 3.5 ms. First CFR: 167 s at eight worlds
(python-recursion-bound); vectorized onto the wave tree → 4.6 s same
root, ~2 min and ~7.4 GiB per typical cap-256 root, gap ≤0.05 in ≤40
iterations regardless of world count. Naive 8-wide sweep: ~45 evals/hr,
killed by bandwidth contention at 43/200.

Benchmark-root solve wall through PR #83's levers:

| entry | lever | anchor-root solve |
|---|---|---|
| 18g (PR #81) | columnar profile: RSS hog was python objects (10.6→6.7 GiB → 5 workers fit) | 120.9 → 92 s |
| 18l | fused iterate: forced-slot compression (83% of slots provably inert) + numba kernels, bitwise | 88.6 → 41.0 s |
| 18m | in-struct gap pricing (Fable-credited: "stop walking"), br 16× | 41.0 → 15.9 s |
| 18n | resident build: stop re-deriving what the walk held | 15.9 → 13.0 s |
| 18o | gap_exit: intermediates only answer continue-vs-stop | (fleet win) |
| 18p | walk elisions | 13.0 → 12.6 s |
| 19a | threaded iterate, bitwise at any thread count; threads=4 | 12.6 → **7.2 s** |

Fleet cost per usable eval: 277 (banked) → 19.5 (18m) → 16.4 (18n) →
14.7 (18o) → 11.7 worker-s (19a, 20-seed subset) → **15.1 worker-s
full-population** (19i; the subset was light by ~26%).

Refutation nights (19b–19f), all kept: solo wins compress ~4:1 at
fleet; width beats decontention every stratum; numpy int argsort is
already radix (three levers died on it); int32 transients kept as
bandwidth only (churn theory dead); the banked 174/24/2 rung mix was a
slow-era artifact — post-speedup in-fleet mix is **197/2/1**.

The line, measured: **19h — 200/200 converged in 17.0 min = 706
evals/hr (38.1× banked)**, models retired (v1 was right by luck), the
**measured-exhaustion receipt** for H4-cap-256 micro-perf. **19i —
snake → pull dispatch, re-measured 990 s = 727 evals/hr (39.3×
banked)**, all 200 reference values bit-identical, line-level 1.03×
quoted flat.

## Current state and the fork

Today: 200 H4 evals in 16.5 min (**727/hr**, 15.1 worker-s/eval,
~7 GiB/rung-0 worker, one 27 GiB wedge running solo). Wall shape:
rung-0 pool ~62% (balanced ±1 s), wedge ~20% (capped verdict),
barriers/tails the rest — every bucket has a closed receipt. The
remaining moves are structural, Jason's to pick: **horizon** (H5/H6
line — BR measured cheap, CFR line unpriced), **cap policy** (256→512
≈ 2× for material-flip zero; a cap-512 line is now ~45 min), or **the
consumer** (lens:ev grading + a distilled student from reference
values — the [[jud]]-side cash-out the distill-for-what rule names).

## Links

[[perf-log]] · [[perf]] · [[hoyt]] · [[walt]] · [[walt-spec]] ·
[[jud]] · [[texas-42]]
