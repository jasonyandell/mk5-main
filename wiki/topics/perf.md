---
title: Perf — the measured laws of making this project fast
kind: topic
first_seen: 2026-07-18
last_updated: 2026-07-19
status: active
---

The umbrella for performance engineering across the project. Two full
campaigns have run — the 2026-04 Burl inference sprint ([[perf-on-the-table]])
and the 2026-07 walt solver rewrite ([[walt-spec]] §4) — on different
substrates (mlx-lm token generation vs numpy/torch tree search), and they
measured the *same laws*. This page carries the laws; the campaign pages
carry the receipts; the [[perf-sprint]] playbook family carries the process;
live work is tracked in the append-only [[perf-log]] (field notes, replaces
issues for perf items per 2026-07-18 directive).

## The laws (each measured at least twice)

1. **Calibrate against the substrate ceiling before touching anything.**
   Both campaigns opened by measuring the gap between what the silicon sells
   and what the harness pays: Gemma 4 E2B decodes 1,334 tok/s while the Burl
   harness ran ~70 ([[perf-on-the-table]]); walt's net forwards at ~1M
   rows/s at batch 8k while the solver fed it 4.6 rows per call and spent
   88 µs/node on bookkeeping. The ratio *is* the roadmap — it says whether
   the campaign is worth running and when it is done (walt stopped at ~55%
   GEMM: near the new ceiling).

2. **Batch the accelerator — "one call costs one tail; be bulk-synchronous."**
   The gomoku-derived call-cost law, now measured natively: walt's single
   biggest win was restructuring the tree walk into level-synchronous waves
   so the net sees thousands of rows per forward instead of 4.6
   (14.8× total, [[walt-spec]] §4). The same law priced Burl's levers
   ([[batch-throughput-bench]]: batch=128 is 31× single-stream).

3. **Contention is the master confounder.** The project has retracted or
   corrected perf wins at least three times after clean re-runs: continuous
   batching "1.8–2.1×" → statistical tie ([[perf-on-the-table]] lever 2,
   [[continuous-batching-dispatcher-design]]); the perf-subset "3.4× wall
   regression" → parallel-scribe contention (clean floor ±4%, detector:
   `decode_tok_s`); walt's pilot p95 measured under 12-worker load. Rule: no
   number is believed until it reproduces on a quiet box, and every bench
   records a contention detector alongside wall.

4. **Golden fixtures + parity gates, sized ≤10 min.** Any paired perf bench
   fits 10 minutes of wall (project rule since 2026-05); the fixture set is
   frozen, stratified, and committed (`walt/tests/fixtures_h4*.jsonl`, the
   Burl 5-row subset). For *exact* code the gate is binary: the wavefront
   engine had to reproduce identical argmax, values @1e-9, and identical
   node/query counts on all 46 fixtures before its 14.8× counted. Sub-ulp
   traps are real — torch's B=1 gemv differs from the gemm path by ~1 ulp,
   which can flip σ near-ties; `ev_rows` pads odd tails even, and the
   tree-count equality is the standing regression signal.

5. **Price approximations by decision damage, not surface metrics.** The
   world-cap probe found raw argmax-flip rate stuck at 12–25% at every K —
   but the flips are near-ties; *material* flips (|ΔV|>0.5 pt) collapse to
   0% at K≥512 ([[walt-spec]] §4). Same shape as Q4 quantization judged by
   paired play-match (5/5 identical) rather than tok/s
   ([[perf-on-the-table]] lever 6). Corollary: pick knobs by error budget
   (K=512 ≈ free, 6.1× on big roots), and expect the honest curve to kill
   the advertised one (#74's "10× at K=1–2k" was really 1.7–2.3×).

6. **Audit for dead weight; benchmark the code you think you run.** walt's
   solver memo provably never hit (history-unique nodes; measured 0/19,566)
   yet cost time and memory in exactly the monster solves. Worse: walt was
   a namespace package whose modules `sys.path.insert`-ed a disposable
   worktree — one process observed running *mixed* main+worktree module
   versions under the profiler. A perf session that skips this audit
   optimizes phantoms.

7. **Amdahl bookkeeping after every win.** Post-wavefront, solves no longer
   dominate grading (~1.14 s/game end-to-end, ≈4.9× not 14.8×) — the next
   lever, if ever needed, is the JudPlay tick loop, not the solver. The
   Burl stack's compounding estimate collapsed the same way once lever 2
   retracted. Track "share of wall" per component, or compound projections
   lie.

8. **Cap-and-ledger beats grind-inline** (measured in gomoku's VCT cascade
   — `~/code/gomoku/wiki/topics/vct-cascade-labeler.md` — and again here,
   [[perf-log]] 18h–j): solvers return result-or-cap as first-class
   verdicts, every item gets an explicit ledger row (no absence-as-state),
   and a budget ladder deepens only the shrinking survivor tail. Corollary
   measured twice in one day: **bandwidth-bound monsters barely
   parallelize** (8-wide big-root solves aggregate to ≈1.3× ONE quiet
   worker) — tune width per rung, never flat. Refined by the zero-diff
   width experiment ([[perf-log]] 19c): decontention is real (1.42×
   per-root at width 3 vs 5) yet **width beats decontention wherever
   decontention < width ratio** — on the M5 Max that is every stratum
   measured, so throughput (Little's law) keeps the flood and
   per-eval worker-seconds stays a pricing tool, not the objective.
   42's cap verdict is
   *quantified* (the exactly-priced gap at stop), so capped rows are
   usable references, and the hard stratum is predictable a priori from
   belief width. Registered metric ([[perf-log]] 18h): **H4 evals per
   wall-clock hour** — evals motivate, they don't steer.

9. **Eviction beats clear-all, and never mid-flight.** The FieldOracle memo
   once cleared itself wholesale at 4M entries — reachable *inside* a 6.2M
   query solve, silently re-forwarding everything. Cache eviction belongs at
   work-unit boundaries, sized by what actually transfers (walt memo keys
   embed full histories; they barely transfer across hands at all).

## Current state of the fast paths (2026-07-18)

- **walt H4 solve**: p50 ~5 ms, p95 193 ms in-game; 46-fixture suite 8.5 s;
  with `--world-cap 512` the fixture suite ≈ 2.3 s (≈54× vs the pilot
  engine). Bulk H4 corpus generation ≈ 10⁷ solves/day on 12 cores. Stage 3
  (Metal batch-VCT) parked until H5/H6 demands it ([#74](https://github.com/jasonyandell/mk5-main/issues/74)).
- **[[hoyt]]** (the net-free referee, promoted from `walt/kernel/`,
  [[perf-log]]): after a one-time σ-compile (≈ one net solve),
  deterministic BR re-solves at **0.9 ms p50** (45× the net wavefront;
  24.7 ns/node; payoff AND belief weights swappable free); H5-cap512 BR
  p50 3.5 ms. CFR+ reference profiles: gap ≤0.05 pts in ~40 iterations
  (scale-invariant in worlds so far); cap-256 anchor solve **7.2 s**
  after the fused iterate + in-struct gap pricing + resident-build +
  threaded-iterate levers ([[perf-log]] 18l/18m/18n/19a, #82: numba
  edge kernels + forced-slot compression, 7.1× iterate, bitwise; exact
  BR priced on the resident wave structure, 16×; the build stopped
  re-deriving what the walk already held — pslot/actor resident +
  numba move fill, 2.36× build; then the iterate went parallel while
  staying bitwise — build-time stable argsorts give every thread a
  disjoint output range with bincount-order folds, 2.67–3.12× iterate
  at 4–8 threads). P6 confirmed, P7/fp32 refuted-and-deleted, P8 16×,
  P9/P10 green, P11/P14/P16 narrowly refuted (quoted flat); the
  regret-bound gap shortcut is dead on theory (the 2p folk bound needs
  utility linear in one opponent); gap_exit prices intermediate
  measurements one seat at a time and exits at the first crossing
  ([[perf-log]] 18o — exact by construction, gate P5). Same-seed paired
  sweep at refsweep's threads=4 default: **23.7× worker-time per
  usable eval vs banked** (11.7 worker-s), projected rung-0
  ~3,400–4,650 evals/hour (was 224); **full line MEASURED
  2026-07-19: 200/200 converged in 17.0 min = 706 evals/hour, 38.1×
  banked, 14.8 worker-s/eval** ([[perf-log]] 19h; in-fleet rung mix
  197/2/1, the models retired), re-measured at **727 evals/hour
  (16.5 min)** after refsweep's static snake shards became one
  pull-based biggest-first claim queue — rung-0 pools balance to ±1 s,
  h2's pool 1.245×, line 1.03× quoted flat, per-seed reference values
  bit-identical; precision cost ordering (banked walls) measured
  WORSE than sigma order, monster co-residency contention
  ([[perf-log]] 19i) — the one wedge is 589.6M slots and
  2× memory-pressured, both measured
  ([[perf-log]] 19e); the pressure term is untouchable from the
  transient side — int32 walk working arrays landed as a 1.16× walk /
  1.07× build bandwidth win with ZERO churn effect, so the remaining
  pressure lever is resident footprint ([[perf-log]] 19f). The anchor's
  full reference line is built (`hoyt/reference_h4_v1_cap256.jsonl`,
  200/200 at gap ≤0.05) via the `hoyt/refsweep.py` cascade — law 8's
  shape ([[perf-log]] 18h–19a).
- **Burl inference**: no confirmed continuous-batching win; production
  picks are turn-aware token budgets + PLE-safe Q4 quant (memory, not
  wall). The sprint is dormant; resume via [[perf-sprint]].

## Links

[[perf-on-the-table]] · [[walt-spec]] · [[walt]] · [[perf-sprint]] ·
[[perf-sprint-levers]] · [[perf-sprint-traps]] ·
[[continuous-batching-dispatcher-design]] · [[batch-throughput-bench]] ·
[[batched-harvest-resilience]] · [[forge]] · [[jud]]
