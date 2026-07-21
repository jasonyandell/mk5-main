# metal_hoyt — the GPU searcher under the fp64 referee

Registered 2026-07-21. A NEW ENGINE, not a port: [[hoyt]]'s CFR+ iterate and
in-struct gap pricing rebuilt as raw Metal (MSL) kernels on the M5 Max GPU
(40 cores, Metal 4), dispatched through `mx.fast.metal_kernel` — zero new
dependencies, unified-memory buffers, kernels written in Metal Shading
Language and JIT-compiled at import.

## Why now (the measured license)

The CPU line at H4-cap-256 is at **measured exhaustion**: 727 evals/hour,
15.1 worker-s/eval, with the scheduling family, width, dtype, and fusion
levers all priced and closed (perf-log 18l–19i). The fused iterate is
gather-latency-bound on CPU; the M5 Max GPU hides exactly that latency with
thread parallelism. Measured before building (2026-07-21): a 3M-edge
segmented backward-value pass — the iterate's canonical fold, random
gathers included — runs in **0.41 ms/dispatch at ~127 GB/s effective**
with an eval per dispatch, ~3–4× the tuned numba lane's per-pass
throughput before any dispatch amortization. Perf is a project enabler for
the next six months of experiments (bigger caps, H5/H6, counter-walt
exploitability sweeps, class-CFR), not a luxury.

## The division of labor (the whole design in one table)

| stage | engine | precision | why |
|---|---|---|---|
| structure build (walk, layouts) | hoyt CPU (`_build_wave` + `_build_fused(par=True)`) | exact ints | verified once; P13's grouping structures ARE GPU segment layouts |
| CFR+ iterate (fwd reach, bwd values, cf, RM+) | **Metal GPU** | fp32 | the wall; latency-hiding silicon |
| intermediate gap (continue/stop only) | **Metal GPU** | fp32 | steering signal, never a claim |
| final gap + reference value | hoyt CPU (`_wave_br`, `_wave_values`) | **fp64 exact** | every banked claim is certified by the standing oracle |
| profile export | hoyt (`StochasticProfile.set_bulk`) | fp64 | same artifact class as hoyt |

hoyt stays the referee; metal_hoyt is the searcher. A metal solve's verdict
is *"profile found on GPU, gap priced by exact fp64 best response"* — the
same claim class as a hoyt solve. What metal_hoyt does NOT claim: bitwise
parity with hoyt's fp64 trajectory (impossible — Metal has no fp64), and
therefore interchangeability with hoyt-banked bit-reference artifacts
(`reference_h4_v1_cap256.jsonl` stays hoyt-produced; the equivalence-census
29/29 bitwise-tie receipts stay hoyt-only instruments).

## The fp32 license (bar arithmetic, registered before building)

The reference bar is gap ≤ 0.05 points. The measured fp32 drift on the CPU
experiment was 2.6e-3 (perf-log 18l, P7); the measured GPU steering-gap
drift is larger — worst observed 8.5e-3 (evalset 555002 at cap-64; the
mechanism is near-tied BR argmax flips, not accumulation) — still 6×
under the bar. The GPU exit test uses a margin: iterate until the fp32
gap ≤ target − margin (default 0.01, covering the observed drift), then
certify in fp64; if certification fails, keep iterating. A wrong fp32 gap
can only cost wasted iterations, never a wrong claim.
Weights are normalized on the GPU (w0/total_w) to keep fp32 magnitudes
tame; RM+ is scale-invariant to this, and all reported values come from
the fp64 certification pass.

## Determinism without atomics

Every GPU fold accumulates sequentially inside a thread over a precomputed
segment (the same `vseg`/`cf_ju`/`iso` groupings P13 built for the numba
lane — stable argsorts at build time, never runtime heuristics). No
atomics anywhere, so a metal solve is bitwise-reproducible in fp32 across
runs on the same GPU. Cross-device reproducibility is not claimed.

## Kernels (metal_hoyt/kernels.py, MSL source)

- `fwd_pass` — map over edges: rmu/ru child from parent × σ, updating-seat
  edges pass reach through (mirror of `fwd_edges_par`).
- `bwd_v` — thread per parent slot, sequential fold over its edge segment
  (mirror of `bwd_v_seg`); doubles as the BR backward when given a chosen
  mask and the deviating seat (u = −1 disables the BR path).
- `cf_seg` — thread per strategy slot, folds r_mu·v over the slot's edge
  group (mirror of `cf_seg`); doubles as the BR score kernel.
- `rm_update` — thread per info set: cfv fold, linear-average accumulate,
  RM+ clamp, σ refresh (mirror of `rm_update_seg`).
- `br_choose` — thread per info set: first-max signed argmax over the
  iset's slots → 0/1 chosen mask (mirror of `_wave_br`'s per-iset argmax,
  lowest-move tie-break).
- `asig_norm` — thread per info set: normalized average strategy (uniform
  where never reached).

Scalars (seat, sign, iteration, block offsets) ride in 1-element input
arrays, not template args, so each kernel compiles exactly once.

Per-seat state is kept as separate buffers (sig/reg/avg per seat, ordered
by the compressed layout's seat-major order) and the global σ used by the
walk kernels is a 4-way concat — this keeps RM+ updates purely functional
(MLX arrays are immutable) without scattering into a shared buffer.

## Contiguity invariants (asserted at layout build, not assumed)

The compressed slot space is seat-major, wave-major within seat
(`_build_fused`), which makes three facts load-bearing and checkable:
per (wave j, seat u), the strategy slots touched by wave-j edges form a
CONTIGUOUS run; per (j, u), the compressed iset ids form a contiguous run;
concatenating the per-wave runs in wave order tiles the seat's slice
exactly. `layout.py` asserts all three per root; a violation is a build
bug, never something to paper over.

## Batching and the line

v1 solves one root at a time on the GPU: at cap-256 scale a single root's
big waves already saturate the GPU, and MLX dispatch overhead measured
~10 µs amortizes to ~0.1 s/solve. The production driver (`line.py`)
overlaps the CPU build (the fleet's top bucket, 48% at threads=4) with GPU
iteration via a builder prefetch pool — the GPU frees the CPU cores that
used to run the iterate, so the line's Amdahl limit moves to the build.
Cross-root batching inside single dispatches (the next multiplier for
small-root fleets) is filed follow-up work, not v1.

## Gates

- **M1 toys**: metal solves on the toy family converge (fp32 gap → 0) and
  match the reference engine's value within ε-equilibrium tolerance.
- **M2 kernel parity**: every MSL kernel against a same-fold-order mirror
  on randomized structures. The map kernel (`fwd_pass`) is bitwise fp32;
  the fold kernels are NOT bitwise against a separate-mul-add mirror —
  Metal contracts mul+add to FMA (measured 1.8e-7 max rel vs the fp64
  same-order fold, i.e. FMA is the *more* accurate rounding) — so their
  gate is ulp-level agreement with the fp64 fold plus exact
  run-to-run repeatability (the determinism claim).
- **M3 certification**: evalset roots converge under the metal engine and
  the fp64-certified gap meets the bar; certified values sit within
  ε-equilibrium distance of hoyt's banked references.
- **M4 honesty**: fp32 gap vs fp64 certified gap deltas are logged; the
  margin covers the observed drift.
- **M5 speed**: paired same-session bench vs hoyt fused threads=4
  (`bench.py`, ≤10 min wall), reporting per-root walls and the projected
  line rate in H4 evals/hour.

## Failure doctrine

Metal-unavailable is a hard error (no CPU fallback — hoyt IS the CPU
lane). KernelMemoryError semantics are inherited from the CPU walk
(build-side). GPU numerical failure (fp32 gap refuses to reach
target − margin while fp64 says above target) caps the solve into a
`gap_capped`-class verdict carrying the certified gap — a quantified
verdict, never a silent retry.
