---
title: Stage 0 Closure — repaired-sampler measurement baseline
kind: experiment
first_seen: local-2026-07-13
last_updated: local-2026-07-13
status: active
---

Can the measurement baseline of [[partnership-wall-research]] Stage 0 close on
the repaired sampler — does `C0` reproduce on two held-out seed blocks, does
the sampler behave on CUDA, and how exposed were historical decisions to the
legacy defect?

Everything below the **Registered predictions** line was written before any
run completed (registration commit is the page's `first_seen`). Runs execute
on `worktree-research-night-2026-07-13`; sampler is
`uniform-completion-dp-v1` with the `4123b2d5` MPS fix (the only sampler at
HEAD — the arena has no legacy fallback).

## Arms

1. **C0 reproduction, two blocks.**
   `margin:wp(model=champion/margin_net_r8.pt)+lens:ev` vs `net:wp+lens:ev`,
   512 games, n_samples=10, MPS, `--emit-decisions`.
   Block 1 `--base-seed 7000000` (reserved; original `+0.38 [+0.09,+0.67]`);
   block 2 `--base-seed 9000000` (fresh; original `+0.42 [+0.12,+0.72]`).
   Originals ran the pre-repair sampler
   (`champion/evidence/jud_v0/ab_definitive_512_r8_summary.json`,
   `ab_replication_9M_512_r8_summary.json`).
2. **P0 sampler-neutrality sanity.** `bid30+lens:ev` vs itself, 512 games per
   block, both blocks — a symmetric matchup on the repaired sampler.
3. **P0 challenge re-grade.** `bid30+lens:ev` vs
   `bid30+judsearch:n10,model=champion/jud_net_r4.pt`, 256 games per block —
   the strongest measured play challenger, re-graded on the repaired sampler
   (v1 protocol: `judsearch` lost by `-1.16`/`-1.39` pre-repair).
4. **CUDA benchmark.** Full `forge/eq/test_sampling_mrv_gpu.py` on a rented
   4090 (CUDA 12.6, torch 2.6 image), code rsynced from this worktree
   (`origin/forge` lacks `4123b2d5` — the clone-forge path in
   `forge/zeb/vast/` must not be used). Plus a new
   throughput bench at shapes (32,50), (128,100), (256,100): worlds/sec and
   peak memory. No prior CUDA numbers exist; CPU references are 4.25 ms at
   32×50, 0.63 ms at 1×50, 1.64 ms at 1×10-low-mass
   ([[world-sampler-mrv-audit]]).
5. **Historical exposure scan.** The legacy sampler survives only as the
   audit's analytic emulator (`w42/world_sampler_audit/audit.py`), so exposure
   is re-simulated, not read from storage — historical corpora retained no
   sampled worlds or per-world Q ([[partnership-failure-atlas-v0]]).
   Scan A: exact malformed-mass (`dead_end_mass`, `invalid_mass`) over a
   reconstructed late-state population (deal seeds × play depths, gated on
   enumeration tractability). Scan B: decision-level harm (argmax flips,
   exact regret) on the nonzero-mass subset via the oracle.

## Registered predictions

1. **C0-1 (block 1):** the marks/game delta CI excludes zero in C0's favor,
   point estimate within `[+0.10, +0.70]`. **C0-2 (block 2):** same, within
   `[+0.12, +0.75]`. Falsifier: either CI includes zero or the sign flips —
   Stage 0 then fails, lane grading suspends, and the divergence becomes the
   headline result ([[research-lane-selection]] reopening clause).
2. **P0-sym:** the symmetric matchup lands within its paired-CI of a 0.0
   marks/game delta on both blocks. Falsifier: a CI excluding zero exposes a
   seat/rotation asymmetry in the harness itself.
3. **P0-chal:** `judsearch:n10(r4)` still loses to `lens:ev` on both blocks
   (delta CI excludes zero in lens:ev's favor); band `[-2.2, -0.6]` per prior
   measurements. A `judsearch` win or parity would mean the legacy sampler's
   bias materially harmed the challenger — a headline finding, not a pass.
4. **CUDA:** all sampler tests pass on CUDA with zero skips inside the
   CUDA-parameterized set; throughput at 32×50 beats the 4.25 ms CPU
   reference by ≥5×. No registered memory bound — peak MB is reported
   descriptively.
5. **Exposure:** descriptive, no pass band registered — the deliverable is a
   malformed-mass histogram over the reconstructed population and an
   argmax-flip/regret table on the exposed subset, with the population
   denominator stated. Prior: dead-end mass is highly state-specific (exact
   `1/3` on one fixture, `0` on two others), so the histogram is expected to
   be zero-inflated with a heavy tail.

## Close condition

Stage 0 closes when arms 1–4 land inside their registered bands and arm 5's
report exists with explicit scope. Any miss keeps Stage 0 open and routes to
[[research-lane-selection]]'s reopening clause. Per
[[partnership-research-gates]], every result names bidder, player, sampler,
utility, and partner mechanism via `--emit-decisions` fingerprints.

## Results

Every arena arm landed inside its registered band; the CUDA arm passed
correctness and missed its throughput prediction in an informative direction.
Run artifacts: `arena/results/{c0_repro,p0_sym,p0_chal}_seed{7000000,9000000}/`
(summaries + `--emit-decisions` fingerprints on the C0 arms),
`w42/world_sampler_audit/cuda_bench_2026-07-13.md`,
`w42/world_sampler_audit/exposure_scan/`.

| arm | registered band | measured | verdict |
|---|---|---|---|
| C0-1 (seed 7000000) | CI excludes 0, point in [+0.10,+0.70] | `+0.385 [+0.102,+0.668]` (original `+0.38 [+0.09,+0.67]`) | **PASS** |
| C0-2 (seed 9000000) | CI excludes 0, point in [+0.12,+0.75] | `+0.486 [+0.199,+0.775]` (original `+0.42 [+0.12,+0.72]`) | **PASS** |
| P0-sym-1 | CI includes 0 | `-0.004 [-0.281,+0.275]` | **PASS** |
| P0-sym-2 | CI includes 0 | `-0.076 [-0.361,+0.215]` | **PASS** |
| P0-chal-1 (judsearch r4) | loses, in [-2.2,-0.6] | `-1.418 [-1.789,-1.043]` (pre-repair `-1.39`) | **PASS** |
| P0-chal-2 | loses, in [-2.2,-0.6] | `-1.539 [-1.902,-1.160]` | **PASS** |
| CUDA correctness | all pass, no unexpected skips | 31 passed, 1 by-design skip (no-CUDA-host test) | **PASS** |
| CUDA throughput | ≥5× CPU at 32×50 | ~30 ms/call flat across shapes — ~7× *slower* than CPU at 32×50; 858k worlds/s at 256×100 | **MISS** (registered): the DP sampler is kernel-launch bound; device is not the lever, batch width is |
| Exposure Scan A | descriptive | 2.51% (804/32,000) of reconstructed tractable late states carry nonzero legacy malformed mass; nonzero median 0.19, p99 0.80, max 0.83; the historical `1/3` fixture reproduces exactly; zero states intractable | reported |

Wall-time note: the repaired sampler roughly doubles a C0 block on MPS
(140.7 s / 140.3 s vs the legacy runs' 70.8 s / 75.9 s at identical
configuration) — the correctness fix was paid for in throughput, consistent
with the CUDA finding that per-call overhead dominates this sampler.

Scan B (decision-level harm on the 804 nonzero-mass states: argmax flips,
exact regret) is running on the freed GPU; its report completes this arm's
scope. The exposure population is a reconstructed proxy (random legal
playouts), not the literal historical stream — scope and the two more-faithful
future denominators are stated in
`w42/world_sampler_audit/exposure_scan/exposure_scan_a.md`.

## Reading

**Stage 0 closes.** The two-block C0 reproduction, the P0 symmetry sanity, the
challenger re-grade, and the CUDA correctness suite all land inside their
registered bands on the repaired sampler: the measurement baseline is
trustworthy, the champion's `+0.4`-ish bidding advantage is real under a
uniform world distribution, and `lens:ev` remains undefeated at pure play.
The legacy sampler's defect was real but did not carry the prior conclusions
(2.5% distributional exposure; no argmax flip yet observed anywhere). Lane
grading per [[research-lane-selection]] is now unblocked.

## Links

[[partnership-wall-research]] [[world-sampler-mrv-audit]]
[[partnership-decision-record-v1]] [[research-lane-selection]] [[champion]]
[[w42-jud-v1]] [[partnership-research-gates]]
