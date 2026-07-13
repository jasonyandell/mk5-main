# CUDA validation + throughput bench — uniform-completion-dp-v1

Stage 0 arm 4 of `wiki/experiments/stage-0-closure.md`. Run 2026-07-13 on a
rented Vast.ai RTX 4090 (instance 44670122, Texas US, $0.375/hr, destroyed
after the run; total spend ≈ $0.10).

- Host: `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`, driver 570.211.01,
  torch 2.6.0+cu126.
- Code: rsynced from worktree `worktree-research-night-2026-07-13` @
  `090e7840` (carries the `4123b2d5` MPS/where fix). `origin/forge` does NOT
  carry the fix — the `forge/zeb/vast/vast_up.sh` clone-forge path must not be
  used for sampler work until forge is rebased.

## Correctness

```
python -m pytest forge/eq/test_sampling_mrv_gpu.py -q
31 passed, 1 skipped in 10.85s
SKIPPED [1] forge/eq/test_sampling_mrv_gpu.py:588: requires a host without CUDA
```

Every device-parameterized regression (historical dead-end fixture,
valid-only-bias uniformity, the CUDA-gated `TestMRVSamplerGPU` class) passed
on CUDA. The single skip is the no-CPU-fallback error test, which by design
runs only on hosts without CUDA. Zero unexpected skips.

## Throughput (`w42/world_sampler_audit/bench_sampler.py`, 3 warmup + 20 timed calls)

```
shape=(32,50)   ms_per_call=30.66  worlds_per_s=52,181   peak_mb=5.4
shape=(128,100) ms_per_call=30.43  worlds_per_s=420,670  peak_mb=32.3
shape=(256,100) ms_per_call=29.84  worlds_per_s=857,973  peak_mb=62.8
```

Inputs: full 21-tile pools, 7/7/7 hand sizes, no voids (the unconstrained
audit shape).

## Reading

Per-call latency is flat (~30 ms) across an 8× batch-size range: the
suffix-DP sampler is **kernel-launch bound on CUDA**, not compute bound. The
registered prediction (≥5× faster than the 4.25 ms CPU reference at 32×50)
**missed** — at that shape CUDA is ~7× *slower* per call than CPU. CUDA pays
off only through batch width (858k worlds/s at 256×100). Peak memory is
negligible (≤63 MB).

Operational consequence: batch-per-call size, not device, is the sampler's
throughput lever. Small-batch consumers (JudSearch n=10 per decision, the
audit shapes) should prefer CPU; only wide batched generation (eval-aux at
128×100+) benefits from CUDA, and even there ~30 ms/call of launch overhead
dominates unless calls are amortized.
