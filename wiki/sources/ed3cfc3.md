---
title: "Source digest: ed3cfc3 — MLX-LM batch_generate ceiling bench (43 → 1334 tok/s)"
kind: source
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Commit

- **SHA:** ed3cfc30c7c48860536712438f4edeba06b8da66
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> bench(burl): MLX-LM batch_generate ceiling on M5 Max — 43 → 1334 tok/s
>
> Peak 1334 tok/s at batch=128, 90% of peak at batch=64, vs 43 tok/s
> single-stream. 16× aggregate speedup, memory plateau 15 GB on 48 GB.
> N=500 rollouts ~3.5 min wall time. Reproducer: bench_batch_throughput.py

Benchmark establishing the batch_generate ceiling. Introduces `bench_batch_throughput.py` sweep script and writeup. Full analysis: [[batch-throughput-bench]].

## Related pages

[[batch-throughput-bench]] · [[mlx-lm]] · [[burl]] · [[6a97d55]]
