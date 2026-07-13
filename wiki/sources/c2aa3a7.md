---
title: "Source digest: c2aa3a7 — async concurrency in STaR rollout (2.5-3.5× speedup)"
kind: source
first_seen: c2aa3a7
last_updated: c2aa3a7
status: active
---

## Commit

- **SHA:** c2aa3a730b09e72861241c3a5f206caf38437a42
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): iteration-speed prep — async concurrency in STaR rollout
>
> Client-side concurrency via asyncio.to_thread + semaphore-gated
> batching. Both Phase A (initial rollouts) and Phase B (EQ-gate) benefit.
> --concurrency N CLI flag, default=1 (bit-identical to prior sequential
> loop at N=1).
>
> Local benchmark (mock, 0.1s/decision):
>   N=8 @ concurrency=1 → 0.809s
>   N=8 @ concurrency=4 → 0.207s (3.92× of theoretical 4.0×)
>
> Expected real-world speedup 2.5-3.5× on L4. Cost-cap checked after
> each batch. 4 new pytest cases. 105/105 passed.

Adds `asyncio.to_thread` + semaphore-gated batching to `run_star_rollout`, benefiting both Phase A (initial rollouts) and Phase B (EQ-gate chains). Default `--concurrency 1` preserves bit-identical output for backwards compatibility. Expected real-world speedup 2.5–3.5× on [[modal]] L4 (vLLM scheduler + shared GPU overhead). Cost-cap logic fires after each gather batch.

## Related pages

[[burl]] · [[star]] · [[modal]] · [[eq-gate-star]] · [[sources/761587c]]
