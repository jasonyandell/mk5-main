---
title: Burl Perf — Phase 2 (Continuous Batching + Prefix Sharing)
kind: experiment
first_seen: TBD
last_updated: TBD
status: active
---

## Overview

Phase 2 of the [[perf-on-the-table]] sprint: drive Burl's per-decision
inference latency from ~16 s/decision (batched-bf16 baseline at batch=5,
M5 Max, 79.5 s wall on the 5-decision frozen subset) toward ~2 s/decision.
Phase 2 owns the two largest single levers in the wiki's perf table:

  1. **Prefix sharing on prefill** — cross-turn KV-cache reuse within a decision.
  2. **Continuous batching** — replace the sync-wave loop's straggler tail with a
     queue-based dispatcher that issues the next decision's next turn as soon
     as the GPU has slots.

## Lever 1 audit — mlx-lm prefix-cache support (already-built findings)

`mlx_lm.batch_generate` (mlx-lm 0.31.2) already exposes the primitives:

  - `prompt_caches: List[List[Any]]` — pre-computed per-stream KV caches.
  - `return_prompt_caches: bool = False` — if `True`, the response carries
    the post-decode caches in `BatchResponse.caches`.
  - `BatchGenerator` (the underlying class in `generate.py:1486`) already
    implements continuous batching: `_unprocessed_sequences` deque,
    `prefill_batch_size` + `completion_batch_size` knobs, automatic prompt→
    generation handoff in `_next()`, and per-stream cache extraction via
    `extract_cache(uids)`.

The Burl wrapper `GemmaLocalNativeBatched` (`burl/modal/gemma_local_batched.py`)
re-prefills the **entire** prompt every turn for every decision — never
passes `prompt_caches`, never asks for `return_prompt_caches`. So
turn-to-turn within a single decision (where the prefix grows monotonically
with each turn's tool-call output) is the lever: keep the cache, trim to
the prompt boundary, prefill only the new suffix.

The Gemma 4 cache types are `KVCache` and `RotatingKVCache`
(`mlx_lm.models.gemma4_text:make_cache`) — both are `is_trimmable()`,
so the LRU trie's prefix-aware fetch works for them.

The mlx-lm 0.31.2 batch>=14 broadcast bug (see [[batched-harvest-resilience]])
remains in the `prefill_batch_size=8` default path; we keep the existing
workaround (`batch=8` or `prefill_batch_size=2`) untouched.

References:
  - mlx-lm `BatchGenerator`: `.venv/.../mlx_lm/generate.py:1486`
  - mlx-lm `batch_generate`: `.venv/.../mlx_lm/generate.py:1879`
  - cache trim: `.venv/.../mlx_lm/models/cache.py:88-111`
  - LRU prompt cache (server-side prefix-aware): `cache.py:1589`

## Phase 2 baseline — temp=0 run vs temp=0.6 run

Phase 0's baseline-bf16 row at temp=0.6 is the production-faithful number
(79.5 s for 5 decisions, decode 87 tok/s, peak 11.59 GB).  At temp=0.6,
sampling-driven turn-count drift gives only an 80% K1-match floor on the
5-row subset, which is too noisy for the validation gate.

Phase 2 introduces a temp=0 baseline (`baseline-bf16-t0`) for the gate.
Two consecutive temp=0 runs land at 71.0 s and 73.7 s (within MLX's
kernel-nondeterminism envelope: same final_play and same eq_delta on all
5 decisions, but n_turns can flip by 1 across runs).  K1-grade match
across two temp=0 runs is **100%** on the 5-row subset.

Validation rule for Phase 2 levers:

  - K1-grade match vs temp=0 baseline on the 5-row subset must be 100%.
  - Wall delta target: ≥10% for Lever 1 (prefix), ≥30% for Lever 2 (continuous).

## Lever 1 — TODO

Cross-turn KV-cache reuse, plumbed through `step_batch`.

## Lever 2 — TODO

Continuous-batching dispatcher.

## Links

[[perf-on-the-table]] · [[burl-perf-phase0]] · [[batch-throughput-bench]] ·
[[batched-harvest-resilience]] · [[mlx-lm]] · [[burl]]
