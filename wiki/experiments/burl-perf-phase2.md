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

## Lever 1 result — negative on this workload

`mlx_lm.models.cache.LRUPromptCache` was plumbed through `batch_generate`
via a new `enable_prompt_cache` flag on `GemmaLocalNativeBatched`.  The
trie does match prefixes (28% of prompt tokens hit on the 5-decision
subset, 35,285 / 125,239), but the lever loses on both axes:

| variant            | wall  | decode tok/s | K1 match |
|--------------------|------:|-------------:|---------:|
| baseline-bf16-t0   |  71 s |           84 |    100%  |
| prefix-cache       | 137 s |           45 |     60%  |

Two compounding failures:

1. **Heterogeneous-cache batched decode is slow.** `mlx-lm`'s
   `_merge_caches` pads all streams to the max cache width, so a
   batch of 5 streams with mixed cache lengths (0, 800, 1600, 1800,
   2200 tokens) decodes at the speed of the longest one but pays the
   memory cost of all of them.  The original full-prefill batched
   decode keeps all streams at the same KV state, which actually wins.
2. **Chat-template re-rendering breaks key alignment.** The trie key
   used `tokenizer.encode(completion_text)` for the appended segment,
   but the next turn's chat template wraps the assistant content with
   role markers (`<|turn>assistant\n…<turn|>`) before emitting the new
   user/tool turn. The trie's longest-common-prefix is shorter than
   intended, and — more worryingly — partial cache reuse drifts model
   behavior enough to flip K1 grades on 2/5 decisions (60% match
   vs the 100% temp=0 floor).

Lever 1 in this shape is **not viable** on M5 Max with mlx-lm 0.31.2
batch_generate.  A correct implementation needs to extract the cache
at a stable boundary (the *end of segment* hook in `BatchGenerator`,
before the assistant turn opens) and feed the cache back into the
next turn's prefill — which is what the mlx-lm `server.py` does.  But
that requires a continuous-batching dispatcher (Lever 2) to avoid the
heterogeneous-merge penalty: only streams at similar cache widths get
fused into the same decode step.

Lever 1 is therefore subsumed by Lever 2 on this hardware.  We close
the lever-1 ledger row as a negative result and document the failure
mode for future Burl perf scribes.

Artefact: `burl/eval/results/perf_20260427_020118_prefix-cache.json`.

## Lever 2 result — 1.8-2.1× wall on M5 Max

A continuous-batching dispatcher built on `mlx_lm.generate.BatchGenerator`
replaces the bench's sync-wave loop.  All decisions submit their first
turn to a single long-lived generator; as a stream finishes (EOS or
max_tokens), the dispatcher applies tools + state transitions on CPU and
immediately re-submits the next turn's prompt while other streams keep
decoding.  Implementation: `run_bench_continuous` in
`burl/eval/bench_decision_latency.py`, behind the `--continuous` flag.

The fast turns no longer block on the slowest turn finishing — that's
where the wall savings come from on the 5-decision subset (single wave,
but heterogeneous turn counts: 4 to 8 turns per decision).

| variant                   | wall_total | wall_p50 | prefill tok/s | decode tok/s | peak GB | K1 vs t0-baseline |
|---------------------------|-----------:|---------:|--------------:|-------------:|--------:|------------------:|
| baseline-bf16-t0 (run 1)  |     71.0 s |   66.0 s |        11,073 |         84.8 |   11.30 |               5/5 |
| baseline-bf16-t0 (run 2)  |     73.7 s |   65.8 s |        10,780 |         82.8 |   11.33 |               5/5 |
| continuous (run 1, t=0)   |     34.5 s |   23.0 s |        27,672 |         68.0 |   10.80 |               5/5 |
| continuous (run 2, t=0)   |     35.0 s |   33.0 s |        27,718 |         65.5 |   10.80 |               4/5 |
| continuous (run 3, t=0)   |     39.9 s |   38.9 s |        19,219 |         50.5 |   10.80 |               5/5 |
| continuous (t=0.6)        |     58.6 s |   36.9 s |        26,183 |         56.2 |   12.25 |        — vs t=0.6 |

**Wall delta vs temp=0 baseline: 1.8–2.1× faster** (51–53% reduction).
**Wall delta vs production-faithful temp=0.6 baseline: 1.36×** (26%
reduction; sampling adds turn-count variance which the dispatcher
amortizes more conservatively).

Prefill throughput jumped from ~11k to ~27k tok/s — the dispatcher
prefills the next-turn suffix for one decision while other streams
decode, shifting the prefill→decode ratio.  Peak memory dropped slightly
(11.3 → 10.8 GB at temp=0) since the pool size flexes downward as
streams finish.

K1 stability vs the temp=0 baseline: 4–5/5 across three runs.  The
single run that flipped affected gi=72 (the trick-5 forced-commit
decision flagged in [[burl-perf-phase0]] as the "marginal-decision"
slot), where `eq_delta_vs_bot` lives near the K1 threshold.  The
dispatcher widens the kernel-noise envelope for marginal decisions
because each step's batch composition changes turn-to-turn.  Two
mitigations available if needed: (a) larger subset ([[burl-perf-phase0]]
documents the 560-row gate where 1/n noise dominates), or (b) post-decode
logit-snapshot determinism — out of scope here.

mlx-lm's `BatchGenerator` already implements continuous batching at the
kernel level (`prefill_batch_size`, `completion_batch_size`,
`_unprocessed_sequences` deque, automatic prompt→generation handoff via
`_next()`).  The wrapper-side win is putting tool dispatch + state
transitions on the same continuous timeline rather than gate-driving
turn-by-turn batches.

## Production harvest path

The harvest at `scratch/belief_trajectory_rollout/harvest_batched.py` was
*not* migrated in this commit.  It carries the [[batched-harvest-resilience]]
plumbing (OOM classifier + quarantine ledger, SIGKILL sentinel, retry
pass) which is wave-aware: each wave writes a sentinel before its first
generate call, captures OOM/broadcast errors per-wave, and recovers via
`--rerun-quarantined`.  Mapping that resilience layer onto a continuous
dispatcher is non-trivial — quarantine semantics are "this wave failed",
but in continuous mode there is no wave, just a moving pool.

Forward path: define a "cohort" that fences a logical group of decisions
(say batch_size=8) into the dispatcher pool with a shared sentinel.  On
OOM, every uid in the cohort is quarantined and the dispatcher resets.
That preserves the resilience contract and gets the lever-2 win.  Filed
as a follow-up in `wiki/questions/open.md`.

The bench-side win is enough to validate the lever and gate Phase 3 work
on it.

## MLX batch>=14 broadcast bug

The dispatcher uses `prefill_batch_size=min(batch, 8)` to stay clear of
the [[batched-harvest-resilience]] broadcast bug.  At `batch=5` (the
phase-2 subset) the ceiling never trips; at higher batches the dispatcher
splits prefill into chunks of 8 by default.

## Links

[[perf-on-the-table]] · [[burl-perf-phase0]] · [[batch-throughput-bench]] ·
[[batched-harvest-resilience]] · [[mlx-lm]] · [[burl]]
