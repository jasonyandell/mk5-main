---
title: Burl Perf — Phase 2 (Continuous Batching + Prefix Sharing)
kind: experiment
first_seen: 29da3d2
last_updated: 29da3d2
status: active
---

> **Headline (re-validated 2026-04-27 on a clean GPU):**
> The original "1.8–2.1× lever-2 win" was almost entirely cross-scribe
> GPU contention with [[burl-perf-phase1]]'s parallel batch=5 jobs.
> A 7-row alternating clean re-run shows **continuous batching is a
> statistical tie with the wave loop on `perf_subset_5`** (mean 38.1 s
> vs 38.9 s, −1.9%; pairwise deltas −31.7%, +41.1%, −9.1% — direction
> alternates).  Prefix-cache also lands as a tie (+8.4% vs adjacent
> baseline; the 137 s contended reading was contention).  **The
> structural Phase-2 contributions stand** — the mlx-lm internals
> writeup, the Lever-1 root-cause diagnosis, and the cohort + dispatcher
> design specs — but the *magnitude* claims for both levers are
> retracted on the 5-row bench.  Phase 4's full-560 pass is the gate
> for any defensible magnitude.
>
> **Why the 5-row bench can't see a real continuous-batching win:** at
> 5 decisions = 1 wave, the only straggler-tail savings continuous
> batching can deliver are within a single wave's heterogeneous turn
> counts — and the bench's intrinsic 3.4× wall variance (memory
> `project_perf_subset_5_noise_floor`: same-config 40 s / 79.5 s /
> 134.5 s across sessions; clean re-run 35 s / 37 s / 45 s) drowns
> any sub-2× lever effect.  Continuous batching's actual production
> win lives in the *cross-wave* straggler tail, which only manifests
> at scale (560+ decisions).
>
> What survives independent of any wall measurement:
>
> 1. The mlx-lm `_merge_caches` heterogeneous-pad penalty is a documented
>    property of `BatchKVCache.merge` (`mlx_lm/models/cache.py:1056-1085`):
>    every batched decode step processes `B × max_length` even when most
>    streams have `size()==0`.  That predicts a slowdown for any
>    mixed-cache batch independent of contention.
> 2. The Lever-1 implementation has a chat-template alignment bug
>    independent of contention — assistant content is stored as plain
>    `content`, but the chat template re-extracts structured
>    `tool_calls` + `reasoning_content` on re-render.  The cached KV
>    from decode covers the raw `<|channel>thought>` /
>    `<|tool_call>` markers; the next-turn render wraps them with
>    `<|turn>model\n…<turn|>` (assistant role normalised to `model`)
>    that the decode never wrote.  Result: trie longest-common-prefix
>    terminates short and partial-cache reuse drifts model behaviour.
>    See "Lever 1 — root-cause writeup" below.
>
> Re-validation has been done on a clean window (results in
> "Re-validated 5-row alternating run" below).  Defensible Phase-2
> magnitude must wait for [[burl-perf-phase4]]'s full-560 pass.

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

## Lever 1 result — closed; structural failure modes confirmed

`mlx_lm.models.cache.LRUPromptCache` was plumbed through `batch_generate`
via a new `enable_prompt_cache` flag on `GemmaLocalNativeBatched`.  The
trie does match prefixes (28% of prompt tokens hit on the 5-decision
subset, 35,285 / 125,239) but the lever closes for two structural
reasons that hold independently of any wall measurement (see "Lever 1
— root-cause writeup" below).  The contended runs read 137 s vs a
71 s baseline; the clean re-run reads:

| variant (clean re-run, alternating)  | wall  | decode tok/s | K1 vs t0 |
|--------------------------------------|------:|-------------:|---------:|
| baseline-bf16-t0 (row 5)             | 35.2 s|          179 |     5/5  |
| prefix-cache (row 6)                 | 38.2 s|          169 |     5/5  |

So in a clean window the lever is a small loss (+8.4% vs adjacent
baseline) — well inside noise but explained by the
heterogeneous-cache merge-pad penalty.  K1 holds at 5/5 because the
trie hit rate is small enough on the 5-row subset that the
chat-template-misalignment correctness bug doesn't surface — but the
bug is still in the code, and would surface at scale where more
turn-to-turn cache hits happen.

Lever 1 in this shape is **not viable** on M5 Max with mlx-lm 0.31.2
`batch_generate`.  A correct implementation needs to extract the cache
at a stable boundary (the *end of segment* hook in `BatchGenerator`,
before the assistant turn opens) and feed the cache back into the
next turn's prefill — which is what the mlx-lm `server.py` does.  But
that requires a continuous-batching dispatcher (Lever 2) to avoid the
heterogeneous-merge penalty: only streams at similar cache widths get
fused into the same decode step.

Lever 1 is therefore subsumed by Lever 2 on this hardware.

Artefact: `burl/eval/results/perf_20260427_024050_prefix-cache-revalidate.json`.

## Lever 1 — root-cause writeup (non-GPU audit)

This section records what is provably true about the Lever-1 implementation
from reading mlx-lm source + the Gemma 4 chat template — independent of
any wall-time measurement.  Two failure modes:

### Failure mode A — heterogeneous-cache batched decode pads to max width

`mlx_lm.generate.PromptProcessingBatch.__init__` wraps each stream's
per-stream cache through `_merge_caches`
(`mlx_lm/generate.py:1036`), which delegates to
`BatchKVCache.merge` (`mlx_lm/models/cache.py:1056-1085`):

```python
@classmethod
def merge(cls, caches):
    lengths = [c.size() for c in caches]
    max_length = max(lengths)
    ...
    keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
    values = mx.zeros((B, H, max_length, Dv), dtype=dt)
    for i, (p, c) in enumerate(zip(padding, caches)):
        keys[i:i+1, :, p:p+c.offset] = c.keys[..., :c.offset, :]
        ...
```

Each batched decode step then runs `model(inputs[:, None],
cache=self.prompt_cache)` over `B × max_length`.  For Lever 1's
turn-N→N+1 pattern, after the first turn the streams have mixed cache
sizes — a stream that finished its turn in 200 generated tokens has
`size()=prompt_T1+200`, while a stream still on turn 1 might be 0.
The fast streams' attention runs at the slow stream's KV width.

This is a **structural property of mlx-lm 0.31.2**, not a contention
artifact.  It predicts that Lever 1, in any shape that shares the same
batched decode call across heterogeneous-cache streams, will lose on
decode tok/s.  Confirmed by reading source; the magnitude (84 → ~45)
needs an uncontended re-run to be precise but the direction is fixed.

### Failure mode B — chat-template re-render misaligns the trie key

The Gemma 4 chat template (`chat_template.jinja` in the `mlx-community/
gemma-4-e2b-it-bf16` snapshot) does three things on each render:

1. **Role normalization.**  `assistant` → `model`:
   ```jinja
   {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
   ```
   The decoded raw token stream during generation has no `<|turn>model`
   wrapper because the model is *inside* the assistant turn that
   `add_generation_prompt=True` opened.  The next-turn render *closes*
   that turn with `<turn|>` and (if continuing) opens a new one.
2. **Continuation detection.**  `continue_same_model_turn` suppresses
   the second `<|turn>model\n` opener if two assistant messages are
   adjacent.  This means the boundary tokens between turn-N and
   turn-N+1 depend on the prior assistant message structure, which
   the trie key has no way to anticipate.
3. **Structured re-extraction of tool calls and reasoning.**
   ```jinja
   {%- if message['tool_calls'] -%}
       {%- for tool_call in message['tool_calls'] -%}
           {{- '<|tool_call>call:' + function['name'] + '{' -}}
           ...
   ```
   The chat template *only* renders `<|tool_call>` markers if
   `message['tool_calls']` is present as a structured list.  But the
   Phase-2 implementation stores the full assistant completion in
   `message['content']` and never populates `tool_calls`/`reasoning`
   on the appended assistant message.  So:

   - **What the model decoded:** raw tokens including
     `<|channel>thought\n…<channel|><|tool_call>call:explore_game{play:0}<tool_call|><eos>`.
   - **What the next turn's chat template renders for the *same*
     content:** `<|turn>model\n{the entire raw text including all
     the markers as plain text}<turn|>`.
   - The two token streams diverge at every Gemma special token
     boundary.

The trie's `search` walks the new prompt token stream until it finds
the first divergence; that's where `common_prefix` ends.  Since the
divergence sits at the `<|turn>system\n…<turn|>` boundary that opens
the assistant turn (very early in the prompt), the cache hit is
short — but worse, on the boundaries that *do* match by coincidence
(generic content tokens), the cached KV is from a position where the
model had different surrounding context, which corrupts the attention
pattern and drifts logits.

This explains the 60% K1 match: not a tolerance issue, a correctness
bug.  Independent of contention.

### What a correct Lever-1 implementation needs

The mlx-lm `server.py` does this right.  `BatchGenerator.insert_segments`
(`generate.py:1599-1647`) accepts `segments: List[List[List[int]]]` —
each stream is split into prompt-segments where each segment-end is a
"stable boundary" the cache should snapshot.  The server uses this to
cache the system+rules+user prefix of every request:

```python
# server.py:746-757
self.prompt_cache.fetch_nearest_cache(...)
batch_generator.insert_segments(
    segments=[segments],
    caches=[cache],
    all_tokens=[prompt[:prompt_cache_count]],
    ...
)
# After end_of_segment: server.py:836-851
caches = batch_generator.extract_cache(eos_ids)
self.prompt_cache.insert_cache(model_key, cache_key, cache,
                               cache_type="user")
```

The cache is captured *at the end of the user segment*, before the
assistant turn opens.  When a later request has the same system+
user prefix, it reuses that cache and resumes from the assistant
boundary — never trying to bridge across an `<|turn>model\n…<turn|>`
re-render.

For Burl, the cache hit pattern would only work *across decisions
that share a literal-token prefix* — i.e. the system prompt + rules
primer (~1500 tok) before the per-decision game-state block diverges.
Cross-turn within a decision is not addressable through this API
because the decision's history grows with raw-decoded tokens that
the chat template would never re-emit verbatim.

The Lever-1 expectation in [[perf-on-the-table]] ("1.5–2× expected")
was therefore over-optimistic on M5 Max for this harness shape.  The
realistic shape on this hardware is "shared system+user prefix
*across the wave's decisions*, captured once per wave-start" — which
saves ~5 × 1500 = 7,500 prompt tokens (~10–15% of total prefill at
the 5-decision subset's ~80k prompt tokens).  Worth doing if it
slots into the [[continuous-batching-dispatcher-design]] cleanly,
not worth a standalone lever.

## Lever 2 result — statistical tie on the 5-row bench

The contended-window read of "1.8–2.1× faster" was almost entirely
GPU contention with [[burl-perf-phase1]]'s parallel batch=5 jobs.
The clean alternating re-run (7 rows on uncontested GPU, sequence
B-C-B-C-B-P-C, 2026-04-27) lands inside the bench's intrinsic noise:

| row | variant         | wall_total | wall_p50 | decode tok/s | K1 vs t0 |
|----:|-----------------|-----------:|---------:|-------------:|---------:|
|  1  | baseline-bf16   |     44.5 s |   42.8 s |          140 |     5/5  |
|  2  | continuous      |     30.4 s |   28.4 s |           67 |     5/5  |
|  3  | baseline-bf16   |     36.8 s |   32.5 s |          173 |     5/5  |
|  4  | continuous      |     52.0 s |   31.6 s |           53 |     5/5  |
|  5  | baseline-bf16   |     35.2 s |   32.5 s |          179 |     5/5  |
|  6  | prefix-cache    |     38.2 s |   36.8 s |          169 |     5/5  |
|  7  | continuous      |     32.0 s |   26.2 s |           64 |     4/5  |

Aggregates:

- baseline mean: 38.9 s (range 35–45 s)
- continuous mean: 38.1 s (range 30–52 s)
- **mean(continuous) − mean(baseline) = −1.9%** — a tie
- pairwise deltas vs immediately adjacent baseline: **−31.7%, +41.1%, −9.1%** —
  *direction alternates*, the hypothesis "continuous is consistently
  faster than its adjacent baseline" fails the alternation test
- prefix-cache vs adjacent baseline: +8.4% (also a tie)

What the clean re-run reveals:

- **Continuous batching's wall variance is *wider* than the baseline's**
  on `perf_subset_5` (range 21.5 s vs 9.3 s).  The dispatcher genuinely
  changes batch composition step-to-step and surfaces kernel-noise that
  the wave loop's lockstep nature suppresses.  This widening is the
  same mechanism that flips gi=72's K1 grade: marginal decisions sit
  on a fence and the noise pushes them either way.
- **Decode tok/s on the baseline jumped from ~85 (contended) to
  140–179 (clean).**  The Phase-0 reference baseline of 87 tok/s
  was itself contended-or-cold; the true clean ceiling for the
  wave-loop baseline is closer to 175 tok/s.
- **Prefix-cache decode tok/s recovers to 169** (vs 45 contended) —
  the merge-pad penalty is real but quiet at small cache sizes in a
  hot session.

Conclusion: **the 5-row bench cannot resolve the lever-2 effect.**
Continuous batching's expected production win is the *cross-wave*
straggler tail savings at scale (560+ decisions, where some decisions
finish 3 turns ahead of others and the wave-loop GPU sits idle on
every wave's tail).  At 5 decisions = 1 wave, only intra-wave straggler
savings apply, and those are smaller than the bench's variance.

What survives the re-run:

- `BatchGenerator` already implements continuous batching at the kernel
  level (`prefill_batch_size`, `completion_batch_size`,
  `_unprocessed_sequences` deque, automatic prompt→generation handoff
  via `_next()`).  The wrapper-side change in `run_bench_continuous`
  is putting tool dispatch + state transitions on the same continuous
  timeline rather than gate-driving turn-by-turn batches — a real
  architectural improvement that costs nothing.
- The cohort + dispatcher design specs ([[continuous-batching-dispatcher-design]],
  [[harvest-cohort-abstraction]]) hold; their value is unblocking
  Phase-4 production-harvest work where the cross-wave straggler
  savings actually manifest.
- Phase 4's full-560 pass is the gate for any defensible Lever-2
  magnitude.

Artefacts: `burl/eval/results/perf_20260427_023638..024141_*.json`
(rows 1–7 of the clean re-run).

## Earlier (contended) reading — preserved for the lessons-learned trail

The original Phase 2 commit (`29da3d2`) reported Lever 2 at 1.8–2.1×
on the basis of three runs collected during scribe-B's parallel-job
window.  Numbers below were never produced on a clean GPU:

| variant                   | wall_total | wall_p50 | prefill tok/s | decode tok/s | peak GB | K1 vs t0-baseline |
|---------------------------|-----------:|---------:|--------------:|-------------:|--------:|------------------:|
| baseline-bf16-t0 (run 1)  |     71.0 s |   66.0 s |        11,073 |         84.8 |   11.30 |               5/5 |
| baseline-bf16-t0 (run 2)  |     73.7 s |   65.8 s |        10,780 |         82.8 |   11.33 |               5/5 |
| continuous (run 1, t=0)   |     34.5 s |   23.0 s |        27,672 |         68.0 |   10.80 |               5/5 |
| continuous (run 2, t=0)   |     35.0 s |   33.0 s |        27,718 |         65.5 |   10.80 |               4/5 |
| continuous (run 3, t=0)   |     39.9 s |   38.9 s |        19,219 |         50.5 |   10.80 |               5/5 |
| continuous (t=0.6)        |     58.6 s |   36.9 s |        26,183 |         56.2 |   12.25 |        — vs t=0.6 |

**Lessons learned:**

1. The 5-row bench's intrinsic 3.4× variance (memory
   `project_perf_subset_5_noise_floor`) is real, not just contention.
   Same-config `baseline-bf16` reproduces at 40 s, 79.5 s, 134.5 s
   across sessions on M5 Max — even unloaded.
2. Cross-scribe GPU contention can produce a *coherent* false signal.
   In the contended window, the lever-2 numbers were consistently
   "fast" because contention hits prefill-heavy code paths harder
   than decode-heavy ones, and the wave loop happens to be more
   prefill-dominant than the dispatcher.  The "win" was a contention
   artifact correlated with the variant.
3. **Always alternate variant-vs-baseline in the same session.**  The
   clean re-run alternation produced the honest "tie" reading; the
   original three-runs-in-a-row pattern cannot tell signal from
   timing-of-day.
4. Direction-of-mean is not the same as direction-of-pairwise-deltas.
   On a 3-vs-3 sample with 3.4× same-config variance, `mean(C) <
   mean(B)` would have to be ≥ 30% to count as signal; we got −1.9%.
5. The **structural** Phase-2 contributions (mlx-lm internals
   writeup, Lever-1 root-cause diagnosis, dispatcher + cohort design
   specs) are the real win.  They unblock Phase-4 work that *can*
   resolve the magnitude question.

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
