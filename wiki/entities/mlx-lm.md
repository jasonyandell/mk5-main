---
title: MLX-LM (Apple Silicon local path for Burl)
kind: entity
first_seen: 6fea6ab
last_updated: 1f48fda
status: active
---

## What it is

MLX-LM is Apple's MLX framework + `mlx-lm` library, used by [[burl]] on M-series Macs for
both training and inference. It replaces [[modal]] B200 for the common development case —
no cloud cost, no container cold-start. (commit message @ 6fea6ab)

## Burl usage

| Path | File | Role |
|---|---|---|
| Inference | `burl/modal/gemma_local.py` | In-process MLX-LM [[gemma-4-e2b]] with `NativeModelCallable` interface; adapter swap via constructor arg |
| Training | `burl/train/star_mlx.py` | Port of `star.py` recipe to `mlx_lm.tuner`; `preserve_thoughts` bypass intact |

`--local` / `--model-source local` flags route eval runners through the local path.
Model: `mlx-community/gemma-4-e2b-it-bf16`. Adapters compatible with
`mlx_lm.load(adapter_path=...)` for M5 serving. (commit message @ 6fea6ab)

## Memory and throughput

| Config | Peak memory |
|---|---|
| Rank 4, `grad_checkpoint` | ~10 GB |
| Rank 16, `max_seq_length=4096` | ~26 GB |

Throughput: at least 1.86× wall speedup vs Modal (common workloads). Weight mmap shares
cleanly across processes; up to 5 concurrent workers fit in 48GB unified memory.
(commit message @ 6fea6ab)

## Dependencies

`burl/requirements-mlx.txt`: `mlx>=0.29`, `mlx-lm>=0.29`, `huggingface_hub`,
`torch>=2.6` (for [[forge]] oracle on MPS/CPU). M-series-only deps; not installed in the
Modal image. (commit message @ 6fea6ab)

## Batch generate ceiling on M5 Max (ed3cfc3, 6a97d55)

`mlx_lm.batch_generate` continuous-batched generation on real Burl prompts (mean 2378
tokens, iter-3-rules shape):

| Batch size | Throughput | Notes |
|---|---|---|
| 1 (single-stream) | 43 tok/s | Baseline |
| 64 | ~1206 tok/s | 90% of peak, recommended knee |
| 128 | **1334 tok/s** | Peak (16× aggregate speedup) |

Memory plateau: 15 GB on 48 GB host. N=500 rollouts at ~4 turns × ~128 tokens ≈ 3.5 min
wall. Unlocks "corpus 10-20× larger" as a cheap iter-5+ lever. See
[[experiments/batch-throughput-bench]]. (commit message @ ed3cfc3)

Operationalized in `GemmaLocalNativeBatched` (`burl/modal/gemma_local_batched.py`) and
`run_move4_star_rollout_batched.py`. Wall: N=16 batched at batch=16 in 58s vs sequential
134s (2.3×). Prompt-cache reuse is the obvious next ~2× lever (untested).
(commit message @ 6a97d55)

## SFT truncation fix (edf86e9)

TRL's `SFTConfig` defaults `max_seq_length=1024`, silently truncating rows whose thought
blocks exceed that length. Burl's `preserve_thoughts` corpus has median 2054 and max 4210
tokens/row. The local MLX path was already fixed at ~line 250 of `star_mlx.py`; commit
edf86e9 applied the same fix (`max_seq_length=4096`) to the Modal `star.py` recipe for
parity.

This reframes ingest B7's iter-4 null result: the byte-identical A/B was almost certainly
truncation, not LoRA capacity saturation — no prior Burl adapter was trained on complete
thought-to-tool-call traces. Parallel to LEM's [[decisions/sft-completion-only-loss]]
finding: TRL defaults are traps. (commit message @ edf86e9)

## Internals — what mlx-lm 0.31.2 actually exposes

Audited during [[burl-perf-phase2]].  Useful for any future scribe touching the
inference path; the [[topics/perf-on-the-table]] surface-level summary lives there,
the deep references live here.

### `BatchGenerator` — the continuous scheduler

`mlx_lm.generate.BatchGenerator` (`generate.py:1486`) is mlx-lm's continuous-batching
implementation.  Three internal data structures:

- `_unprocessed_sequences: deque` — submitted prompts waiting for a prefill slot.
  `insert(prompts, max_tokens, caches=...)` appends here; `_make_batch(n)` pops the
  next `n` into `_prompt_batch`.
- `_prompt_batch: PromptProcessingBatch` — streams currently being prefilled.
  `prefill_batch_size` caps width; `prefill_step_size` (default 2048) caps tokens per
  step.  `prompt(tokens)` runs `model(tokens, cache=self.prompt_cache)` over a chunk
  and extends `self.tokens` so it always represents what's in the KV cache.
- `_generation_batch: GenerationBatch` — streams currently decoding.
  `completion_batch_size` caps width.  `next()` runs one decode step and emits a
  `Response` per stream with the new token + (on EOS) the post-decode KV cache.

The scheduler runs in `_next()` (`generate.py:1761-1837`):

1. If `_generation_batch` is non-empty, decode one step.
2. If `_generation_batch` is below `completion_batch_size`, pull from
   `_unprocessed_sequences` to top up `_prompt_batch`.
3. Promote any prefill-finished streams from `_prompt_batch` into `_generation_batch`
   via `PromptProcessingBatch.split` + `generate(last_inputs)`.
4. Run one prefill chunk on whatever's left in `_prompt_batch`.

This is end-to-end continuous batching: prompts enter at any time, decode and prefill
happen on the same step, finished streams free their slot for the next pending prompt.
**The `mlx_lm.batch_generate` function wraps this scheduler but submits all prompts
up-front** (`generate.py:1922`) — it doesn't expose continuous *insertion* across the
generation lifetime.  For continuous insertion, callers must drive the scheduler
directly via `insert()` + `next_generated()`.

### `_merge_caches` and the heterogeneous-cache pad penalty

`PromptProcessingBatch.__init__` calls `_merge_caches(caches)` on the per-stream KV
caches passed in (`generate.py:1036`).  For trimmable caches, this delegates to
`BatchKVCache.merge` (`models/cache.py:1056-1085`):

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
```

Every batched decode step processes `B × max_length` even when most streams have
`size()==0`.  Streams shorter than the longest are **left-padded** and the pad slots
get masked out by `BatchKVCache.make_mask`.  The attention math is correct; the FLOP
cost is not amortized.

**Practical implication for Burl:** the naive cross-turn cache-reuse pattern
("turn N+1 starts from a stream's post-decode cache; other streams in the batch
start fresh") loses on M5 Max because the fresh streams pay the long-cache stream's
KV width.  See [[burl-perf-phase2]]'s "Lever 1 — root-cause writeup" for the full
diagnosis.  The right pattern is the server's: cache the system+user prefix only,
captured at a stable boundary via `BatchGenerator.insert_segments` + `extract_cache`,
and reuse it only across requests that literally share that prefix.

### `LRUPromptCache` — the prefix-aware trie

`mlx_lm.models.cache.LRUPromptCache` (`models/cache.py:1589`) is what
`mlx_lm.server` uses to dedupe prefixes across requests.  Two operations:

- `fetch_nearest_cache(model_key, tokens) -> (cache, remaining_tokens)`.  Walks the
  trie under `model_key`, finds the longest-common-prefix among inserted entries,
  deepcopies that entry's cache, trims it to the prefix length, and returns
  `(cache, tokens_after_prefix)`.  Caller passes the trimmed `tokens_after_prefix`
  as the prompt to prefill, with the deepcopied cache as starting state.
- `insert_cache(model_key, tokens, prompt_cache, *, cache_type="assistant")`.
  Adds a post-decode cache to the trie; supports per-type LRU eviction
  (`assistant` < `user` < `system` priority).

**Constraint that bit Phase 2:** `model_key` must be hashable.  `nn.Module` is not.
Use a string key (`f"{repo}:{adapter}:{instance_id}"`).

**Constraint that ruled out a Phase-2 lever:** the trie key is the literal token
sequence.  Decoded tokens during generation include raw `<|channel>thought>` /
`<|tool_call>` markers.  The next turn's chat-template re-render wraps those same
tokens with `<|turn>model\n…<turn|>` (Gemma 4 normalises `assistant` → `model`)
that the decode never wrote.  The trie's longest-common-prefix terminates at the
first divergence — short.  See [[burl-perf-phase2]] for line citations.

### `BatchKVCache` left-padding semantics

`BatchKVCache.__init__(left_padding: List[int])` expects per-stream left padding so
that the attention mask treats the pre-padded positions as causal-history-absent.
This is how shorter prompts fit alongside longer ones in a fused batched prefill.
`prepare(left_padding=, lengths=, right_padding=)` lets the caller adjust padding
at insert time; `finalize()` rolls right-padding into the cache via `dynamic_roll`
once the prompt processing finishes.  Burl never needs to touch this directly — it
sits below `BatchGenerator` — but the existence of `dynamic_roll` (an `mx`
operation that's metal-cost-non-trivial) is the kind of thing that explains odd
prefill stalls if you ever measure them.

### Minor gotchas worth a memory entry

- `BatchGenerator.insert(prompts=...)` requires every prompt non-empty; passing
  `prompts=[[]]` for a 100%-cache-hit stream raises.  mlx-lm's server works around
  this with `seq.append(seq[-1][-1:])` — splits off the last token of the prompt
  to keep the segment non-empty (`generate.py:1638-1640`).
- `prefill_batch_size=8` (default) trips the 0.31.2 `broadcast_shapes` bug at
  batch≥14 with heterogeneous prompts (>2300-token variance).  Workaround:
  `prefill_batch_size=2` or `batch=8`.  See [[topics/batched-harvest-resilience]].
- `BatchGenerator.close()` releases the wired-memory limit set in `__init__` via
  `mx.set_wired_limit(...)`.  Long-running dispatchers should call `close()`
  between cohorts on OOM-rebuild paths to avoid leaking wired memory.
