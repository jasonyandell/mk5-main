# Batch throughput on M5 Max — MLX-LM `batch_generate` ceiling sweep

**Date**: 2026-04-19
**Host**: M5 Max, 48 GB unified memory
**Model**: `mlx-community/gemma-4-e2b-it-bf16`
**MLX-LM**: 0.31.2
**Prompt shape**: iter-3-rules (rules-as-tools + primer + 42-framing), mean 2378 tokens across 16 real held-out decisions.
**Reproducer**: `PYTHONPATH=. python -u -m burl.eval.bench_batch_throughput`

## TL;DR

Continuous-batched generation via `mlx_lm.batch_generate` scales near-linearly through batch=64 and peaks ~16× single-stream at batch=128. Memory is nowhere near saturated.

| batch | wall (s) | peak mem (GB) | aggregate tok/s | per-seq tok/s | vs single (83 tok/s) |
|------:|---------:|--------------:|----------------:|--------------:|---------------------:|
|     1 |      2.9 |          10.1 |              93 |          92.5 |                1.1× |
|     4 |      5.1 |          10.7 |             166 |          41.6 |                2.0× |
|     8 |      5.4 |          11.5 |             278 |          34.8 |                3.3× |
|    16 |      4.0 |          11.7 |             512 |          32.0 |                6.2× |
|    32 |      6.0 |          12.1 |             908 |          28.4 |               10.9× |
|    48 |      7.7 |          12.5 |            1032 |          21.5 |               12.4× |
|  **64** | **9.9** |      **13.8** |        **1206** |      **18.8** |            **14.5×** |
|    96 |     14.4 |          14.7 |            1245 |          13.0 |               15.0× |
| **128** | **19.2** |     **15.5** |        **1334** |      **10.4** |            **16.1×** |
|   192 |     27.5 |          17.0 |            1301 |           6.8 |               15.7× |
|   256 |     37.4 |          19.2 |            1309 |           5.1 |               15.8× |

Single-stream baseline (8 serial `stream_generate` calls, same model/prompts): **83 tok/s aggregate.**

## Bottom line

- **Peak throughput**: ~1334 tok/s at batch=128, 15.5 GB peak memory.
- **Recommended default for the batched rollout harness**: **batch=64** — 90% of peak aggregate (1206 tok/s), half the wall-per-cycle of batch=128, 14 GB peak (28% of available memory). Leaves room for tool-dispatch work, `prompt_caches` overhead, and sampling jitter.
- **Headroom**: even batch=256 only uses 40% of the 48 GB box. Bigger models (E4B) would still fit in these shapes.
- **KV marginal cost**: ~40 MB per additional sequence (sequences here hold ~2400 prompt + 128 generation = 2528 tokens of KV). Very affordable.

## Old vs new arithmetic

The previous baseline for `gemma_local.py` was **43 tok/s** single-stream (documented in `burl/modal/gemma_local.py` header). Corpus generation for N=30 rows × ~4 turns × ~128 tokens/turn ≈ 15 000 generation tokens, roughly 6 minutes wall time.

At 1206 tok/s (batch=64), **N=500** rows at the same turn profile: 250 000 generation tokens ≈ **3.5 minutes wall time.**

Corpus scale stopped being a throughput problem. iter-5's "what if the corpus is 10-20× larger" variable is now a trivial lever.

## What the numbers don't include

- **Ragged batches.** All prompts in this bench are the same length per run (16 real prompts, cycled to fill larger batches). Real rollouts mix decisions at different turn counts, which will shift the speedup. `BatchGenerator`'s continuous-batching design should handle it but the delta is unmeasured.
- **`prompt_caches` reuse across turns.** The shared ~2000-token system prompt re-prefills on every turn in the naive path. `batch_generate(..., return_prompt_caches=True)` + `prompt_caches=...` on the next call is a second multiplier on top of this. Untested; expected to help significantly for multi-turn decisions.
- **Real tool dispatch.** The bench is pure generation. Each batched step in the real harness will stall while per-decision tool calls execute (engine lookups, E[Q] rollouts). If tool calls dominate, continuous batching's advantage narrows — but E[Q] at N=10 is ~290 ms/play, small vs the 20-second batch=128 generation step, so the dispatch-vs-generation ratio should stay in batching's favor.

## Why the plateau above batch=128

Aggregate throughput barely moves past batch=128 (1334 → 1301 → 1309). Generation becomes memory-bandwidth bound once enough concurrent sequences live in the KV cache; adding more sequences per step spreads the same bandwidth thinner without recovering it. Diminishing returns are visible in per-seq tok/s dropping from 10.4 at batch=128 to 5.1 at batch=256 — half the per-sequence throughput for no aggregate gain.

## Next up

1. Build `burl/modal/gemma_local_batched.py` driving `BatchGenerator` from the rollout harness — the actual vehicle for applying this throughput.
2. Sibling `burl/eval/run_move4_star_rollout_batched.py` that consumes it.
3. Smoke that on 8 real decisions vs single-stream, match semantically (sampling is stochastic; pin tool-call counts and commit rates, not byte strings).
4. Layer `prompt_caches` reuse on top once the harness is working — expected multiplier on this already-large speedup.

## Files

- **Bench**: `burl/eval/bench_batch_throughput.py` (this doc's reproducer).
- **Target harness files** (to be built): `burl/modal/gemma_local_batched.py`, `burl/eval/run_move4_star_rollout_batched.py`.
- **Background docs**: `burl/modal/gemma_local.py` (single-stream original, 43 tok/s baseline noted in header), `burl/ITER4_PLAN.md` §M5-Max-specific considerations (predicted the throughput unlock; now measured).
