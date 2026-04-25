---
title: MLX-LM (Apple Silicon local path for Burl)
kind: entity
first_seen: 6fea6ab
last_updated: 6a97d55
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
