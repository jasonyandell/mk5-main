---
title: "MLX batch_generate Ceiling on M5 Max"
kind: experiment
first_seen: ed3cfc3
last_updated: 6a97d55
status: active
---

## Summary

`mlx_lm.batch_generate` benchmarked on real Burl prompts (mean 2378 tokens, iter-3-rules shape). Peak 1334 tok/s at batch=128 vs 43 tok/s single-stream — 16× aggregate speedup. Operationalized in a batched rollout harness achieving 2.3× wall time on N=16.

([burl/experiments/batch_throughput_bench.md @ ed3cfc3](../sources/ed3cfc3.md))

## Setup

- **Hardware:** M5 Max, 48 GB unified memory
- **Prompts:** real Burl prompts, mean 2378 tokens, iter-3-rules shape
- **Batch sizes swept:** 16..256 with 90%-of-peak stop

## Results

| Batch size | Throughput | Notes |
|---|---|---|
| 1 (sequential) | 43 tok/s | gemma_local.py baseline |
| 64 | 1206 tok/s | 90% of peak |
| 128 | **1334 tok/s** | Peak |
| >128 | plateaus | Memory plateau at 15 GB |

## Operationalized (6a97d55)

`GemmaLocalNativeBatched` + `run_move4_star_rollout_batched.py`: N=16 batched (batch=16) runs in 58s vs sequential 134s — 2.3× wall. Larger batches and `prompt_cache` reuse expected to push toward the bench's 14.5× aggregate win.

## Significance

N=500 rollouts at ~4 turns × ~128 tokens = ~3.5 min wall time vs hours sequential. Corpus scale stops being a throughput constraint. Unlocks "corpus 10-20× larger" as the next iter-5+ lever.

**Load-bearing incidental finding** (flagged in [[sources/7321952]]): `WorldSamplerMRV` marginal distribution is biased vs uniform enumeration by ~6.8 Q points at trick 6. Enumeration is ground truth; all historical Burl eval numbers and forge/eq training data use the biased sampler. Worth a bead.

## Related pages

[[mlx-lm]] · [[burl]] · [[selfplay-arena]] · [[sources/ed3cfc3]] · [[sources/6a97d55]]
