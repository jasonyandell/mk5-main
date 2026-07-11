---
title: "MLX batch_generate Ceiling on M5 Max"
kind: experiment
first_seen: ed3cfc3
last_updated: 6a97d55
status: active
---

## Summary

`mlx_lm.batch_generate` benchmarked on real Burl prompts (mean 2378 tokens, iter-3-rules shape). Peak 1334 tok/s at batch=128 vs 83 tok/s measured single-stream — 16× aggregate speedup (the older documented `gemma_local.py` baseline was 43 tok/s). Operationalized in a batched rollout harness achieving 2.3× wall time on N=16.

([burl/experiments/batch_throughput_bench.md @ ed3cfc3](../sources/ed3cfc3.md))

## Setup

- **Hardware:** M5 Max, 48 GB unified memory
- **Prompts:** real Burl prompts, mean 2378 tokens, iter-3-rules shape
- **Batch sizes swept:** 1..256; batch=64 recommended as default (90% of peak throughput at half the wall-per-cycle)

## Results

| Batch size | Throughput | Notes |
|---|---|---|
| 1 (sequential) | 83 tok/s | measured single-stream baseline (old `gemma_local.py` header cited 43 tok/s) |
| 64 | 1206 tok/s | 90% of peak; recommended default |
| 128 | **1334 tok/s** | Peak, 15.5 GB peak memory |
| >128 | plateaus | memory-bandwidth bound; memory keeps rising (19.2 GB at 256) |

## Operationalized (6a97d55)

`GemmaLocalNativeBatched` + `run_move4_star_rollout_batched.py`: N=16 batched (batch=16 in the measured run; the harness default is batch=64) runs in 58s vs sequential 134s — 2.3× wall. Larger batches were expected to push toward the bench's 14.5× aggregate win — **ragged-batch deltas remain unmeasured**, and `prompt_cache` reuse was later tested and came back **negative** in this batched shape (1.9× slower, K1 grade match dropped to 60%; `c002075`, see [[burl-perf-phase2]]); the 2.3× figure is measured, the extrapolation toward 14.5× is not.

## Significance

N=500 rollouts at ~4 turns × ~128 tokens = ~3.5 min wall time vs hours sequential. Corpus scale stops being a throughput constraint. Unlocks "corpus 10-20× larger" as the next iter-5+ lever.

**Load-bearing incidental finding** (flagged in [[sources/7321952]]): `WorldSamplerMRV` marginal distribution is biased vs uniform enumeration by ~6.8 Q points at trick 6. Enumeration is ground truth; all historical Burl eval numbers and forge/eq training data use the biased sampler.

**This was never tracked.** "Worth a bead" was the recommendation at the time, but no
bead and no GitHub issue was ever filed for it — before or after the 2026-06
beads→GitHub-issues migration. This is a real, still-open project gap affecting every
historical Burl eval number and forge/eq training-data quality claim, not merely a
wiki staleness issue.

## Related pages

[[mlx-lm]] · [[burl]] · [[selfplay-arena]] · [[sources/ed3cfc3]] · [[sources/6a97d55]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 3 corrections applied in place and independently re-verified; second pass amended 2 (reverted one first-pass over-correction, added the prompt_cache negative result).

- The commit quote in [[6a97d55]] is truncated; the full message (`git show 6a97d55`) and `burl/experiments/batch_rollout_harness.md` confirm the measured run used batch=16 (batch=64 is only the harness default).
- WorldSamplerMRV bias gap re-verified as untracked 2026-07-07: `gh issue list --state all --search WorldSamplerMRV` returns nothing; issue #33 has no sampler/bias mention. Filing a GitHub issue (milestone "Champion") remains the outstanding action.
- Cheap next probe: measure the ragged-batch delta the source doc flags as unmeasured (the prompt_cache half is done and negative — see [[burl-perf-phase2]]).
