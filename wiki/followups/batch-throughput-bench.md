# batch-throughput-bench — audit 2026-07-07

## Corrections

- Page said the 16× speedup was vs 43 tok/s single-stream; the bench's measured single-stream baseline is 83 tok/s (16× is relative to 83; 43 tok/s was only the old `gemma_local.py` header figure) (evidence: burl/experiments/batch_throughput_bench.md, "Single-stream baseline ... 83 tok/s").
- Page said batch sizes swept were "16..256 with 90%-of-peak stop"; the sweep was 1..256 with no stop rule — "90% of peak" describes the batch=64 recommended default, not a stopping criterion (evidence: burl/experiments/batch_throughput_bench.md table).
- Page said the >128 plateau was a "memory plateau at 15 GB"; the source attributes it to memory-bandwidth-bound generation, and peak memory keeps rising (15.5 GB at 128, 19.2 GB at 256) (evidence: burl/experiments/batch_throughput_bench.md, "Why the plateau above batch=128").
- ~~Page said the operationalized harness ran N=16 at batch=16; the commit message says batch=64 ("bench knee-of-curve")~~ — **retracted on second pass**: the original page was right. The full 6a97d55 commit message says "Wall: N=16 batched batch=16 58s vs sequential concurrency=1 local 134s", and `burl/experiments/batch_rollout_harness.md` records the run as `--batch-size 16`; batch=64 is only the harness *default*. The truncated quote in wiki/sources/6a97d55.md misled the first pass.

## Follow-ups

- The "14.5× aggregate win" extrapolation still cites the batch=64 vs 83-tok/s ratio; a cheap next probe is measuring the ragged-batch delta the source doc flags as unmeasured. (The prompt_caches half of this probe is already done: c0020751 tested LRUPromptCache through batch_generate and it came back negative — 1.9× slower, K1 grade match 60% — see wiki/experiments/burl-perf-phase2.md.)
- The WorldSamplerMRV ~6.8-Q-point bias gap is still untracked — filing a GitHub issue (milestone "Champion") remains the outstanding action the page itself calls out. (Re-verified 2026-07-07: `gh issue list --state all --search WorldSamplerMRV` returns nothing; issue #33's body has no sampler/bias mention.)

## Review (second pass, 2026-07-07)

- Corrections 1-3 (83 vs 43 tok/s baseline, 1..256 sweep with batch=64 as recommended default not a stop rule, memory-bandwidth-bound plateau with 15.5→19.2 GB) verified against `burl/experiments/batch_throughput_bench.md` — they stand.
- **Reverted correction 4**: the first pass changed the operationalized run from "(batch=16)" to "(batch=64)" citing wiki/sources/6a97d55.md, but the digest's commit quote is truncated. The full commit message (`git show 6a97d55`) says "Wall: N=16 batched batch=16 58s vs sequential concurrency=1 local 134s", and `burl/experiments/batch_rollout_harness.md` lists the run command as `--batch-size 16`. Restored batch=16 with a parenthetical noting the harness default is 64.
- **Fixed a staleness miss**: the page (both before and after the first pass) said "prompt_cache reuse ... untested". It was tested 2026-04-27 in commit `c0020751` ("lever-1 LRUPromptCache experiment + negative result": 136.7s vs 71.0s baseline, K1 grade match 60%), documented in `wiki/experiments/burl-perf-phase2.md`. Page now records the negative result and links [[burl-perf-phase2]]. Ragged-batch delta remains unmeasured as stated.
- Follow-up suggestion about probing prompt_caches deltas amended accordingly (already done, negative); WorldSamplerMRV-untracked suggestion re-verified and kept.
