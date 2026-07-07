# batch-throughput-bench — audit 2026-07-07

## Corrections

- Page said the 16× speedup was vs 43 tok/s single-stream; the bench's measured single-stream baseline is 83 tok/s (16× is relative to 83; 43 tok/s was only the old `gemma_local.py` header figure) (evidence: burl/experiments/batch_throughput_bench.md, "Single-stream baseline ... 83 tok/s").
- Page said batch sizes swept were "16..256 with 90%-of-peak stop"; the sweep was 1..256 with no stop rule — "90% of peak" describes the batch=64 recommended default, not a stopping criterion (evidence: burl/experiments/batch_throughput_bench.md table).
- Page said the >128 plateau was a "memory plateau at 15 GB"; the source attributes it to memory-bandwidth-bound generation, and peak memory keeps rising (15.5 GB at 128, 19.2 GB at 256) (evidence: burl/experiments/batch_throughput_bench.md, "Why the plateau above batch=128").
- Page said the operationalized harness ran N=16 at batch=16; the commit message says batch=64 ("bench knee-of-curve") (evidence: wiki/sources/6a97d55.md).

## Follow-ups

- The "14.5× aggregate win" extrapolation still cites the batch=64 vs 83-tok/s ratio; a cheap next probe is measuring ragged-batch and prompt_caches deltas the source doc flags as unmeasured.
- The WorldSamplerMRV ~6.8-Q-point bias gap is still untracked — filing a GitHub issue (milestone "Champion") remains the outstanding action the page itself calls out.
