# burl-perf-phase3 audit — 2026-07-07

## Corrections

- Page's "Bench rows in the ledger" snippet said timestamps 20260427_030755 / 030849 / 031040 for continuous-paired-stack2, phase3-stack-best-v2, and -v3; the ledger says 030558 / 030708 / 030825 (evidence: burl/eval/results/perf_ledger.csv, burl/eval/results/perf_20260427_030825_phase3-stack-best-v3.json).
- Pointers section listed quant model repos as `FakeRocket543/gemma-4-e2b-it-MLX-{4,8}bit`; the models live under `FakeRockert543/...` (extra 'r'), as the page's own Dragon #2 documents (evidence: HF cache dirs models--FakeRockert543--gemma-4-e2b-it-MLX-4bit / -8bit).

## Verified

- All wall/decode/peak numbers in the tradeoff matrix and ledger snippet match burl/eval/results/perf_ledger.csv exactly (34.1/62.2/9.17, 27.4/90.8/9.83, 44.1/61.3/6.24, 48.7/66.1/5.08, 43.0/61.9/6.07, etc.).
- burl/eval/bench_decision_latency.py and burl/modal/gemma_local_batched.py exist as described.
- q4-unsloth JSON confirms model_repo unsloth/gemma-4-E2B-it-UD-MLX-4bit, peak 6.24 GB, 5-decision run at temp=0.

## Not verifiable in-repo

- Paired play-match counts (4/5, 3/5) and Q-pts regret deltas quoted per-variant come from cross-run comparison of the per-decision JSONs; per-decision grades exist in the JSONs but the paired-play tallies were not independently recomputed in this pass.
- HF discussions, LM Studio / mlx-lm GitHub issues, and tokenizer-probe raw output live outside the repo.

## Follow-ups

- The "3.4× run-to-run wall variance" framing predates the later finding (perf-subset-5 noise floor) that the variance was GPU contention with parallel scribes; a one-line cross-reference on the page would keep future readers from over-trusting the paired-only protocol rationale.
