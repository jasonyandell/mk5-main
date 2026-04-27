---
title: Perf on the table
kind: topic
first_seen: 74464e9
last_updated: 1f11d28
status: active
---

## Calibration

Gemma 4 E2B is a mobile-class small model. `mlx_lm.batch_generate` on M5 Max with real Burl prompts hits **1334 tok/s aggregate at batch=128** ([[batch-throughput-bench]]); single-stream runs at **43 tok/s**. The eval harness, however, sits at **18.7 s/decision** at batch=6 (n=180 base eval, 2026-04-26) — and the sequential reference was **26.3 s/decision** ([[burl-star-run3]] run-3c eval at n=560). At a mean ~6.2 turns/decision and ~200 tokens/turn, the harness's effective throughput is ~70 tok/s per decision in the batched path and ~50 tok/s sequential. **The harness is operating ~20× under what the same model + same library can deliver in raw inference.**

This isn't a model problem. It's harness overhead — chat-template rendering on every turn, CPU-bound tool dispatch in series, no prefix sharing, no continuous batching, generous per-turn token caps, sync-wave straggler tax. None of it is research; all of it is engineering.

## Six levers, ranked by ROI

Estimates are best-guess for M5 Max with the iter-3-rules-shape prompt distribution. Real numbers will diverge; the order is more reliable than the magnitudes.

1. **Prefix sharing on prefill — 1.5–2× expected.** Each batched wave shares ~1500–2500 tokens of system prompt + rules primer + game-state preamble across N decisions, but currently re-prefills them N times. `mlx_lm.batch_generate` may already do prefix-cache-aware prefill; verify. If not, hand-roll the shared prefix once and resume from a shared KV cache. Cheapest lever; just an instrumentation pass + a flag.
2. **Continuous batching — 3–5×.** The current sync-wave loop ([[batched-eval-resilience]]) waits for the slowest decision in each wave to finish before starting the next batch. The 50-min wave incident on the live n=180 was exactly this — 4 of 6 decisions committed in 5 turns, two ran to 14 turns, the GPU sat idle on the tail. Continuous batching slots fresh decisions into freed positions as decisions terminate; mlx-lm's async path supports this but the harness doesn't use it. Biggest single ROI.
3. **Smaller per-turn token budgets — 1.3–1.5×.** `max_tokens=8192` is the eval default; harvest defaulted to 2048 ([[max-tokens-2048-floor]]). Per-turn p95 generation length is ~600 tokens. A turn-aware budget (small for the early "look up belief" turns, larger reserved for the final commit reasoning) recovers most of this. Easiest to ship behind a flag; most defensible because it doesn't change semantics, just stops paying for unused capacity.
4. **Speculative decoding — 2–4×.** Gemma 4 E0.5B as the draft model, the E2B as the verifier. Tool-call-heavy outputs (structured `<|tool_call>...{}<tool_call|>` shapes) tend to have high acceptance rates because the surface form is templated. Requires running two models simultaneously which doubles memory pressure; needs measurement before scaling out.
5. **Tool call parallelization within a decision — 1.5–2×.** Gemma 4's native chat template supports parallel tool calls in a single assistant turn (multiple `<|tool_call>` blocks). The current harness sequentializes them — the model emits parallel calls but the dispatcher serializes. Free turn-savings on the ~30% of decisions where the model emits 2+ tool calls in one turn (e.g., `belief_trajectory()` + `explore_game(X)`).
6. **Quantization (bf16 → INT4/INT8) — 1.2–1.5× speed + ~2× memory headroom.** Q4_K_M GGUF was the local-runner format earlier in the project ([[candlewax-spike-e2e]]); switching the eval/harvest path to it would free batch budget for a higher batch-size (where the bench shows 16× aggregate). Lower priority because the speed delta alone is small; the memory headroom unlocking larger batches is the actual win, and that's already covered by levers 1–2.

## Compounded realistic stack

Stacking the top three (prefix sharing × continuous batching × turn-aware budgets) is roughly multiplicative on the GPU-bound portion of the wall: **~7–10× on M5 Max alone, no model changes.** The [[burl-harvest-2]] budget that cost ~5h overnight would land in ~30–45 min. None of these levers requires a Modal multi-GPU spend; M5 Max stays the production host.

The remaining three (speculative decoding × parallel tool calls × quantization) compound to another ~3–5× when the harness can absorb the complexity. The end-state — same model, same hardware, same corpus — is plausibly **~20–40× over today's harvest**. That's the gap the calibration above flagged: we're not bottlenecked on model capacity; we're paying for a harness that was written for correctness first and never revised for throughput.

## Measurement harness

Phase 0 of the sprint shipped at `1f11d28`: a tracked bench that drives
the production batched eval path against a frozen 5-decision subset and
records per-decision wall, prefill/decode tok-s, peak memory, plus a
K1-grade-match-pct vs the latest baseline-bf16 row.  Everything from
here forward is measured against this bench — Phase 1 (cheap wins),
Phase 2 (continuous batching + prefix sharing), and Phase 3 (speculative
decoding + quantization) thread their levers through `--variant <name>`
and write a row each.

Canonical baseline-bf16 (M5 Max, batch=5, max_tokens=8192, temp=0.6,
sha `1f11d28`):

- `wall_s_total` ≈ 79.5 s for 5 decisions — about 16 s/decision.
- `decode_tok_s` ≈ 87 per stream; `prefill_tok_s` ≈ 10,300.
- `peak_mem_gb` = 11.59 (deterministic to 4 dp run-vs-run).

Stability budget at temp=0.6: total wall reproduces to 0.5%, decode
tok/s to 9%, K1 grade to 80–100% on the 5-row subset.  The 5-row
floor isn't tight enough to confirm sub-10% regret deltas; that's
why the bench also accepts `--subset 560` for the phase-exit gate.

Where the artifacts live:

- `burl/eval/bench_decision_latency.py` — the CLI.
- `burl/eval/data/perf_subset_5.jsonl` — the frozen subset (5 rows +
  header with `corpus_eval_20.pt` SHA256 + per-decision fingerprints).
- `burl/eval/results/perf_ledger.csv` — append-only ledger row per run.
- `burl/eval/results/perf_<ts>_<variant>.json` — full per-step detail.
- `burl/eval/gus_eval_bridge.py` — promoted from
  `scratch/belief_trajectory_rollout/diagnostic/` so tracked benches can
  resolve `global_idx → BurlDecision` without sourcing from scratch.

Full setup, subset rationale, and noise-floor read at
[[burl-perf-phase0]].

## Why it matters for STaR planning

A 12h iter wall and a 1.5h iter wall are different regimes, not the same regime with a faster clock. At 12h, breadth experiments are expensive — running the [[backwards-curriculum]] scout that rolls out from trick 5/0 is a one-shot bet. At 1.5h, the scout becomes a routine sanity check and curriculum-transfer becomes a multi-condition sweep. The same is true of [[r1-rationalization]]'s verifier loop, the rank-16 ablation ([[burl-star-run3]] task #9), and any of the prompt-variant sweeps documented in [[star]].

The work itself is engineering, not research — each lever is a 2–5 day sprint with a measurable bench delta. Worth one focused sprint before the next major STaR iteration ([[burl-harvest-2]] onward) so the throughput floor stops being the constraint that picks our experiment portfolio for us.

## Related pages

[[burl-perf-phase0]] · [[batch-throughput-bench]] · [[batched-eval-resilience]] · [[batched-harvest-resilience]] · [[max-tokens-2048-floor]] · [[burl-star-run3]] · [[backwards-curriculum]] · [[mlx-lm]] · [[modal]] · [[candlewax-spike-e2e]] · [[star]]
