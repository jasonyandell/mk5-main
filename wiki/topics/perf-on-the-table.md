---
title: Perf on the table
kind: topic
first_seen: 74464e9
last_updated: 29da3d2
status: active
---

## Calibration

Gemma 4 E2B is a mobile-class small model. `mlx_lm.batch_generate` on M5 Max with real Burl prompts hits **1334 tok/s aggregate at batch=128** ([[batch-throughput-bench]]); single-stream runs at **43 tok/s**. The eval harness, however, sits at **18.7 s/decision** at batch=6 (n=180 base eval, 2026-04-26) — and the sequential reference was **26.3 s/decision** ([[burl-star-run3]] run-3c eval at n=560). At a mean ~6.2 turns/decision and ~200 tokens/turn, the harness's effective throughput is ~70 tok/s per decision in the batched path and ~50 tok/s sequential. **The harness is operating ~20× under what the same model + same library can deliver in raw inference.**

This isn't a model problem. It's harness overhead — chat-template rendering on every turn, CPU-bound tool dispatch in series, no prefix sharing, no continuous batching, generous per-turn token caps, sync-wave straggler tax. None of it is research; all of it is engineering.

## Six levers, ranked by ROI

Estimates are best-guess for M5 Max with the iter-3-rules-shape prompt distribution. Real numbers will diverge; the order is more reliable than the magnitudes.

1. **Prefix sharing on prefill — closed on M5 Max ([[burl-perf-phase2]]).** mlx-lm 0.31.2 already exposes `prompt_caches` + `LRUPromptCache`; the trie-based prefix match works mechanically (~28% prompt-token hit on the 5-row subset) but **two structural failure modes** apply: (a) `BatchKVCache.merge` pads all streams to the longest cache, so decode tok/s drops on heterogeneous batches (independent of contention; provable from `mlx_lm/models/cache.py:1056-1085`); (b) the chat template re-extracts structured `tool_calls` + `reasoning_content`, but the bench stored decoded completions in plain `content`, so trie keys never align across turns (independent of contention; provable from `chat_template.jinja`). Clean re-run lands as a tie (+8.4% vs adjacent baseline, well inside noise; the original 1.9× contended slowdown was contention amplifying the merge-pad penalty). The "1.5–2× expected" estimate was over-optimistic for this harness shape. Useful prefix sharing for Burl is "shared system+user prefix across the wave's decisions, captured once per wave-start" (~10–15% of total prefill at the 5-decision subset), and that slots naturally into Lever 2's dispatcher rather than running as a standalone lever.
2. **Continuous batching — magnitude unresolvable on `perf_subset_5` ([[burl-perf-phase2]]).** Implemented at the bench layer as `run_bench_continuous` (`burl/eval/bench_decision_latency.py`) atop `mlx_lm.generate.BatchGenerator`. All decisions submit to a long-lived dispatcher; tools dispatch + state transitions happen on CPU as soon as a stream finishes, while other streams keep decoding. Clean alternating re-run (B-C-B-C-B-P-C, 2026-04-27) lands as a **statistical tie**: baseline mean 38.9 s vs continuous mean 38.1 s (−1.9%), pairwise deltas −31.7% / +41.1% / −9.1% with direction *alternating*. Continuous's wall variance is *wider* than baseline's (range 22 s vs 9 s) because the dispatcher genuinely changes batch composition step-to-step and surfaces kernel noise. The 1.8–2.1× contended-window reading was a contention artifact correlated with the variant, not signal. The structural argument for the lever — `BatchGenerator` is already a continuous scheduler we weren't using; the wrapper-side change is a real architectural improvement at zero cost — survives independent of any wall measurement. Cross-wave straggler savings only manifest at scale (560+ decisions); Phase-4's full-560 pass is the gate for any defensible magnitude. Production harvest migration deferred; needs a cohort abstraction over [[batched-harvest-resilience]]'s wave-sentinel plumbing. Designs at [[continuous-batching-dispatcher-design]] + [[harvest-cohort-abstraction]].
3. **Smaller per-turn token budgets — 1.3–1.5×.** `max_tokens=8192` is the eval default; harvest defaulted to 2048 ([[max-tokens-2048-floor]]). Per-turn p95 generation length is ~600 tokens. A turn-aware budget (small for the early "look up belief" turns, larger reserved for the final commit reasoning) recovers most of this. Easiest to ship behind a flag; most defensible because it doesn't change semantics, just stops paying for unused capacity.
4. **Speculative decoding — 2–4×.** Gemma 4 E0.5B as the draft model, the E2B as the verifier. Tool-call-heavy outputs (structured `<|tool_call>...{}<tool_call|>` shapes) tend to have high acceptance rates because the surface form is templated. Requires running two models simultaneously which doubles memory pressure; needs measurement before scaling out.
5. **Tool call parallelization within a decision — 1.5–2×.** Gemma 4's native chat template supports parallel tool calls in a single assistant turn (multiple `<|tool_call>` blocks). The current harness sequentializes them — the model emits parallel calls but the dispatcher serializes. Free turn-savings on the ~30% of decisions where the model emits 2+ tool calls in one turn (e.g., `belief_trajectory()` + `explore_game(X)`).
6. **Quantization (bf16 → INT4/INT8) — 1.2–1.5× speed + ~2× memory headroom.** Q4_K_M GGUF was the local-runner format earlier in the project ([[candlewax-spike-e2e]]); switching the eval/harvest path to it would free batch budget for a higher batch-size (where the bench shows 16× aggregate). Lower priority because the speed delta alone is small; the memory headroom unlocking larger batches is the actual win, and that's already covered by levers 1–2.

## Compounded realistic stack

Stacking the top three (prefix sharing × continuous batching × turn-aware budgets) is roughly multiplicative on the GPU-bound portion of the wall: **~7–10× on M5 Max alone, no model changes.** The [[burl-harvest-2]] budget that cost ~5h overnight would land in ~30–45 min. None of these levers requires a Modal multi-GPU spend; M5 Max stays the production host.

Phase 2 result update (post clean re-validation, 2026-04-27): lever 1 closed (structural mlx-lm 0.31.2 failure modes; clean re-run +8.4% vs adjacent baseline = tie).  Lever 2 magnitude on `perf_subset_5` is a statistical tie with the wave loop (mean −1.9%, pairwise direction alternates).  The structural Phase-2 contributions stand — mlx-lm internals writeup, lever-1 root-cause diagnosis, dispatcher + cohort design specs — but the **5-row bench cannot resolve the magnitude question** for either lever.  Phase 4's full-560 pass is the gate.  Until then the compounded realistic stack should plan for **Phase 2 = ~1.0× on the bench, with cross-wave straggler savings unlocked at production scale**: 4–6× total Phase 1+3 speedup, with Phase 2 contributing the *architectural* unblock for the harvest migration rather than a measurable bench delta.

The remaining three (speculative decoding × parallel tool calls × quantization) compound to another ~3–5× when the harness can absorb the complexity. The end-state — same model, same hardware, same corpus — is plausibly **~20–40× over today's harvest**. That's the gap the calibration above flagged: we're not bottlenecked on model capacity; we're paying for a harness that was written for correctness first and never revised for throughput.

## What mlx-lm 0.31.2 actually exposes

Audited 2026-04-27 during Phase 2.  Useful for any future scribe touching
the inference path:

**Public surfaces:**

- `mlx_lm.batch_generate(model, tokenizer, prompts, *, max_tokens, sampler, prompt_caches=None, return_prompt_caches=False, completion_batch_size=N, prefill_batch_size=N, prefill_step_size=N, ...)`.  The wrapper around `BatchGenerator` (`generate.py:1879`).  `prompt_caches` accepts a list of pre-computed per-stream KV caches; `return_prompt_caches=True` puts post-decode caches into `BatchResponse.caches`.
- `mlx_lm.generate.BatchGenerator(model, *, completion_batch_size, prefill_batch_size, prefill_step_size, max_kv_size, sampler, stop_tokens, ...)`.  The continuous scheduler (`generate.py:1486`).  Methods: `insert(prompts, max_tokens, caches=...)`, `insert_segments(segments=...)`, `next()`, `next_generated()`, `extract_cache(uids)`, `remove(uids, return_prompt_caches=...)`, `stats()`.
- `mlx_lm.models.cache.make_prompt_cache(model)` — model-aware fresh cache.  For Gemma 4 returns `[KVCache | RotatingKVCache, ...]` (one per layer; sliding-window layers get rotating).
- `mlx_lm.models.cache.LRUPromptCache(max_size, max_bytes)` — the prefix-aware trie used by `mlx_lm.server`.  `fetch_nearest_cache(model_key, tokens)` returns `(cache, remaining_tokens)`; `insert_cache(model_key, tokens, cache, cache_type=...)` adds a post-decode cache.
- `cache.can_trim_prompt_cache(cache)` / `cache.trim_prompt_cache(cache, n)` — cache-state surgery for cross-turn reuse.

**Constraints worth knowing:**

- `LRUPromptCache._trie` keys on a hashable `model` argument; `nn.Module` is **not** hashable.  Use a string key (`f"{repo}:{adapter}:{id}"`).
- `BatchKVCache.merge` pads all batched streams to `max(c.size() for c in caches)` (`models/cache.py:1056`).  Heterogeneous-cache batched decode pays for the worst stream's KV width.  This rules out the naive cross-turn cache-reuse pattern within a single batched call.
- `BatchGenerator.insert` requires non-empty input segments; passing `prompts=[[]]` for a 100%-cache-hit stream raises.  Server-side workaround: leave the last token of the prompt outside the cache key; mlx-lm splits prompt suffix vs the last-token "split" automatically (`generate.py:1638-1640`).
- `prefill_batch_size=8` (default) trips the 0.31.2 broadcast-shapes bug at batch≥14 with heterogeneous prompts (>2300-token variance).  Workaround: `prefill_batch_size=2` or `batch=8` ([[batched-harvest-resilience]]).
- The chat template's role normalization (`assistant` → `model`) and structured `tool_calls` extraction means raw decoded tokens don't round-trip through `apply_chat_template(messages)`.  A trie key built from `prompt_ids + tokenizer.encode(completion_text)` does **not** match the next turn's rendered token stream.  See [[burl-perf-phase2]] for the diagnosis.

**The right pattern for prefix reuse** (per `mlx_lm.server.py`): use
`insert_segments` to flag "stable boundaries" the cache should snapshot
(end of system+user, before assistant turn opens).  The server captures
the cache *at end-of-segment* via the prompt-response stream's
`end_of_segment` flag and inserts it into the trie keyed on the
*literal token sequence up to that boundary*.  Subsequent requests
that share that prefix get a clean cache resume.  Burl's harness can
do the same trick for the system+rules+user prefix that's identical
across decisions in a wave; cross-turn within a single decision is
not addressable through this API because the conversation grows with
raw decoded tokens that the chat template won't re-emit verbatim.

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
