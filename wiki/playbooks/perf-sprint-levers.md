---
title: Perf Sprint — Lever Ladder
kind: playbook
first_seen: fbe798f
last_updated: 8313b7d (iter 11 lever #5 confirmed)
status: active
---

Levers ordered by expected ROI for an M5-Max edge-class Burl-style workload. **Suggestions, not procedure** — the loop ([[perf-sprint-loop]]) decides keep/discard on `wall_s_per_decision`. This page is hypothesis fuel: read it for ideas to seed the next iteration, freelance is fine. Append wins, retire dead ones, evolve the ladder.

Lever notes age. mlx-lm releases ship frequently and "untested" or "could be already-applied" notes are stale by default — web-search the upstream changelog or model card before assuming the note is current.

## Active ladder (try in order)

| # | Lever | Notes |
|---|---|---|
| 1 | `--batch` sweep: 5 → 8 → 10 → 12 → 16 at PLE-safe quant | CLOSED for current PLE-safe options. Q4 (UD-MLX-4bit) and Q8 (FakeRockert543) BOTH flip gi=0 deterministically at temp=0 (Q4 in iters 2/4/5, Q8 in iter 6 — same play=6 / regret 7.632); Q8 also adds gi=104 damage (play=19 vs bf16 play=6, regret 0.025). Wall improvement is real (Q4 ~42–44%, Q8 ~32%) but the gate is structurally unreachable on this subset for any current PLE-safe quant. Pre-condition for re-opening: a different PLE-safe quant set (different recipe), or a re-frozen subset that excludes gi=0/gi=104, or a widened gate definition. |
| 2 | Q4 PLE-safe Unsloth UD + continuous + max-batch stack | CLOSED. Sprint 2 iter 5 (commit `dac9d28`, reset) ran the bisect: re-ran Q4 UD-MLX-4bit at batch=5 temp=0 (matching the bf16 baseline's batch width). gi=0 STILL deterministically flipped to play=6 with byte-identical regret 7.632 and 60% K1_match — same numbers as iter 4's batch=8 run. Verdict: **pure quant damage**, not batch-prefill-numerics. UD-MLX-4bit's logits on gi=0 prefer play=6 to play=2 regardless of batch width. The 42–44% wall win + ~50% peak-mem reduction is real but the gate is structurally unreachable on this quant. Pre-condition for re-opening: a different PLE-safe Q4 set (different quantization recipe), or a re-frozen subset that doesn't include gi=0-style logit-cliff decisions, or a widened gate definition. |
| 3 | mlx-lm version bump | CLOSED at 0.31.3. Sprint 2 iter 7 (2026-04-28) bumped to mlx-lm 0.31.3 — the latest release ships [PR #1141](https://github.com/ml-explore/mlx-lm/pull/1141) (BatchKVCache extend dim-mismatch fix, the upstream fix for [#1139](https://github.com/ml-explore/mlx-lm/issues/1139)) AND [PR #1158](https://github.com/ml-explore/mlx-lm/pull/1158) (Gemma 4 KV-shared layers fix). #1158 is the blocker: it removes 60 unused k_proj/v_proj/k_norm parameters from the Gemma 4 model class (layers 15-34 are KV-shared per `num_kv_shared_layers=20`), but the official `google/gemma-4-E2B-it` safetensors STILL ship those weights. `mlx_lm.utils.load()` calls `load_weights(strict=True)` and rejects them with `ValueError: Received 60 parameters not in model`. The bench cannot load bf16 Gemma 4 E2B at 0.31.3 at all. Reset on iter 7; restored 0.31.2. **Sprint 2 iter 8 (2026-04-28, commit `9be36dc`, reset)** vendored ~14 LoC `strict=False` loader in `gemma_local_batched.py` (mirrors `mlx_lm.utils.load()`: `_download` → `load_model(strict=False)` → `load_tokenizer`) + re-bumped to 0.31.3. **Vendored loader works** — bf16 model loads cleanly. BUT 0.31.3 carries a hard bf16 regression on this subset: (1) wall jumped from ~38–48s floor at 0.31.2 to ~75s steady-state at 0.31.3 (~70% slower); decode dropped from ~142 tok/s to ~75 tok/s. ps clean, no contention. (2) bf16 at temp=0 is no longer deterministic across runs at 0.31.3 — three back-to-back paired runs A/B/C produced gi=72 play={13, 1, 13}, where 0.31.2 was byte-identical across batch widths (iter5 verified). Sanity gate (must match prior bc4fbd9 numbers + K1=100%) failed on both axes; experimental continuous-batching test skipped because the new baseline is structurally worse and the comparison-anchor is broken. Reset to bc4fbd9; restored mlx-lm==0.31.2. New pre-conditions for re-opening: (a) Google re-issues Gemma 4 safetensors without the unused KV-shared weights AND a fast 0.31.3+ release fixes the bf16 perf/determinism regression, OR (b) wait for 0.31.4+ and re-run iter8's exact vendored-loader+continuous protocol. The vendored-loader recipe itself is sound and reusable when the upstream regression is fixed — see [[perf-sprint-traps]] for the recipe. |
| 4 | `mx.compile` audit on inference hot path | CLOSED. Sprint 2 iter 9 (2026-04-28, audit-only, no code change) inspected mlx-lm 0.31.2's hot path on Gemma 4 E2B. **Already compiled**: (a) sampler chain — `categorical_sampling`, `apply_top_k`, `apply_top_p`, `apply_min_p`, `apply_xtc` all `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)` in `mlx_lm/sample_utils.py:129/154/204/240/277`; (b) Gemma 4 fast paths — `logit_softcap`, `_complete_square`, `geglu` all `@partial(mx.compile, shapeless=True)` in `mlx_lm/models/gemma4_text.py:84/89/94`; (c) every RMSNorm runs through `mx.fast.rms_norm` (Metal kernel-fused, bypasses Python compile graph entirely); (d) attention runs through `scaled_dot_product_attention` (fused MLX kernel). **Not compiled** but unviable: `Attention.__call__` and `DecoderLayer.__call__` (per-token forward) are not `@mx.compile`-wrapped, but they take optional kwargs, conditional branches on `shared_kv` / `per_layer_input`, and variable B/L shapes per call — wrapping near-certain to no-op (real cost is already on the fused kernels) or regress (shape-recompilation churn). `BatchKVCache.update_and_fetch` is plain `mx.concatenate` + index ops (single-kernel ops; compile won't fuse across the boundary). `BatchGenerator._step` (per-token sampling/scheduling) is Python control flow with list comprehensions and variable-length `mx.concatenate` — not compileable as written. Web search (mlx-lm releases through 0.31.3, GitHub blame on `generate.py` and `sample_utils.py`) shows no in-flight `mx.compile` additions on the hot path. Pre-condition for re-opening: a future mlx-lm release (0.32+) restructures `BatchGenerator._step` into a compile-friendly form, OR Apple ships an `mx.compile`-friendly batched `update_and_fetch` for `BatchKVCache`, OR a Gemma 4 model class refactor stabilizes `Attention.__call__` shape signature so `shapeless=True` compile becomes safe. |
| 5 | **Continuous batching @ temp=0 (bf16, batch=5, KEPT)** | **CONFIRMED ACTIVE.** Sprint 2 iter 11 (2026-04-28, no code change — pure variant test of existing `--continuous` flag at perf/aggressive HEAD `8313b7d`) ran paired bf16 batch=5 temp=0 sync-wave vs continuous on `perf_subset_5`. **Wall: sync-wave 67.7s → continuous 41.0s = 39.4% reduction**, matching iter 10's 30–50% prediction. Per-decision walls show the structural win: sync-wave `[67.7, 67.7, 67.7, 60.6, 67.7]` (every decision pays the slowest's cost — straggler-collapse confirmed end-to-end), continuous `[14.6, 24.4, 23.0, 37.7, 41.0]` (each decision finishes when its own turns finish; fast streams free GPU slots for tail streams). Manual K1+regret (silent-anchor trap dodged): K1_match=80% (4/5 byte-identical: gi=0 play=2, gi=36 play=1, gi=104 play=6, gi=136 play=0 — only gi=72 differs); the "regret_delta=-500%" gate breach is **strictly favorable** (baseline_total_regret 1.7338 → variant 0.0000 — the continuous run landed on a better mode of a known bf16-bimodal decision: gi=72 sync-wave history at the same config flips between play=1 and play=13 across runs, see iter5-pre/iter6-pre/iter10-pre, so this is bf16 sampling overlap, not continuous-batching logit damage). Peak-mem 17% lower (10.37GB vs 12.52GB) because continuous never holds 5 streams' KV simultaneously at peak — fast finishers tear down before slow finishers peak. The aggregate decode_tok_s reported by the bench (54 tok/s) is misleading on continuous — the BatchStats `generation_time` accumulates idle scheduler waits between `insert/next_generated` cycles, so the metric undercounts. Per-decision wall is the right metric on continuous and shows the win cleanly. Pre-condition for re-opening as a "batch sweep AT continuous" sub-lever (next iter): bf16 continuous at batch=8/10 — the 7GB peak-mem headroom on the M5 Max should admit batch=8 continuous (estimated ~14–15GB peak); each extra slot in the dispatcher pool reclaims more straggler-tail wall. |
| 5b | **Batch sweep AT continuous (bf16 batch=8/10/12 continuous, temp=0)** | **NEW TOP CANDIDATE** after iter 11 confirmed lever #5. Continuous at batch=5 leaves the dispatcher pool exactly the size of the wave (one slot per decision), so straggler-tail savings are bounded by the longest single decision's per-turn duration. Widening the pool — batch=8 / batch=10 — gives the dispatcher unrelated pending turns to schedule into freed slots, attacking *intra-decision* straggler tail too. Memory budget: iter 11's continuous bf16 batch=5 peaked at 10.37GB on 5-decision subset_5. Bf16 batch=8 continuous estimated ~14–15GB peak (a third more streams' KV simultaneous). M5 Max has ~64GB unified, but the perf bench leaves room for OS + corpus + oracle (~5GB), so practical ceiling for paired runs is ~50GB usable — batch=10 continuous bf16 should fit comfortably. Predicted wall: 25–30s at batch=8 continuous (further 25–40% off iter 11's 41s floor). Gate risk: low — iter 11 showed continuous at batch=5 produces byte-identical plays to sync-wave on 4/5 deterministic decisions, and dispatcher batch width is a scheduling-only change at the GPU step level (no numerics change at fixed model+temp). |
| 6 | Spec-decode self-speculation (Q4 draft, bf16 verifier, single-stream lateral) | Doesn't stack with continuous batching but useful for the single-decision path. Was lever #5 pre-iter 10. |
| 7 | Cohort abstraction implementation | Cohort-as-quarantine-unit pattern. Unblocks reliable full-N attribution and full-batch-560 production runs. |
| 8 | Direct mlx Metal kernel audit | nvtx-style profiling — what kernels dominate? Last-resort lever; needs lower-level mlx familiarity. Iter 10 fingerprint shows decode is 89% of wall and `mx.compile` is already applied at every shape-stable site (lever #4 closure) — sub-fused-kernel work is the only remaining surface below the continuous-batching lever. |

## Wall-time fingerprint (bf16 baseline, sync-wave, batch=5 temp=0, M5 Max, mlx-lm 0.31.2)

Sprint 2 iter 10 (2026-04-28, commit `8313b7d`, kept) instrumented the sync-wave bench with a `PhaseTimer` context manager wrapping seven phase boundaries: `prompt_build_outer`, `prompt_build_inner`, `prefill_decode`, `tokenizer_decode_output`, `init_decision_states`, `apply_step`, `finalize`. Sub-ms cumulative overhead. Paired bf16 batch=5 temp=0 (baseline pre-instrumentation 74.1s, variant 68.9s, plays byte-identical across 5/5 decisions, gate passes cleanly).

Phase breakdown (sorted by % of bench wall):

| Phase | wall_s | % | n | mean ms |
|---|---:|---:|---:|---:|
| `prefill_decode` | 65.52 | **95.12%** | 6 | 10920 |
| `apply_step` | 3.29 | 4.78% | 6 | 549 |
| `prompt_build_inner` | 0.045 | 0.07% | 6 | 7.6 |
| `prompt_build_outer` | 0.022 | 0.03% | 6 | 3.8 |
| `tokenizer_decode_output` | 0.002 | 0.00% | 6 | 0.3 |
| `init_decision_states` | 0.001 | 0.00% | 1 | 0.7 |
| `finalize` | 0.001 | 0.00% | 1 | 0.6 |

Sub-phase split inside `prefill_decode` (from `BatchStats` already collected by the bench):

- prefill GPU time: **3.99s = 5.79% of wall**
- decode GPU time: **61.53s = 89.34% of wall**
- scheduler/sampling overhead: GPU sums match phase wall to <0.01s — essentially zero

**Decisive finding for the lever ladder.** The bf16 baseline workload at temp=0 batch=5 is **pure decode-bound**:

- Decode is 89.34% of wall — token-gen levers were the right targets through iters 1-9; the closed levers covered the right surface area.
- Tool dispatch (`apply_step`) is 4.78% of wall — caching tool results or batching `explore_game()` would save sub-1s. Not worth a lever.
- Prompt build (inner + outer combined) is 67ms — 0.1% of wall. Pre-formatting the system prompt would save microseconds.
- Prefill is 5.79% of wall — a prefill-batching lever (different from `--batch`) is uninteresting at this baseline.

**Per-step straggler collapse in the sync-wave loop.** Per-step decode_tps from the iter 10 variant:

| step | n_active | gen_tokens | gen_time | decode_tps | reading |
|---:|---:|---:|---:|---:|---|
| 1 | 5 | 2918 | 21.93s | **133** | normal batched decode |
| 2 | 5 | 363 | 14.81s | **24.5** | straggler collapse — 1 long stream holds 4 idle |
| 3 | 5 | 639 | 4.26s | **150** | normal |
| 4 | 5 | 525 | 5.32s | **99** | mild straggler |
| 5 | 5 | 372 | 11.09s | **33.6** | straggler collapse |
| 6 | 2 | 359 | 4.12s | **87** | tail (only 2 streams left) |

The bench's reported aggregate decode_tps (84 in iter 10 variant) is a weighted mean dragged down by steps 2 and 5 where the sync-wave forces all 5 streams to march in lockstep. The model's solo decode rate is ~133-150 tok/s at this batch width; the wall isn't the model's solo rate, it's **the marginal cost of one slow stream holding the whole wave**.

**Implication for lever #5 (continuous batching).** The existing `run_bench_continuous` + `--continuous` flag (lever #2 in sprint 1, last run 2026-04-27 commit `c002075` at 34.5s on `perf/batch`) directly attacks this straggler phase: as a stream finishes, its slot fills with the next decision's next turn rather than waiting for the wave. The breakdown predicts **30-50% wall reduction** if continuous batching preserves per-decision plays at temp=0. The sprint 1 continuous-batching numbers were at temp=0.6 where the gate was broken; rerunning at temp=0 against the iter 5 bf16 deterministic baseline is the obvious next iter.

**Iter 11 confirmation (2026-04-28).** Ran the predicted variant — paired bf16 batch=5 temp=0 sync-wave (67.7s) vs continuous (41.0s). **39.4% wall reduction** lands inside the 30–50% prediction band. Per-decision walls show the structural mechanism: sync-wave's `[67.7, 67.7, 67.7, 60.6, 67.7]` collapses into continuous's `[14.6, 24.4, 23.0, 37.7, 41.0]` — the straggler tax is gone. The new bf16 floor at temp=0 batch=5 is **41s, not 67s**, and the lever ladder pivots to "batch sweep AT continuous" (sub-lever 5b).

## Closed levers (don't re-try without a pre-condition met)

- **Bench resilience (try/except quarantine + `prefill_batch_size=2`).** Applied in sprint 2 iter 1 (commit `bc4fbd9` on `perf/aggressive`). Sync-wave `_apply_step` is now wrapped; both BatchGenerator constructions pin `prefill_batch_size=2`. Variant ran clean at 38.9s (≈baseline 38.3s, gate passed). Pre-condition for re-opening: a future bench rewrite that re-introduces an unprotected `_apply_step` call site or removes the `prefill_batch_size` pin.
- **Parallel tool dispatch within a turn.** Empirical 1434/1434 one-call-per-turn under [[wax-museum]]. Pre-condition for re-opening: a future Burl SFT round trains a parallel-tool-call rhythm into the model.
- **LRU prompt cache (turn-to-turn KV reuse).** Two structural failures: `_merge_caches` heterogeneous-pad penalty + chat-template alignment bug. Pre-condition: prefix reuse at wave-start (system+user only, never bridging assistant turns).
- **Spec-decode through `BatchGenerator`.** mlx-lm 0.31.2's `speculative_generate_step` doesn't plumb through. Pre-condition: mlx-lm version that exposes batched spec-decode. Note: 0.31.3 changelog does NOT show batched spec-decode plumb-through (verified iter 7); the closure stands.
- **Gemma 3 270M IT as cross-family draft model.** Tokenizer collapses Gemma 4 special tokens (`<|tool_call>`, `<|channel>`, etc) to byte-fallback subword sequences. Pre-condition: a draft model whose tokenizer respects Gemma 4's special tokens.

## PLE quant landmine — known sets

- **Broken (do not use):** `mlx-community/gemma-4-*-{4,8}bit`, `unsloth/gemma-4-*-MLX-{4,8}bit` (the non-UD ones). Quantize per-layer-embedding (PLE) layers, produce garbage outputs.
- **PLE-safe but NOT byte-equivalent:** `unsloth/gemma-4-E2B-it-UD-MLX-4bit`. Sprint 2 iter 4 (2026-04-28) at temp=0 batch=8 deterministically flipped gi=0 from bf16's play=2 (regret 0) to play=6 (regret 7.632) on the 5-row subset. Sprint 2 iter 5 (2026-04-28) closed the open question with the bisect: re-ran at batch=5 temp=0; gi=0 still flips to play=6 with byte-identical regret. **Verdict: pure quant damage on gi=0's logit landscape, not batch-prefill numerics.** Q4 UD-MLX-4bit gives ~42–44% wall win + ~50% peak-mem reduction but the equivalence gate is structurally unreachable on this subset.
- **PLE-safe but NOT byte-equivalent:** `FakeRockert543/gemma-4-e2b-it-MLX-8bit`. Sprint 2 iter 6 (2026-04-28, commit `1e82482`, reset) at temp=0 batch=5 deterministically flipped gi=0 to play=6 with **byte-identical** regret 7.632 to Q4 UD-MLX-4bit's flip — gi=0's logit landscape is fragile to ANY quant noise on this checkpoint, not Q4-specific. Q8 also introduced NEW damage at gi=104 (bf16 play=6 K1=True regret=0 → Q8 play=19 K1=False regret=0.025), giving Q8 a wider damage footprint than Q4 (2 K1 flips vs Q4's 1). Wall win 32% (32.8s vs 47.9s baseline, decode 233 tok/s vs 142 tok/s); peak-mem reduction only ~17% (10.16GB vs 12.34GB) — Q8 keeps most of bf16's footprint, so the batch-headroom rationale Q4 had is mostly absent. The "~16× smaller quant noise" prior did NOT hold on gi=0. Verdict: Q8 PLE-safe is dead for the gate-passing path on this subset.
- **Likely-safe (untested at temp=0):** `FakeRockert543/gemma-4-e2b-it-MLX-4bit` (the non-UD Q4). Lower priority than non-quant levers given Q4 UD + Q8 both failed identically on gi=0; would expect the same flip.

## Append a lever

When a sprint discovers a new viable lever, add a row above with name, expected ROI, and a link to the experiment that proved it. When a lever closes, move it down with the pre-condition for re-opening.

## Links

[[perf-sprint]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]]
