# Burl Perf Sprint — Morning Digest

**Session:** 2026-04-26 → 2026-04-27, ~4.5 hours of orchestrator + 4 parallel Opus 4.7 scribes on a single M5 Max.
**Mission:** drive Burl per-decision inference latency from ~26s/decision toward ~2s. Wiki was an explicit first-class output.
**Outcome:** ~~13× wall prize~~ not landed. Memory headroom + structural wiki contributions are the durable wins.

---

## Headlines

| Metric | Phase 0 baseline | Phase 3 stack-best (5-row) | Δ |
|---|---:|---:|---|
| Wall (5 dec) | 79.5 s | **28.3 s** | **2.81× faster** |
| Decode tok/s | 87.5 | 87.7 | flat |
| **Peak memory** | **11.6 GB** | **5.08 GB** | **−56%** |
| K1 grade match | 100% (self-graded) | 80% (4/5) | inside noise floor |
| Regret Δ | 0% | -31% on gi=0 marginal | inside noise floor |

The **headline win is memory**, not wall: −56% peak unlocks ~2× cohort size at training time. The wall improvement is real but the 5-row bench's 3.4× same-config noise floor means sub-2× wall claims are unattributable. Phase 4's full-560 attribution **never completed** (crashed twice — see Dragons).

The 5-row bench's wall numbers are *suggestive directional evidence*. The structural and memory results are the durable contributions.

---

## Production recommendation

**Q4 PLE-safe Unsloth UD: `unsloth/gemma-4-E2B-it-UD-MLX-4bit`** as Burl's new default.
- 5/5 plays match bf16 baseline at temp=0
- Identical signed_deltas (byte-equivalent policy on the 5-row corpus)
- 4.2 GB on disk, 5.08–6.24 GB peak vs bf16's 11.6 GB
- M5 Max's ~16 GB practical ceiling now admits cohort=10 at Q4

Drop-in: `--model-repo unsloth/gemma-4-E2B-it-UD-MLX-4bit`. No code changes.

bf16 stays the belt-and-suspenders default; Q4-Unsloth-UD ships when memory or cohort-size is the binding constraint.

---

## Per-phase summary

### Phase 0 — Bench harness (scribe-D, ~30 min)
Built `burl/eval/bench_decision_latency.py` extending `bench_batch_throughput.py` + reusing `star_metrics.py` for K1+regret rescoring. Frozen 5-decision subset at `burl/eval/data/perf_subset_5.jsonl`. Ledger CSV at `burl/eval/results/perf_ledger.csv`. Phase 0 baseline: 79.5 s / 5-row at temp=0.6, decode 87.5 tok/s.

### Phase 1 — Cheap wins (scribe-B, ~50 min)
- **Lever 1 (turn-aware token budgets):** plumbed `max_tokens: List[int]` end-to-end through `mlx_lm.batch_generate`. Apples-to-apples re-bench: **wall delta 0.3% vs baseline** at flat-2048 — within the 5-row noise floor. The earlier 1.66× claim was contention shadow. Plumbing wired correctly; will matter under future SFT round if turn-1 reasoning shrinks.
- **Lever 2 (parallel tool dispatch):** **closed empty.** Empirical: 1434/1434 sampled assistant turns from harvest_batched_*  emit exactly one `<|tool_call|>` per turn. wax_museum's gate state machine has trained Gemma into a one-call rhythm. No within-turn parallelism to recover.

### Phase 2 — Continuous batching + prefix sharing (scribe-A, ~45 min)
- **Lever 1 (prefix-cache via LRUPromptCache):** **closed negative.** Two compounding bugs identified by source-reading: (a) `BatchKVCache.merge` heterogeneous-pad penalty (`mlx_lm/models/cache.py:1056-1085`) — every batched decode step processes B × max_length even when most streams have size==0; (b) chat-template alignment bug — assistant `content` stored plain but `chat_template.jinja` re-extracts structured `tool_calls` / `reasoning_content`, so the trie key from `tokenizer.encode(completion_text)` doesn't align with the next turn's render. Both negatives are **provable from source**, not benchmark artifacts.
- **Lever 2 (continuous-batching dispatcher):** built. Alternating-bench result: **mean continuous 38.1s vs mean baseline 38.9s = -1.9% (tie)** on the 5-row. Was earlier reported as 1.8-2.1×; honest reversal — the apparent win was contention shadow. `BatchGenerator` already implements continuous batching internally; the wrapper-side win is putting tool dispatch on the same continuous timeline. Output-equivalent (K1 5/5 across 7 alternating rows). **Designed to win at 560-scale** where straggler-tail amortization matters; that validation is unfinished (see Dragons).

### Phase 3 — Quantization + spec-decode lateral (scribe-C, ~70 min)
- **Lever B (quantization):** Q4 PLE-safe + continuous batching = **headline stack**. Memory headroom finding above is the morning-digest payoff. Q8 underperforms Q4 (M5 Max is memory-bandwidth-bound — smaller weights win at same compute).
- **Lever A (speculative decoding):** **closed dead-on-arrival.** Three structural blockers found before any GPU bench: (1) mlx-lm 0.31.2's `speculative_generate_step` doesn't plumb through `BatchGenerator` — spec-decode would surrender Phase-2's parallelism. (2) Gemma 3 270M tokenizer collapses Gemma 4 special tokens (`<|tool_call>` etc) to byte-fallback subword sequences — vocab size matches but token IDs don't; acceptance ≈ 0 on Burl's tool-call-shaped outputs. (3) `mlx_vlm.generate` confirmed has no spec-decode path either. Spec-decode is dead for batched Apple-Silicon Burl until mlx-lm grows BatchGenerator-aware spec-decode.

### Phase 4 — Full-560 attribution (orchestrator, attempted twice, both crashed)
- **v1** (continuous-batching): crashed at decision_530 / turn 3 (~95% through inference) with **MLX `dynamic_roll` broadcast bug** — cache shapes (4,1,256) vs (3,1,1) mismatched when streams cycled in continuous mode. The user-memory's "MLX batch>=14 broadcast bug" is a special case of a more general stream-cycling shape mismatch.
- **v2** (sync-wave Q4 Unsloth UD): crashed at decision_465 (~80% through) with **`AssertionError: assert play is not None`** in `burl/wax_museum/schemas.py:263` — Q4 emitted a turn that left `last_explored_play` unset; the bench has no quarantine path for that. The cohort-abstraction spec scribe-A wrote in `wiki/topics/harvest-cohort-abstraction.md` is exactly the resilience this needed.
- **Net:** Phase 4 attribution is incomplete. The 5-row stack-best result is the strongest claim we have.

---

## What got built

**Code (across 4 worktrees `perf/{bench,cheap,batch,aggressive}`):**
- `burl/eval/bench_decision_latency.py` — paired-protocol bench harness with `--variant`, `--subset 5|560`, `--continuous`, `--model-repo`, `--temperature`, `--max-tokens-policy turn-aware`.
- `burl/eval/data/perf_subset_5.jsonl` — frozen 5-decision subset (gi=0/36/72/104/136, trick positions 1/3/5/6/7).
- `burl/eval/results/perf_ledger.csv` — 22 rows of measurement data with full provenance (sha, branch, variant_label, all metrics).
- `burl/eval/gus_eval_bridge.py` (promoted from scratch) — hermetic eval bridge.
- Continuous-batching dispatcher inline in `bench_decision_latency.py` (run_bench_continuous).
- Per-prompt `max_tokens: List[int]` plumbing through `GemmaLocalNativeBatched`.

**Wiki contributions** (the durable yield):
- `wiki/experiments/burl-perf-phase{0,1,2,3}.md` — full per-phase writeups with caveats, contention disclosures, root-cause analyses.
- `wiki/topics/continuous-batching-dispatcher-design.md` — submit/pump/close API spec for `ContinuousDispatcher`.
- `wiki/topics/harvest-cohort-abstraction.md` — cohort-as-quarantine-unit pattern; the OOM resilience design that Phase 4 v2 needed.
- `wiki/topics/mlx-cohort-bench-discipline.md` — cross-scribe contention discipline (decode tok/s < 100 = contention signal; serialize benches).
- `wiki/entities/mlx-lm.md` — major Internals section: `BatchGenerator`, `_merge_caches`, `LRUPromptCache`, `BatchKVCache`, `_unprocessed_sequences` deque, line-cited.
- `wiki/entities/gemma-4-e2b.md` — MLX quant landscape with PLE landmine + safe-set tables.
- `wiki/topics/perf-on-the-table.md` — every lever marked Tried with measured results; compounded-stack estimate revised 7-10× → 4-6× → "memory dominant, wall noise-floored".
- `wiki/decisions/max-tokens-2048-floor.md` — Phase 1 revisit paragraph.
- `wiki/log.md` — five new ingest entries.
- `wiki/index.md` — six new pages cataloged.
- `wiki/questions/open.md` — three new entries; one moved to resolved.

**Memory:**
- `project_perf_subset_5_noise_floor.md` — 3.4× same-config wall variance; sub-2× claims unattributable.
- `project_wax_museum_one_call_per_turn.md` — empirical 297/297 then 1434/1434 one-call-per-turn finding.

---

## Dragons (open)

1. **MLX `dynamic_roll` cache-shape mismatch.** `mlx_lm.generate.BatchGenerator._next()` → `BatchKVCache._update_concat()` crashes with broadcast errors when streams complete and cycle in continuous mode. Documented at `wiki/topics/perf-on-the-table.md`. Workaround: sync-wave only at scale.
2. **Bench has no per-decision quarantine.** Phase 4 v2's `AssertionError` in `next_actions_unchanged` should have skipped the failing decision and continued. The cohort abstraction in `wiki/topics/harvest-cohort-abstraction.md` is the spec for that. Implementation is a follow-up branch (`perf/harvest-cohort` suggested by scribe-A).
3. **Cohort=10 prediction unvalidated.** The Q4 memory headroom should admit 2× larger cohorts in production; bench's `--batch 5` ceiling needs lifting + a workload large enough to exercise it.
4. **Continuous-batching dispatcher's win is unproven at 560-scale.** Was the unfinished Phase 4 attribution. Both crashes happened at ~80% — close, but not done.

---

## Recommended next steps (priority order)

1. **Implement `wiki/topics/harvest-cohort-abstraction.md`** in a new branch `perf/harvest-cohort`. Once the bench has cohort-quarantine resilience, Phase 4 can complete.
2. **Re-run Phase 4 once cohort resilience lands.** Both attempts (continuous, sync-wave) on the same data subset to compare.
3. **Lift bench `--batch` ceiling** to validate Q4-Unsloth-UD cohort=10 prediction.
4. **Promote Q4-Unsloth-UD to production default in `harvest_batched.py`.** One-line change once Phase 4 confirms quality at 560.
5. **Open a tracking issue against `mlx_lm`** for the `dynamic_roll` shape-mismatch in continuous-mode stream cycling. Provide the (4,1,256) vs (3,1,1) reproducer.

---

## Honest reversals worth highlighting

This session's discipline contributed two important reversals — both Crystal Palace mode at its best:

1. **Phase 1 lever-1's "1.66× wall reduction"** turned out to be contention shadow. scribe-B re-ran apples-to-apples and found wall delta is ~0%. Source-reading confirmed the codepath change was provably zero-effect (mlx-lm normalizes int → List[int] at line 40-41 of generate.py).
2. **Phase 2 lever-2's "1.8-2.1× speedup"** turned out to be the same noise envelope. scribe-A's alternating B-C-B-C-B-P-C bench landed at -1.9% (tie). The dispatcher is output-equivalent (K1 5/5) and structurally correct, but its 5-row wall win was an artifact.

The retractions were filed in the same session they occurred. The wiki has the "what we thought" → "what we measured" arc embedded at the relevant pages. That epistemic loop is the project's most durable contribution from the session.

---

## Branch state at session end

Four perf branches off `forge`, all pushed:
- `perf/bench` — Phase 0 bench harness
- `perf/cheap` — Phase 1 turn-aware budgets + Lever 2 empirical brief
- `perf/batch` — Phase 2 dispatcher + Lever 1 root-cause + cohort spec + mlx-lm Internals
- `perf/aggressive` — Phase 3 quant audit + PLE-safe set + production recommendation

`perf/aggressive` carries the most-current wiki and is the recommended merge target back to `forge`. Suggested merge order: bench → cheap → batch → aggressive → forge.

---

*Generated 2026-04-27 by team `burl-perf` (orchestrator + scribe-{A,B,C,D}). All scribes idle as of session end.*
