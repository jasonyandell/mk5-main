---
title: Perf Sprint — Session History
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Append-only log of perf sprints. Each entry: target, outcome, what worked, what didn't, what changed in the playbook. Future sprints read this so the wisdom of past sprints is already encoded.

The per-sprint `results.tsv` is the truth — every iteration's wall_s, gate values, and keep/discard decision. The prose entries below are texture: why a sprint went the way it did, what the rows don't say on their own.

## 2026-04-27 — Sprint 1 — Burl per-decision latency

**Target:** 26s/decision → 2s/decision on M5 Max (13×).

**Outcome:** Wall improvement on full-N never attributed (bench crashed at ~80% twice). 5-row directional evidence positive. -56% peak memory confirmed (Q4 PLE-safe Unsloth UD vs bf16). Sprint wrapped before the goal was hit and with levers still on the ladder.

**What worked.** Four parallel work-streams (measurement, cheap wins, big lever, aggressive) made the sprint productive in wall-clock terms. Research-mode work in parallel with someone else holding the GPU turned idle into structural-finding time. Honest in-session retractions caught two false wins (1.66× → 0.3% and 2.1× → tie) that source-reading exposed. Wiki-first culture — findings written as they were discovered — meaningfully enriched the wiki.

**What didn't.**
- **Cross-process GPU contention** corrupted hours of data before being recognized. Apple Silicon's unified-memory GPU is one resource and `ps aux` doesn't expose Metal contention. Codified in [[perf-sprint-traps]] (`decode_tok_s` < 100 detection signal, `bench.lock` convention).
- **The "3.4× noise floor" claim was a misdiagnosis.** Same-config wall variance of 3.4× across runs was attributed to small-subset noise; it was actually the contention above. Clean-GPU floor is ±4%. The misdiagnosis matters because "the bench is unreliable" became a felt-true reason to wrap, when it was really a fixable systems problem.
- **Bench crashed twice at ~80% of full-N runs.** Each crash had a small fix (try/except quarantine; `prefill_batch_size=2`). Both are now lever #1 and the first two trap recipes.
- **The sprint wrapped instead of fixing forward.** Two crashes plus the false noise floor produced a felt-true "ship the digest" instinct that read as discipline but was rationalization. Lever #1 was about an hour of work away from giving full-N attribution. This is the failure mode the new playbook's contract is shaped around: **don't give up — crashes are work, not a stop sign.**

**What changed in the playbook.** This session is the reason the playbook exists. The shape was over-specified at first (mood framing, wrap predicates, mandatory rules, scribe team shape) — clarified on 2026-04-27 to: clear goal, equivalence bar, don't-give-up clause, lever ladder, trap recipes, history. Trust the model to do the rest.

**Open levers when sprint ended:** see [[perf-sprint-levers]] (8 entries; #1 — bench resilience — would have unblocked full-N attribution).

**Worktrees + branches at session end:**
- `perf/bench` — Phase 0 bench harness
- `perf/cheap` — Phase 1 turn-aware budgets + Lever 2 empirical brief
- `perf/batch` — Phase 2 dispatcher + Lever 1 root-cause + cohort spec + mlx-lm internals
- `perf/aggressive` — Phase 3 quant audit + PLE-safe set + production recommendation

Suggested merge order back to `forge`: bench → cheap → batch → aggressive → forge. `perf/aggressive` carries the most-current wiki.

## 2026-04-28 — Sprint 2 — Burl per-decision latency, redux

**Target:** 26s/decision → 2s/decision on M5 Max (13×). Same target as Sprint 1.

**Outcome.** Two material wall_s/decision wins land cleanly; M5 Max structural envelope mapped. 18 iterations, ~12 hours of wall time, three orchestrator-driven inflection points.
- **subset_5 win — iter 11**: bf16 batch=5 continuous batching at temp=0 = 41.0s wall (39.4% reduction vs sync-wave 67.7s same-session baseline). Per-decision wall vector tightens from sync-wave [67.7, 67.7, 67.7, 60.6, 67.7] (every decision pays slowest's cost) to continuous [14.6, 24.4, 23.0, 37.7, 41.0] (each finishes when its own turns finish). Textbook straggler-collapse elimination — iter 10's wall-time profiling (decode 89.34% of wall, per-step decode_tps {133-150 on full waves, 24.5/33.6 on 4-idle waves}) was the diagnostic that justified the lever.
- **subset_560[:50] win — iter 14**: bf16 batch=8 continuous at temp=0 = 13.28s/decision vs batch=5 cont 17.67s/dec same-subset baseline = 24.8% per-decision reduction. Gate passes K1=94%, regret_delta=-8.2% favorable. 5 final_play flips (4 favorable). Real decode 135.8 → 183.5 tok/s (+35% throughput). Peak only +3.3% over baseline (BatchKVCache amortization dominates over extra-stream-slot overhead at N>>batch).

**What worked.**
- **Wall-time profiling before pulling another lever** (iter 10). Replaced 9 iters of hypothesis-driven lever pulls with a measured fingerprint. Decode 89% of wall + visible straggler-collapse pointed straight at continuous batching; the lever landed in iter 11.
- **Subset pivot when subset_5 was structurally exhausted.** Iter 12 falsified pool-size scaling at N=5 (3 permanently-empty slots paying overhead with zero amortization). Iter 14's pivot to subset_560[:50] gave continuous batching the structural surface it needed (N >> batch_width); the win replicated cleanly.
- **Iteration agent contract held throughout.** Every iter shipped one coherent variant with a clean keep/discard call. Workers committed/reset on perf/aggressive in their own context. The orchestrator never read bench output, source dumps, or tracebacks — every iter cost ~2k tokens to the orchestrator's context. ~18 iters of work fit comfortably without context degradation.
- **Wiki updates compounded across iters.** Worker-3 documented the gate-broken-at-temp=0.6 trap (caught iter 2's false discard); worker-7 documented the mlx-lm 0.31.3 60-param load regression; worker-13 mapped the kernel-class fingerprint; worker-18 documented the silent-jetsam recipe. Each iter inherited prior wisdom without orchestrator relay.
- **Honest in-session retractions.** Iter 3 invalidated iter 2's "Q4 quant damage" verdict (gi=0 was multimodal at temp=0.6, Q4 picked an in-distribution mode). Iter 4 then DID prove Q4 quant damage at temp=0. Iter 11's gate breach on regret_delta=-500% was called keep on "spirit-of-gate" because the breach was favorable on a documented bimodal decision — borderline call but defensible.
- **Worker shutdown discipline.** After the user flagged session leakage, every returned worker got a `shutdown_request`. Most acknowledged cleanly via `shutdown_response`; the silent ones leaked sessions but cleaned up at process end. Playbook updated to make this explicit.

**What didn't.**
- **PLE-safe Q4/Q8 quants flip gi=0 deterministically on subset_5.** Iters 2/4/5 (Q4 UD-MLX-4bit) and iter 6 (Q8 FakeRockert543/MLX-8bit) all pick play=6 (regret 7.632) where bf16 picks play=2 (regret 0). Iter 3's "temp=0.6 multimodal sampling" hypothesis turned out to be incomplete — at temp=0 Q4/Q8 still flip. Pure quant damage on this checkpoint's gi=0 logit landscape, not Q4-specific. The 42-49% wall improvement is real but unreachable through the gate. Levers #1 and #2 closed pending a different quant set, a re-frozen subset, or a widened gate.
- **mlx-lm 0.31.3 introduces a hard bf16 regression.** Iter 7 found PR #1158 removes unused KV-shared k_proj/v_proj weights from Gemma 4 E2B layers 15-34, but the official safetensors still ship those weights → `load_weights(strict=True)` rejects them. Iter 8 vendored a `strict=False` loader; bf16 then loads, but 0.31.3 itself runs ~70% slower on bf16 (75s vs 44s baseline) AND breaks determinism at temp=0 (gi=72 produced different plays across runs). Lever #3 closed pending 0.31.4+.
- **mx.compile lever was already closed** by upstream. Iter 9 audited every shape-stable site; mlx-lm has compiled what's compileable (sampler chain, Gemma 4 fast paths, RMSNorm/SDPA on fused Metal kernels). Remaining un-compiled sites (Attention.__call__, BatchKVCache.update_and_fetch, BatchGenerator._step) have variable shapes + Python control flow and aren't viable.
- **Metal kernel audit found no sub-fused-kernel ROI.** Iter 13 identified that ~85% of decode wall is bandwidth-bound bf16 gemv on Apple's MPS-fused matmul path (lm_head + 3 MLP matmuls per layer). Lever #7 closed.
- **Pool-size scaling at N=50 is a sharp threshold.** Iters 15/16 (batch=12, batch=10 cont at N=50) both fail the gate with regret_delta=+27.7%/+28.8% (unfavorable). Iter 14's batch=8 is the regret-stable ceiling; the favorable-flip mode does not generalize smoothly with batch width. Sub-lever 5d closed.
- **N>50 is infeasible on M5 Max with this bench architecture.** Iter 17 bisected N=100/80/64/56/52; all silent-died via macOS jetsam (peak KV+heap exceeds Apple's 40.2GB recommended Metal working-set). Workaround paths: stay at N≤50, wrap `mx.set_memory_limit(40GB)` for visible RuntimeError, implement lever #6 cohort abstraction, or hardware bump.

**What changed in the playbook.**
- Added explicit worker-cleanup discipline (TaskStop returned workers; TeamDelete at sprint end). Backgrounded workers don't auto-release on return.
- New trap recipes: silent-anchor-mismatch (confirmed firing); single-shot equivalence gate broken at temp=0.6 on multimodal decisions; bf16 60-parameter rejection on mlx-lm 0.31.3; mlx-lm 0.31.3 bf16 wall+determinism regression; vendored-loader pattern for upstream/checkpoint mismatches; run_bench_continuous queue-fill at scale; quant-fragility on logit-cliff decisions (Q4 and Q8 byte-identical at gi=0); silent SIGKILL via OS jetsam at N≥52 with bf16 cont.
- Lever ladder pruned. #1, #2 closed (quant-fragility); #3 closed pending 0.31.4+; #4 closed (mx.compile already applied); #5 sub-levers a-f mostly closed (5c stays as confirmed-active at N≤50); #7 closed (kernel landscape fully fused). #6 cohort and `mx.set_memory_limit` guard are the only remaining levers.

**Open levers when sprint ended:**
- **#6 cohort abstraction** — multi-iter structural project; the only path to subset_560 full-N on M5 Max.
- **bench `mx.set_memory_limit(40GB)` guard** — small hygiene patch so future scale attempts surface RuntimeError instead of silent jetsam. Saves future workers ~2hr of debugging.
- **mlx-lm 0.31.4+ retest** — when upstream ships, re-run iter 8's vendored-loader + continuous-batching protocol. PRs #1141 (dynamic_roll fix), #1090 (thread-local streams), #1170/#1171 (parallel tool calls) all unblock if the bf16 wall+determinism regression clears.

**Worktrees + branches at session end:**
- `perf/aggressive` — iter 17 final state at 4263721 (queue-cap experiment reverted). Carries iter 1 bench resilience, iter 10 phase-time instrumentation, iter 13 kernel-audit instrumentation, iter 14 --subset-limit flag, iter 16 adapter symlink fix. The instrumented bench is the durable artifact.
- `forge` — wiki updates throughout; latest at de67c4d.

Suggested merge order back to `forge`: perf/aggressive's bench improvements stack cleanly (resilience → instrumentation → subset-limit). The diagnostic prints landed and reverted, so the working tree is clean.

## Append a sprint reflection

When a sprint ends, add a section here. Durable institutional memory across garage weekends.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-traps]]
