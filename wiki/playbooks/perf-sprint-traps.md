---
title: Perf Sprint — Trap Recipes
kind: playbook
first_seen: aae430b
last_updated: aae430b
status: active
---

Known crashes, footguns, and contention modes. When you see X, do Y. Append to this when a sprint discovers a new failure mode — future scribes shouldn't re-discover from scratch.

## Bench crashes

### `ValueError: [broadcast_shapes] Shapes (4,1,256) and (3,1,1) cannot be broadcast` in `mlx_lm.models.cache.dynamic_roll`

- **What it is.** MLX `dynamic_roll` shape mismatch when batched streams cycle in continuous mode. Cache shape persists from completed-stream count; new input batch has different stream count. The user-memory's "MLX batch>=14 broadcast bug" is a special case of this more general pattern.
- **Workaround.** Pass `prefill_batch_size=2` to `BatchGenerator` (or whatever lower number works for the current batch size). Or revert that specific run to sync-wave (`run_bench` not `run_bench_continuous`).
- **Long-term fix.** mlx-lm version bump may resolve. Watch for cache-related fixes in changelog. Consider filing an upstream issue with the (4,1,256) vs (3,1,1) reproducer.
- **First seen.** Sprint 1 Phase 4 v1, decision_530 / turn 3 (~95% through inference).

### `AssertionError: assert play is not None` in `burl/wax_museum/schemas.py:next_actions_unchanged`

- **What it is.** A stream completed a turn without calling `explore_game(X)`, leaving `last_explored_play` unset. The bench has no per-decision quarantine path, so the whole run dies.
- **Workaround.** Wrap the `_apply_step` call in `try/except AssertionError`; on failure, mark the decision as quarantined and continue to the next.
- **Long-term fix.** Implement [[harvest-cohort-abstraction]] cohort-as-quarantine-unit pattern.
- **First seen.** Sprint 1 Phase 4 v2, decision_465 (~80% through).

## Cross-scribe contention

### Symptom: same-config baseline drifts from 71s → 134s in the same session

- **What it is.** Multiple scribes running mlx-lm inference simultaneously on one Apple-Silicon GPU. Metal + unified-memory contention. `ps aux` won't show the contention; the GPU is one shared resource without OS-level visibility.
- **Detection signal.** `decode_tok_s` < 100 = contended; < 60 = heavily contended. Same-config wall variance > 1.5× across runs in one session.
- **Recipe.**
  1. `ps aux | grep python | grep -v grep` to find competing processes.
  2. Send hard-stand-down DMs to all scribes except one.
  3. Wait 30s for system to settle.
  4. Re-baseline before continuing.
- **Discipline doc.** [[mlx-cohort-bench-discipline]].
- **Prevention.** `bench.lock` file convention. Scribe writes `bench.lock` before run, deletes after. Other scribes refuse to start if it exists. ~5 lines in the bench. Eliminates detective work.

## Bench protocol footguns

### Silent comparison-anchor mismatch

- **What it is.** The bench's `latest_baseline_run()` selector grabs the latest `baseline-*` row in the ledger as the comparison anchor for K1+regret. If the latest baseline was contended, at a different temperature, or from a different code path, the comparison is nonsense.
- **Recipe.** Use the paired protocol — run a fresh baseline IMMEDIATELY before each variant, with same flags. Reference your variant against THAT row, not the ledger's "latest." Better: extend the bench to take an explicit `--baseline-row <timestamp>` flag.
- **Caused.** Two false 1.66× ([[burl-perf-phase1]]) and 2.1× ([[burl-perf-phase2]]) wins in sprint 1. Both retracted in-session.

### Misleading "loading bf16" log line

- **What it is.** Bench logs `[bench] loading Gemma 4 E2B (bf16, adapter=None)` regardless of what `--model-repo` was passed. The "bf16" is hardcoded printf text in `bench_decision_latency.py:962`.
- **Recipe.** Verify model from RSS instead — bf16 ~10–11 GB resident, Q4 PLE-safe ~4–5 GB. Or check the run's JSON output (`model_repo` field is correctly threaded through `args.model_repo` at line 971).
- **Long-term fix.** Patch line 962 to use `args.model_repo` instead of the hardcoded string.

## Wrap-rationalization (the human bug)

### Symptom: orchestrator considers wrapping when goal isn't achieved and levers remain

- **What it is.** Under stress (crashes, ambiguous results, long uncontended idle windows), the orchestrator's judgment converges toward "ship the digest" because that feels like discipline. It is not discipline. It is giving up dressed as discipline.
- **Recipe.** Re-read [[perf-sprint-loop]] wrap conditions. None of "the bench is unreliable," "Phase 4 crashed twice," "the noise floor is high," or "we have a memory win" is a wrap condition. The only wrap conditions are the three predicates listed there.
- **First seen.** Sprint 1, after Phase 4 v2 crashed at decision_465. Orchestrator wrapped instead of patching the assertion + re-running. Real failure mode worth naming.

## Append a trap

When a sprint discovers a new failure mode, add a section with: symptom, what it is, recipe, links to the experiment that hit it.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-history]] [[mlx-cohort-bench-discipline]] [[harvest-cohort-abstraction]]
