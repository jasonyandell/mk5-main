---
title: Perf Sprint — Trap Recipes
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Known crashes, footguns, and contention modes. When the loop hits one of these, here's the smallest fix — apply it and re-run rather than re-deriving from scratch. Append when a sprint discovers a new failure mode.

## Bench crashes

### `ValueError: [broadcast_shapes] Shapes (4,1,256) and (3,1,1) cannot be broadcast` in `mlx_lm.models.cache.dynamic_roll`

- **What it is.** MLX `dynamic_roll` shape mismatch when batched streams cycle in continuous mode. Cache shape persists from completed-stream count; new input batch has different stream count.
- **Workaround.** Pass `prefill_batch_size=2` to `BatchGenerator` (or whatever lower number works for the current batch size). Or revert that specific run to sync-wave (`run_bench` not `run_bench_continuous`).
- **Long-term fix.** mlx-lm version bump may resolve. Watch for cache-related fixes in changelog. Consider filing an upstream issue with the `(4,1,256)` vs `(3,1,1)` reproducer.

### `AssertionError: assert play is not None` in `burl/wax_museum/schemas.py:next_actions_unchanged`

- **What it is.** A stream completed a turn without calling `explore_game(X)`, leaving `last_explored_play` unset. The bench has no per-decision quarantine path, so the whole run dies.
- **Workaround.** Wrap the `_apply_step` call in `try/except AssertionError`; on failure, mark the decision as quarantined and continue to the next.
- **Long-term fix.** Implement the cohort-as-quarantine-unit pattern (lever #7).

## Cross-scribe contention

### Symptom: same-config baseline drifts wildly across runs in one session

- **What it is.** Multiple processes running mlx-lm inference simultaneously on one Apple-Silicon GPU. Metal + unified-memory contention. `ps aux` won't show the contention — the GPU is one shared resource without OS-level visibility.
- **Detection signal.** `decode_tok_s` < 100 = contended; < 60 = heavily contended. On clean GPU at batch=5 bf16 Gemma 4 E2B temp=0, the floor is ~36–40s wall, ~170 decode tok/s, ±4% run-to-run.
- **Recipe.**
  1. `ps aux | grep python | grep -v grep` to find competing processes.
  2. Pause every other inference job until the bench finishes.
  3. Wait 30s for the system to settle.
  4. Re-baseline before continuing.
- **Prevention.** `bench.lock` file convention. Write `bench.lock` before run, delete after; refuse to start if it exists. ~5 lines in the bench. Eliminates detective work.

## Bench protocol footguns

### Silent comparison-anchor mismatch

- **What it is.** The bench's `latest_baseline_run()` selector grabs the latest `baseline-*` row in the ledger as the comparison anchor for K1+regret. If the latest baseline was contended, at a different temperature, or from a different code path, the comparison is nonsense.
- **Recipe.** Run a fresh baseline immediately before each variant, with same flags. Reference your variant against THAT row, not the ledger's "latest." Better: extend the bench to take an explicit `--baseline-row <timestamp>` flag.

### Misleading "loading bf16" log line

- **What it is.** Bench logs `[bench] loading Gemma 4 E2B (bf16, adapter=None)` regardless of what `--model-repo` was passed. The "bf16" is hardcoded printf text in `bench_decision_latency.py:962`.
- **Recipe.** Verify model from RSS instead — bf16 ~10–11 GB resident, Q4 PLE-safe ~4–5 GB. Or check the run's JSON output (`model_repo` field is correctly threaded through `args.model_repo` at line 971).
- **Long-term fix.** Patch line 962 to use `args.model_repo` instead of the hardcoded string.

## Append a trap

When a sprint discovers a new failure mode, add a section with: symptom, what it is, recipe.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-history]]
