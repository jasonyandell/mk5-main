---
title: Perf Sprint — Trap Recipes
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Known crashes, footguns, and contention modes. When the loop hits one of these, here's the smallest fix — apply it and re-run rather than re-deriving from scratch. Append when a sprint discovers a new failure mode.

Trap recipes age. mlx-lm and Gemma 4 are moving weekly — before spending an iteration on a workaround, web-search the upstream changelog or GitHub issue tracker; the bug may already be fixed.

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

- **What it is.** The bench's `latest_baseline_run()` selector grabs the latest `baseline-bf16` row in the ledger as the comparison anchor for K1+regret. If the latest baseline was contended, at a different temperature, or from a different code path, the comparison is nonsense. **Confirmed firing in sprint 2 iter 2** (2026-04-28): both the paired baseline (variant_label `baseline-iter2-pre`) and the variant compared against a stale `baseline-bf16` from 20260427_013851 — even though I ran `baseline-iter2-pre` immediately before the variant. Cause: the selector matches on `baseline-bf16` exactly (or some prefix), not "the most recent baseline-* row of any name." A baseline labeled `baseline-iter2-pre` is invisible to it.
- **Recipe.** Don't trust `comparison_vs_baseline` in the JSON output. Read `per_decision_grades` from BOTH the paired baseline JSON and the variant JSON, compute K1 (count of `k1_pass=true`) and regret (`sum(regret)`) yourself, then `regret_delta_pct = (variant_total_regret - baseline_total_regret) / baseline_mean_regret_or_eps * 100`. Or: run your paired baseline with `--variant baseline-bf16` exactly so the selector picks it up. Or: extend the bench to take an explicit `--baseline-row <timestamp>` flag.

### Single-shot equivalence gate is broken at temp=0.6 on gi=0-style multimodal decisions

- **What it is.** The gate (`K1_match >= 60% AND regret_delta within ±10%`) compares one variant run against one paired baseline. If any decision has a *multimodal sampling distribution* at temp=0.6 — multiple final_plays with materially different regret — the baseline lands on one mode and the variant lands on another, then the regret_delta blows past ±10% even when both are valid samples of the same distribution. **Confirmed firing in sprint 2 iter 3** (2026-04-28): two paired bf16 batch=5 temp=0.6 runs back-to-back produced bf16↔bf16 k1_pass agreement=100% and regret_delta=0% across the full subset, but `gi=0` history across 11 prior bf16 temp=0.6 runs shows final_play ∈ {25, 25, 2, 25, 25, 25, 25, 25, 2, 6, 19} with regrets {12.5, 0, 7.6, 9.8} — a true multimodal sampler. At temp=0, bf16 always picks play=2 (regret 0). Iter 2's Q4 verdict (gi=0 picked play=6, regret 7.63) was almost certainly not Q4 damage but Q4 sampling a mode bf16 also samples (the `q8-bf16-cont` run on 2026-04-27 hit play=6 with the same 7.632 regret).
- **Recipe.** Either (a) **pin `--temperature 0` for the gate** so K1+regret comparisons are deterministic — wall_s loses some realism, but the equivalence question becomes well-posed; or (b) **average K1+regret over N≥3 reseeded paired runs** at temp=0.6 and gate against the averaged distribution. Option (a) is the smaller code change and the right default for the perf sprint's gate.
- **Long-term fix.** Make the gate a two-track measurement: deterministic equivalence at temp=0 (must pass) AND temp=0.6 wall_s_per_decision (the perf metric). Don't gate on temp=0.6 K1/regret at single-shot.

### Misleading "loading bf16" log line

- **What it is.** Bench logs `[bench] loading Gemma 4 E2B (bf16, adapter=None)` regardless of what `--model-repo` was passed. The "bf16" is hardcoded printf text in `bench_decision_latency.py:962`.
- **Recipe.** Verify model from RSS instead — bf16 ~10–11 GB resident, Q4 PLE-safe ~4–5 GB. Or check the run's JSON output (`model_repo` field is correctly threaded through `args.model_repo` at line 971).
- **Long-term fix.** Patch line 962 to use `args.model_repo` instead of the hardcoded string. **Status:** patched and reverted in sprint 2 iter 4 (whole iter 4 commit was reset on gate failure). Re-apply on the next keep iteration.

### Q4 UD-MLX-4bit byte-equivalence claim doesn't survive temp=0

- **What it is.** Sprint 1 declared `unsloth/gemma-4-E2B-it-UD-MLX-4bit` byte-equivalent to bf16 on the 5-row corpus. Sprint 2 iter 4 (commit `6353da6`, reset) ran Q4 vs bf16 paired at temp=0 and found gi=0 deterministically flipped (bf16 play=2 regret 0 → Q4 play=6 regret 7.632). Same temp, same subset, deterministic both sides — the equivalence claim is wrong on this subset. The 44% wall win + ~50% peak-mem reduction is real, but the gate fails. **Sprint 2 iter 5 (commit `dac9d28`, reset) closed the bisect**: re-ran Q4 at batch=5 temp=0 (matching bf16's batch width); gi=0 still flips to play=6 with byte-identical regret 7.632 and identical 60% K1_match. The flip is **pure quant damage**, not batch-prefill numerics. UD-MLX-4bit's logits on gi=0 deterministically prefer play=6 to play=2; batch width is not load-bearing.
- **Recipe.** Don't trust prior byte-equivalence claims when bumping defaults to a quant model — re-verify at temp=0 against bf16 on the actual subset before declaring a new floor. UD-MLX-4bit is dead for this subset's gate; for the next quant attempt, try Q8 (`FakeRockert543/gemma-4-e2b-it-MLX-8bit`) — Q8 quant noise is typically ~16× smaller than Q4 and is the next floor candidate.
- **Long-term fix.** When promoting any quant set to "safe," include the temp=0 deterministic comparison artifact in the lever ladder note, not just a prose claim.

### gi=0 is a quant-fragile logit-cliff decision (the "~16× smaller noise" prior failed)

- **What it is.** Sprint 2 iter 6 (commit `1e82482`, reset) ran Q8 (`FakeRockert543/gemma-4-e2b-it-MLX-8bit`) at temp=0 batch=5 expecting Q8's ~16× smaller quant noise to byte-match bf16 on gi=0. **Q8 deterministically flipped gi=0 to play=6 with byte-identical regret 7.632 to Q4 UD-MLX-4bit.** Two different quant recipes (UD-MLX 4-bit, FakeRockert 8-bit) at two very different bit-widths produced the same wrong answer with the same regret. Q8 also added NEW damage at gi=104 (bf16 play=6 K1=True regret=0 → Q8 play=19 K1=False regret=0.025) — wider damage footprint than Q4. The "Q8 ≈ bf16" rule of thumb is unreliable on individual logit-cliff decisions; bit-width reduction does not commute with argmax across all decisions.
- **Recipe.** When a single decision flips deterministically across two unrelated quant recipes with byte-identical regret, treat the *decision* as quant-fragile, not the *quant set* as broken. Either (a) re-freeze the perf subset to exclude that decision before chasing quant wall wins, or (b) widen the equivalence gate to tolerate one K1-flip per subset (e.g. `K1_match >= 60% AND |regret_delta| <= max(10%, 1 fragile-decision worth of regret)`), or (c) pivot to non-quant levers (mlx-lm version bump, `mx.compile`, spec-decode) that don't perturb logits at all.
- **Long-term fix.** Subset-freeze protocol should include a "quant-fragility audit": for each frozen decision, verify its bf16 argmax survives a Q4 + Q8 perturbation. Decisions that flip across quant should either be excluded from the perf subset or marked as known-fragile so the gate weights them differently.

## Stuck worker

### Symptom: worker silent for 2+ `/loop` fires

- **What it is.** The worker is wedged — could be an mlx-lm internal hang, a stuck subprocess, an infinite loop in the bench, or the worker reasoning itself into a corner. The orchestrator's `/loop` fires don't get a `SendMessage` reply.
- **Recipe.**
  1. `SendMessage` the worker once more with a tight question (e.g. "respond with one word: alive?").
  2. If no response: `TaskStop` the worker.
  3. Respawn a fresh worker with the same variant via the spawn template in [[perf-sprint]]. The new context may avoid whatever wedge the previous one hit.
  4. If the same variant wedges twice in a row: log a `crash` row to `results.tsv` with description "wedge — variant skipped," and pick a different variant.
- **Long-term fix.** None at the playbook level. If a specific variant or technique consistently wedges, document it as a closed lever in [[perf-sprint-levers]] with the wedge as the closure reason.

## Append a trap

When a sprint discovers a new failure mode, add a section with: symptom, what it is, recipe.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-history]]
