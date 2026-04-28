---
title: Perf Sprint — Trap Recipes
kind: playbook
first_seen: fbe798f
last_updated: 8deaba4 (iter 15-redo: gus belief adapter not found in worktree)
status: active
---

Known crashes, footguns, and contention modes. When the loop hits one of these, here's the smallest fix — apply it and re-run rather than re-deriving from scratch. Append when a sprint discovers a new failure mode.

Trap recipes age. mlx-lm and Gemma 4 are moving weekly — before spending an iteration on a workaround, web-search the upstream changelog or GitHub issue tracker; the bug may already be fixed.

## Bench crashes

### `ValueError: [broadcast_shapes] Shapes (4,1,256) and (3,1,1) cannot be broadcast` in `mlx_lm.models.cache.dynamic_roll`

- **What it is.** MLX `dynamic_roll` shape mismatch when batched streams cycle in continuous mode. Cache shape persists from completed-stream count; new input batch has different stream count.
- **Workaround.** Pass `prefill_batch_size=2` to `BatchGenerator` (or whatever lower number works for the current batch size). Or revert that specific run to sync-wave (`run_bench` not `run_bench_continuous`).
- **Long-term fix.** mlx-lm version bump may resolve. Watch for cache-related fixes in changelog. Consider filing an upstream issue with the `(4,1,256)` vs `(3,1,1)` reproducer.

### `ValueError: Received 60 parameters not in model: language_model.model.layers.{15..34}.self_attn.{k,k_norm,v}_proj.weight` at bf16 Gemma 4 E2B load time on mlx-lm 0.31.3+

- **What it is.** mlx-lm 0.31.3 ships [PR #1158](https://github.com/ml-explore/mlx-lm/pull/1158) which removes the `k_proj`/`v_proj`/`k_norm`/`v_norm` modules from Gemma 4 attention layers that are KV-shared (`num_kv_shared_layers=20` for E2B → layers 15-34). The model class no longer has weight slots for those tensors. The official `google/gemma-4-E2B-it` (and `mlx-community/gemma-4-E2B-it-bf16`) safetensors STILL ship them. `mlx_lm.utils.load()` calls `load_weights(strict=True)` and the load aborts with the 60-parameter rejection. Confirmed firing in sprint 2 iter 7 (2026-04-28). bf16 model cannot be loaded at all on 0.31.3 — this is a hard wall, not a degradation.
- **Recipe (vendored loader, ~14 LoC).** Sprint 2 iter 8 (2026-04-28, commit `9be36dc`, reset) verified the vendored-loader path loads bf16 Gemma 4 E2B successfully on 0.31.3. The patch (in `burl/modal/gemma_local_batched.py`):
  ```python
  from mlx_lm import batch_generate  # drop `load` from this import
  from mlx_lm.utils import (
      _download as _mlx_download,
      load_adapters as _mlx_load_adapters,
      load_model as _mlx_load_model,
      load_tokenizer as _mlx_load_tokenizer,
  )

  def _load_strict_false(path_or_hf_repo, adapter_path=None):
      model_path = _mlx_download(path_or_hf_repo)
      model, config = _mlx_load_model(model_path, lazy=False, strict=False)
      if adapter_path is not None:
          model = _mlx_load_adapters(model, adapter_path)
          model.eval()
      tokenizer = _mlx_load_tokenizer(
          model_path, eos_token_ids=config.get("eos_token_id", None),
      )
      return model, tokenizer
  ```
  Then in `__init__`: `self.model, self.tokenizer = _load_strict_false(model_repo, adapter_path=adapter_path)`. Mirrors the four-line body of `mlx_lm.utils.load()` exactly, with the single `strict=False` change. Loader is sound and reusable in any future iter targeting 0.31.3+.
- **Caveat — vendored loader DOES NOT make 0.31.3 usable for the perf gate.** Iter 8 also discovered an mlx-lm 0.31.3 bf16 regression (next trap below) that breaks the sanity gate even with the loader fix. Until that regression clears in 0.31.4+ or upstream, pin `mlx-lm==0.31.2` regardless.
- **Long-term fix.** File an upstream issue requesting `mlx_lm.utils.load()` accept `strict=False` (the underlying `load_model` already does). Or PR upstream a `sanitize` hook for Gemma 4 that drops the unused KV-shared weight names before `load_weights` sees them.

### bf16 Gemma 4 E2B perf + determinism regression on mlx-lm 0.31.3

- **What it is.** Sprint 2 iter 8 (2026-04-28) ran three back-to-back paired bf16 batch=5 temp=0 sanity baselines on mlx-lm 0.31.3 (with the vendored strict=False loader unblocking model load). All three returned per-decision grades that match prior bf16 grades on the deterministic decisions, but two regressions surfaced. **Wall regression**: A=99.3s, B=72.4s, C=73.6s — steady-state ~75s vs the 0.31.2 floor of 38–48s, so ~70% slower. Decode dropped from ~142 tok/s to ~75 tok/s, ps clean, no contention. **Determinism regression**: at the same config (bf16 batch=5 temp=0), runs A and C produced gi=72 play=13 K1=True regret=0; run B produced gi=72 play=1 K1=False regret=1.7338. Iter 5 explicitly verified at mlx-lm 0.31.2 that bf16 at temp=0 was byte-identical across batch widths and runs. **0.31.3 broke deterministic equivalence at temp=0 for bf16 Gemma 4 E2B on this subset.** Either PR #1158's KV-shared cleanup or PR #1141's BatchKVCache rework introduced numerical drift; root cause not bisected.
- **Recipe.** Pin `mlx-lm==0.31.2` until 0.31.4+ ships and the regression is verified clear. Do NOT trust 0.31.3 wall numbers for paired-baseline comparisons even after the strict=False loader fix unblocks model load — both arms of the pair will land at the new structurally-worse floor, so the comparison metric is moot. Re-run the iter 8 sanity protocol (3× bf16 batch=5 temp=0 paired runs, manual cross-comparison of K1+regret) on each future bump to detect when the regression clears.
- **Long-term fix.** File upstream issue with the iter 8 reproducer (3 paired bf16 batch=5 temp=0 runs producing differing gi=72 plays). On a known-good future release, retire this trap and re-open lever #3.

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

## Profiling / instrumentation footguns

### Apple Metal capture (`mx.metal.start_capture`) blows up to multi-GB on a real bench run

- **What it is.** mlx exposes `mx.metal.start_capture(path)` / `stop_capture()` with `MTL_CAPTURE_ENABLED=1` to record a `.gputrace` file for Xcode. Apple Metal capture serializes every command buffer dispatch and writes the binary state into the trace bundle (the `.gputrace` is a directory with thousands of `MTLBuffer-*` files, often hardlinked to share state). Sprint 2 iter 13 (2026-04-28, commit `e9f1e6c`) wrapped a 5-decision continuous bench run in `start_capture` and the trace ballooned to **5.3 GB at 6 minutes elapsed** with no signs of stopping (iter 11's clean baseline of the same workload is 41s). Killed before it filled the disk; the .gputrace was unrecoverable to Xcode anyway since the host runs macOS 26.4.1 (issue [#2846](https://github.com/ml-explore/mlx/issues/2846) — "metal_capture does not capture working GPU trace in MacOS 26.1+", same on .4.x).
- **Recipe.** Don't wrap a multi-second whole-bench run in `mx.metal.start_capture`. If you need kernel-level profiling, two options: (a) capture a SINGLE forward pass (~10-50ms) — small enough that the trace fits in low MB and replay won't take forever; (b) skip the .gputrace entirely and use a `KernelCounter` analytical breakdown (call-counts × per-call bandwidth-bound cost) — reusable from iter 13's `bench_decision_latency.py:--kernel-audit` flag. Option (b) is what produced the iter 13 5-row breakdown without ever opening Xcode.
- **Long-term fix.** None at the playbook level. Apple's Metal capture is structurally not designed for whole-application traces of long-running workloads; it's a single-frame-snapshot tool.
- **Caveat — instrumentation overhead.** Even WITHOUT `MTL_CAPTURE_ENABLED=1`, the iter 13 kernel-counter wrappers (per-call Python wrapping of `nn.Linear.__call__` at the class level) added ~38% wall overhead on the bf16 continuous bench (56.5s instrumented vs 41s clean). Use the `--kernel-audit` flag only when needed, never for paired wall comparisons against an un-instrumented baseline.

### `nn.Module.__call__` cannot be overridden at the instance level

- **What it is.** Setting `instance.__call__ = hooked_fn` on an `nn.Linear` (or any `nn.Module`) silently does nothing — Python's special-method lookup for `__call__` skips the instance dict and goes straight to the class via type(...). Sprint 2 iter 13 hit this when adding `KernelCounter` hooks; the original "set `layer.__call__ = wrapped`" approach left every Linear unwrapped (zero counter increments, but no visible error). Wasted ~10 minutes of debug time on an audit run.
- **Recipe.** To intercept `nn.Linear.__call__` per-instance, replace the CLASS-level `__call__` with a wrapper that reads a per-instance tag (`getattr(self, "_audit_kernel_name", None)`) and falls through to the original for untagged instances. Tag the instances you care about during install. Reverse on teardown by restoring the original class-level `__call__`. See `bench_decision_latency.py:install_kernel_hooks` for the working recipe.
- **Long-term fix.** None — this is a Python-language constraint, not an MLX issue.

## Stuck worker

### Symptom: worker silent for 2+ `/loop` fires

- **What it is.** The worker is wedged — could be an mlx-lm internal hang, a stuck subprocess, an infinite loop in the bench, or the worker reasoning itself into a corner. The orchestrator's `/loop` fires don't get a `SendMessage` reply.
- **Recipe.**
  1. `SendMessage` the worker once more with a tight question (e.g. "respond with one word: alive?").
  2. If no response: `TaskStop` the worker.
  3. Respawn a fresh worker with the same variant via the spawn template in [[perf-sprint]]. The new context may avoid whatever wedge the previous one hit.
  4. If the same variant wedges twice in a row: log a `crash` row to `results.tsv` with description "wedge — variant skipped," and pick a different variant.
- **Long-term fix.** None at the playbook level. If a specific variant or technique consistently wedges, document it as a closed lever in [[perf-sprint-levers]] with the wedge as the closure reason.

### `FileNotFoundError` on `gus/adapters/v3_consistency_10000g.pt` mid-decision when running the bench from a worktree

- **What it is.** Two adapter-resolution paths in the bench disagree when the bench runs from a `git worktree`. The bench top-level (`bench_decision_latency.py:1248-1258`) explicitly falls back to the absolute main-checkout path and pre-loads gus successfully, populating `belief_trajectory._load_gus_cached`'s lru_cache keyed on the main-checkout string. But `belief_trajectory.DEFAULT_ADAPTER` is computed from `Path(__file__).resolve().parents[2] / "gus/adapters/v3_consistency_10000g.pt"` — i.e. **worktree-relative**. The first time a stream calls `belief_trajectory()` as a tool with `adapter_path=None`, `load_gus()` falls through to that worktree path, misses the lru_cache (different string key than the pre-load used), and `torch.load` aborts with `[Errno 2] No such file or directory`. Bench setup looks healthy (corpus loaded, oracle loaded, gus pre-loaded, model ready) and decision dirs are created up-front by continuous mode — the bench dies on `decision_0`'s first turn. Confirmed firing in sprint 2 iter 15 (2026-04-28) when `gus/adapters/` did not exist in the perf/aggressive worktree.
- **Recipe.** From the worktree, symlink the adapter into the worktree-relative path so both resolution paths converge:
  ```bash
  mkdir -p .claude/worktrees/<name>/gus/adapters
  ln -s /Users/jason/code/mk5-main/gus/adapters/v3_consistency_10000g.pt \
        .claude/worktrees/<name>/gus/adapters/v3_consistency_10000g.pt
  ```
  Pre-flight check before any iter that runs from a worktree: `ls <worktree>/gus/adapters/v3_consistency_10000g.pt`. The .gitignore on `gus/adapters/` means a fresh `git worktree add` will never carry the directory across.
- **Long-term fix.** Make `belief_trajectory.DEFAULT_ADAPTER` use the same fallback list as the bench top-level (try worktree-relative, then `/Users/jason/code/mk5-main/gus/adapters/...` absolute). One ~6-line edit in `burl/tools/belief_trajectory.py`. Or thread the resolved path through the tool registration so callers don't fall back to the module default. Either fix removes the trap permanently.

## Append a trap

When a sprint discovers a new failure mode, add a section with: symptom, what it is, recipe.

## Links

[[perf-sprint]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-history]]
