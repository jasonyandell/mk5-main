---
title: Burl STaR Run-3 (filter-only, strict pool)
kind: experiment
first_seen: 02d9096
last_updated: 06bf5bf
status: active
---

## Goal

Filter-only [[star]] pass on the 1062-row strict pool from [[burl-2000-harvest]], rank-8 lr-3e-5 1-epoch on M5 Max. Isolates one variable: does [[burl]] learn good reasoning when shown only Burl-already-correct traces, before adding [[r1-rationalization]]? Run-2 (the prior 71-row attempt) collapsed to loss 0.10 and looped on history strings — postmortem at `scratch/belief_trajectory_rollout/star/POSTMORTEM_002918.md` traced it to a tiny corpus crossed with rank-16 + lr-1e-4. Run-3 is the conservative redo on a 10× larger corpus with a capacity-bounded rank.

## Recipe

Corpus: 1062-row strict pool from [[burl-2000-harvest]] (`ALL_AGREE_CORRECT` + `BURL_ALONE_FIXES` + `BOTH_FIX` + `BURL_INDEPENDENT_RIGHT` + `BURL_FOLLOWS_PI_RIGHT`), with `--min-assistant-chars 300` strip — yields ~755 train + ~169 val rows. Pre-built at `scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/`. Token cap follows [[max-tokens-2048-floor]]; the underlying harvest's per-wave resilience is documented at [[batched-harvest-resilience]].

Hyperparameters: rank=8, lr=3e-5, 1 epoch, `--steps-per-eval 50`, `--early-stop-val-rise 1.02 --early-stop-patience 2` (multiplicative — 1.02 = 2% above best, not 0.02). Best-checkpoint snapshot lands the lowest-val-loss params, not the final iteration. Trainer: `burl/train/star_mlx.py`.

Eval: held-out sequential 560 (`harvest_20260424_133611/`), `D_required_first`, `max_tokens=2048`, batch=6 via `eval_adapter_smoke.py`. Re-tag against `per_decision_eval_k200.jsonl` and compare bucket distributions to the unadapted 560 baseline.

Full plan and rationale: `burl/STAR_RUN3_PLAN.md`.

## Pre-flight findings

Scout pass on 2026-04-25 caught that the prior session's corpus had been sourced from the held-out eval harvest — training on the eval set would have invalidated all downstream comparisons. Corpus rebuilt from `harvest_batched_20260425_072910/`. See [[burl-2000-harvest]] "Footgun caught (2026-04-25)" once landed.

## Training

Three launch attempts on 2026-04-25; all three died before the adapter was written.

**Attempt 1** (`run3_20260425_144858_DIED_AT_BASELINE_VAL_PARENT_SHELL_KILLED/`): launched as a sub-agent background bash. Killed at iter ~29/50 of baseline val eval when the parent Claude Code session was renamed/`/remote-control`'d — the trainer process was a child of the dying shell. Lesson: trainers must be launched with full session detachment (`nohup setsid` or equivalent), not as Bash-tool children.

**Attempt 2** (`run3_20260425_145914_DIED_BARE_PYTHON_NO_MLX/`): relaunch using bare `python` instead of `.venv/bin/python` — `mlx` isn't on the system Python. Exited immediately with `ModuleNotFoundError`. Lesson: any venv-bound dependency must be invoked through the venv interpreter; never `python ...`, always `.venv/bin/python ...`.

**Attempt 3** (`run3_20260425_150538_FAILED_MLX_OOM_AT_ITER_487/`): the real run. Detached via `nohup ... </dev/null >log 2>&1 &; disown`. Recipe as planned. Survived ~37 min before crashing.

Val-loss curve before the crash (val cap = 50 batches, eval every 50 micro-steps):

| iter | val loss | wall (val) |
|------|----------|------------|
| 1    | 2.354    | 79s        |
| 50   | 1.920    | 86s        |
| 100  | 1.369    | 85s        |
| 150  | 0.999    | 85s        |
| 200  | 0.560    | 78s        |
| 250  | 0.410    | 31s (cache warm) |
| 300  | 0.358    | 29s        |
| 350  | 0.333    | 30s        |
| 400  | 0.317    | 28s        |
| 450  | 0.302    | 28s        |

Crash at iter 487 / 1343 (36% of one epoch, ~62 optimizer steps in). Train loss at crash: 0.319 (well above the 0.15 collapse floor). Curve was concave-down, no early-stop trip, no divergence.

```
RuntimeError: [metal::malloc] Resource limit (499000) exceeded.
  at mlx/optimizers/schedulers.py line 85
  in cosine schedule's `mx.cos(...)` call inside `optimizer.update`
```

The `train_mlx` exception handler only catches the local `EarlyStopRequested` exception, so a generic MLX `RuntimeError` propagates uncaught and the in-memory best snapshot (which would have been at iter 450, val 0.302) is lost. **No adapter weights were written to disk for any of the three attempts.**

## Eval (held-out 560)

Not run. Adapter never materialized.

## Verdict

**Hard fail with strong partial signal.** The recipe was working — val loss dropped 7× (2.354 → 0.302) over 9 evals with no collapse, divergence, or early-stop trigger — but environmental + ergonomic failures prevented capture. Per the [[burl-2000-harvest]] "trust the manifest" rule, three pre-launch bugs were caught (corpus source, three flag names, early-stop semantics); the post-launch failure modes (parent-shell death, bare-python, MLX OOM) join the recipe-lesson canon for [[star]].

Scored against `burl/STAR_RUN3_PLAN.md` §"Success criteria":

- ✅ Training survived without collapse, val curve healthy, best-checkpoint snapshot would have fired (in-memory only, lost on uncaught crash).
- ❌ Strict-pool sanity eval not run (no adapter).
- ❌ Held-out 560 not run (no adapter).
- ✅ No illegal commits or forced-commit issues — these are runtime concerns and there was no runtime.

## What's next

Three orthogonal blockers to resolve before run-3b:

1. **Resumable checkpointing on `star_mlx.py`** (priority fix for any future Burl/STaR/forge run, captured as the `project_resumable_training_priority` agent memory). Periodic on-disk adapter writes every N optimizer steps + a `--resume-from <adapter-dir>` flag that re-loads optimizer state and step counter. Without this, any environmental crash wastes the entire wall-clock investment.
2. **Catch generic exceptions in `train_mlx`** so that on any crash (not just `EarlyStopRequested`), the in-memory best snapshot still gets serialized. Cheap fix; can land in the same commit as resumability.
3. **Reduce peak memory headroom for the 4096-token-cap regime.** OOM hit at peak 36.5 GB at iter 487. Options: drop `--max-seq-length 4096 → 2048` (likely sufficient — the corpus's p99 turn was ~600 tok per [[max-tokens-2048-floor]]), or drop `--batch 2 → 1` (doubles wall-clock), or check whether mlx-lm has accumulated a known cosine-scheduler memory leak at this version (0.31.2). Evidence-based call: try `--max-seq-length 2048` first.

Once those land, **run-3b** is the same recipe relaunched. If the val curve continues the trajectory we saw (0.302 at iter 450 → likely sub-0.20 at convergence), the held-out 560 eval is the next gate. If filter-only is clean: [[r1-rationalization]] on the 299-row `BURL_BREAKS_CONSENSUS` bucket, gated by [[reasoning-coherence-verification]] (don't ship rationalization without a verifier in the loop, per the [[candlewax]] workstream). If filter-only is null or negative: corpus-filter or hyperparameter iteration before assuming the recipe is wrong.

## Pointers

- Plan: `burl/STAR_RUN3_PLAN.md`
- Trainer: `burl/train/star_mlx.py`
- Corpus builder: `scratch/belief_trajectory_rollout/star/build_filtered_corpus.py`
- Training corpus: `scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/`
- Adapters (all failed, quarantined under `scratch/belief_trajectory_rollout/star/adapters/`): `run3_20260425_144858_DIED_AT_BASELINE_VAL_PARENT_SHELL_KILLED/`, `run3_20260425_145914_DIED_BARE_PYTHON_NO_MLX/`, `run3_20260425_150538_FAILED_MLX_OOM_AT_ITER_487/` — last one's `train.log` has the full val-loss curve.
- Eval harness: `scratch/belief_trajectory_rollout/star/eval_adapter_smoke.py`
- Eval corpus (held-out): `harvest_20260424_133611/`
- Postmortem of the 71-row collapse: `scratch/belief_trajectory_rollout/star/POSTMORTEM_002918.md`
- Source digest: TBD `[[sources/<sha>]]` after post-run commit lands

## Links

[[burl]] [[star]] [[burl-2000-harvest]] [[r1-rationalization]] [[max-tokens-2048-floor]] [[batched-harvest-resilience]] [[reasoning-coherence-verification]] [[candlewax]] [[iter5-e1-rank-sweep]]
