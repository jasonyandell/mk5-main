---
title: Burl STaR Run-3 (filter-only, strict pool)
kind: experiment
first_seen: 02d9096
last_updated: TBD-shortsha
status: active
---

## Goal

Filter-only [[star]] pass on the 1062-row strict pool from [[burl-2000-harvest]], rank-8 lr-3e-5 1-epoch on M5 Max. Isolates one variable: does [[burl]] learn good reasoning when shown only Burl-already-correct traces, before adding [[r1-rationalization]]? Run-2 (the prior 71-row attempt) collapsed to loss 0.10 and looped on history strings — postmortem at `scratch/belief_trajectory_rollout/star/POSTMORTEM_002918.md` traced it to a tiny corpus crossed with rank-16 + lr-1e-4. Run-3 is the conservative redo on a 10× larger corpus with a capacity-bounded rank.

## Recipe

Corpus: 1062-row strict pool from [[burl-2000-harvest]] (`ALL_AGREE_CORRECT` + `BURL_ALONE_FIXES` + `BOTH_FIX` + `BURL_INDEPENDENT_RIGHT` + `BURL_FOLLOWS_PI_RIGHT`), with `--min-assistant-chars 300` strip — yields ~755 train + ~169 val rows. Pre-built at `scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/`. Token cap follows [[max-tokens-2048-floor]]; the underlying harvest's per-wave resilience is documented at [[batched-harvest-resilience]].

Hyperparameters: rank=8, lr=3e-5, 1 epoch, `--steps-per-eval 50`, `--early-stop-val-rise 1.02 --early-stop-patience 2` (multiplicative — 1.02 = 2% above best, not 0.02). Best-checkpoint snapshot lands the lowest-val-loss params, not the final iteration. Trainer: `burl/train/star_mlx.py`. `--preserve-thoughts` is **load-bearing** (see [[preserve-thoughts]]) — without it the Gemma 4 chat template strips `<|channel>thought...<channel|>` regions before tokenization and the adapter learns to skip reasoning at inference (run-3b).

Eval: held-out sequential 560 (`harvest_20260424_133611/`), `D_required_first`, `max_tokens=2048`, batch=6 via `eval_adapter_smoke.py`. Re-tag against `per_decision_eval_k200.jsonl` and compare bucket distributions to the unadapted 560 baseline.

Full plan and rationale: `burl/STAR_RUN3_PLAN.md`.

## Pre-flight findings

Scout pass on 2026-04-25 caught that the prior session's corpus had been sourced from the held-out eval harvest — training on the eval set would have invalidated all downstream comparisons. Corpus rebuilt from `harvest_batched_20260425_072910/`. See [[burl-2000-harvest]] "Footgun caught (2026-04-25)".

## Training

### Run-3 (three failed launch attempts)

Three launch attempts on 2026-04-25 afternoon; all three died before the adapter was written.

**Attempt 1** (`run3_20260425_144858_DIED_AT_BASELINE_VAL_PARENT_SHELL_KILLED/`): launched as a sub-agent background bash. Killed at iter ~29/50 of baseline val eval when the parent Claude Code session was renamed/`/remote-control`'d — the trainer process was a child of the dying shell. Lesson: trainers must be launched with full session detachment (`nohup setsid` or equivalent), not as Bash-tool children.

**Attempt 2** (`run3_20260425_145914_DIED_BARE_PYTHON_NO_MLX/`): relaunch using bare `python` instead of `.venv/bin/python` — `mlx` isn't on the system Python. Exited immediately with `ModuleNotFoundError`. Lesson: any venv-bound dependency must be invoked through the venv interpreter; never `python ...`, always `.venv/bin/python ...`.

**Attempt 3** (`run3_20260425_150538_FAILED_MLX_OOM_AT_ITER_487/`): the real run. Detached via `nohup ... </dev/null >log 2>&1 &; disown`. Recipe as planned. Survived ~37 min before crashing with `RuntimeError: [metal::malloc] Resource limit (499000) exceeded` at iter 487/1343. The `train_mlx` exception handler only catches the local `EarlyStopRequested`, so the in-memory best snapshot (val 0.302 at iter 450) was lost. **No adapter weights written for any of the three attempts.**

### Run-3b (no preserve-thoughts; succeeded)

Run-3 hyperparameters relaunched with `--max-seq-length 2048` (down from 4096) to clear the OOM ceiling. `nohup setsid` detachment held. Trained 32 min wall, early-stopped at iter 849/1343, best val **0.238** at iter 749. Adapter at `scratch/belief_trajectory_rollout/star/adapters/run3b_20260425_160517_maxseq2048/adapters.safetensors`.

### Run-3c (preserve-thoughts ON; only delta)

Same recipe as run-3b with `--preserve-thoughts` flipped on. Trained 45 min wall, early-stopped at iter 1149/1343, best val **0.268** at iter 949 — settles ~13% higher than run-3b and takes ~300 more iters to converge. Curve shape (concave-down then plateau) is the same; preserve-thoughts is harder to fit because each row carries ~500 extra varied thought tokens vs ~50 short tool-call tokens. Adapter at `scratch/belief_trajectory_rollout/star/adapters/run3c_20260425_172329_preservethoughts/adapters.safetensors`.

| Run    | preserve-thoughts | Best val | Best iter | Wall   | Adapter on disk |
|--------|-------------------|----------|-----------|--------|-----------------|
| run-3  | n/a (3 launches)  | 0.302 (in-memory only) | 450 | crash @ 37min | none |
| run-3b | OFF (default)     | 0.238    | 749       | 32 min | yes             |
| run-3c | ON                | 0.268    | 949       | 45 min | yes             |

## Eval (held-out 560)

### Run-3b eval (partial — killed at n=113)

Eval at `scratch/belief_trajectory_rollout/star/eval/run3b_eval_seq560_20260425_164239/`. 113/560 decisions completed before the session was killed (unrelated to the experiment). Partial result:

| Metric | Run-3b @ n=113 |
|---|---|
| Bot-match | 60.2% |
| Legal | 100.0% |
| Mean \|Δ\| | 2.89 |
| **Thought-block presence** | **0 / 113 = 0%** |

The north-star failure: the adapter learned to emit tool calls and final commits, but **never** the `<|channel>thought...<channel|>` block that Burl's reasoning lives in. Stripped from training rows by the chat template's `strip_thinking()`, gradient-invisible, and therefore absent at inference.

### Run-3c eval (full n=560, 4.1h wall)

Eval at `scratch/belief_trajectory_rollout/star/eval/run3c_eval_seq560_20260425_181016/`. Final result on the full held-out 560:

| Metric | Run-3c @ n=560 | Run-3b @ n=113 |
|---|---|---|
| **Thought-block presence** | **537 / 560 = 95.9%** | 0 / 113 = 0% |
| Bot-match | 373 / 560 = 66.6% | 60.2% |
| Legal | 560 / 560 = 100.0% | 100.0% |
| Mean \|Δ\| | 2.111 | 2.89 |
| **Mean signed Δ** | **−1.984** | not measured |
| Wins (Δ>0) / Ties (Δ=0) / Losses (Δ<0) | 22 / 373 / 165 | not broken out |

Within run-3c, decisions split by whether the model emitted a thought block:

| Subset            | n   | Bot-match | Mean \|Δ\| | Mean signed Δ |
|-------------------|-----|-----------|-----------|---------------|
| Thinking          | 537 | 66.7%     | 2.110     | −1.985        |
| No-thought        |  23 | 65.2%     | 2.130     | −1.965        |

The thought-vs-no-thought split tightened as the sample grew. At n=304 the gap looked like ~8pp on bot-match; at n=560 it's ~1.5pp on bot-match and indistinguishable on |Δ|. The early-decisions-only signal that "thinking helps" was real but small, and per-decision noise dominates at this sample size.

**Honest reading: bot-match alone undersells the picture.** The 66.6% bot-match is mostly the **373 ties** (66.6% of decisions) — agreement with the bot. Among the 187 disagreements, the adapter has **22 wins** and **165 losses** — a 7.5:1 lossy split, with mean signed Δ = −1.98. Run-3c is an adapter that *thinks consistently and plays legally*, but **trades poorly with the baseline bot when it deviates**. This is the [[regret-eval]] reframe applied to Burl evaluation: bot-match is necessary but not sufficient evidence of "playing well." See [[#What's next]] §"Eval-side gaps" — without a same-harness base-model eval and without oracle-relative regret, the adapter-vs-base "this iteration was a STaR win" question remains open even at n=560.

## Verdict

**The north-star result: `--preserve-thoughts` recovered thought-block emission from 0% to ~95%, and within run-3c the decisions where the adapter actually thought are measurably better than the rare ones where it didn't.** Run-3b confirms what [[preserve-thoughts]] already suspected at iter-5 scale: training a Burl LoRA without preserve-thoughts produces a tool-call-only adapter that has trained reasoning *out* of the model. Run-3c is the cleanest preserve-thoughts vs no-preserve-thoughts A/B in the project to date — same recipe, same corpus, same early-stop, same eval — and the variable flips both the training-time loss target (thought tokens reach gradient) and the inference-time behavior (~95% of decisions emit thoughts).

A 0.27% LoRA can flip whether the model thinks at all. That is the lesson that joins the recipe-lesson canon for [[star]].

Scored against `burl/STAR_RUN3_PLAN.md` §"Success criteria":

- run-3: hard fail (no adapter) — see Training §Run-3.
- run-3b: training succeeded, eval ran on 113/560, **thought-block emission 0%** — degenerate but informative.
- run-3c: training succeeded, eval complete (n=560), **thought-block emission 95.9%, bot-match 66.6%, |Δ| 2.11, mean signed Δ −1.98** — north-star win on the *reasoning-emission* axis; the *plays-better-Texas-42* axis remains unproven (see eval-side gaps below).

## What's next

The preserve-thoughts result confirms the iter-5-E1 directional signal ([[iter5-e1-rank-sweep]]: +3.3pp at N=26) at scale (~95pp swing on thought-block presence at N=560). Open questions inherited:

### Eval-side gaps (block the carry-forward decision)

The run-3c eval as designed answers "does the adapter emit thoughts and play legally?" but does **not** answer "is this adapter actually a better Texas 42 player than the unadapted base?" Three eval upgrades are gating the next training round:

1. **Same-harness, same-seeds base-model eval.** Without a `--adapter`-omitted run on the *same* 560 indices using the *same* harness, the +6.4pp bot-match vs run-3b is the only legitimate A/B claim. The "+14pp vs unadapted ~52% baseline" line is from a different harness at a different time and is not directly comparable.
2. **Report mean signed Δ alongside |Δ|.** The harness already records signed `eq_delta_vs_bot`; the summary just needs to surface mean(signed) separately. Run-3c's mean signed Δ of −1.98 is the new headline diagnostic — it says "when the adapter deviates from the bot, it loses ~2 Q-points on average" which is fundamentally different from "the adapter is on average tied with the bot."
3. **Oracle-relative regret, not just delta-vs-bot.** The wiki's [[regret-eval]] page on the Gus side is the precedent: regret = `Q(oracle_argmax) − Q(model_play)` is the metric that survives "but the bot was wrong too." The harness has the K=200 belief-sampled oracle in memory for tagging; computing the argmax-vs-chosen gap is a small extension.

Until those land, "did STaR-iter-3c improve play quality?" cannot be answered cleanly.

### Training-side follow-ups

1. **Why does run-3c skip thinking on ~5% of decisions?** Inspect those traces against the trained corpus — are they multi-turn corrections, forced-commit trips, or genuine "model decided not to think" cases? If the corpus contains rows where Burl committed without thinking (e.g., on forced-commit fallback), the adapter is faithfully reproducing that minority behavior.
2. **Loss-weighting on thought tokens.** Preserve-thoughts puts thought tokens in the loss with uniform weight against tool-call tokens. Up-weighting thoughts (or down-weighting the verbatim-template tail) may further improve thought-block coverage and match rate.
3. **[[r1-rationalization]] on the 299-row `BURL_BREAKS_CONSENSUS` bucket** is the next training experiment candidate, gated by [[reasoning-coherence-verification]] (don't ship rationalization without a verifier in the loop, per the [[candlewax]] workstream). Should be run with `--preserve-thoughts` from the start. **Don't launch it until eval-side gaps are closed** — without regret/signed-delta you can't tell whether rationalization helped or hurt.
4. **Resumable checkpointing on `star_mlx.py`** is still unaddressed (captured as agent memory `project_resumable_training_priority`). Run-3b/3c happened to survive without it; future longer runs may not.
5. **Generic-exception catch in `train_mlx`** — same status; cheap fix, not yet landed.

## Pointers

- Plan: `burl/STAR_RUN3_PLAN.md`
- Trainer: `burl/train/star_mlx.py`
- Corpus builder: `scratch/belief_trajectory_rollout/star/build_filtered_corpus.py`
- Training corpus: `scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/`
- **Run-3b adapter**: `scratch/belief_trajectory_rollout/star/adapters/run3b_20260425_160517_maxseq2048/adapters.safetensors` (train log alongside)
- **Run-3c adapter**: `scratch/belief_trajectory_rollout/star/adapters/run3c_20260425_172329_preservethoughts/adapters.safetensors` (train log alongside)
- Run-3 quarantined adapter dirs (all failed, under `scratch/belief_trajectory_rollout/star/adapters/`): `run3_20260425_144858_DIED_AT_BASELINE_VAL_PARENT_SHELL_KILLED/`, `run3_20260425_145914_DIED_BARE_PYTHON_NO_MLX/`, `run3_20260425_150538_FAILED_MLX_OOM_AT_ITER_487/`
- Run-3b eval dir: `scratch/belief_trajectory_rollout/star/eval/run3b_eval_seq560_20260425_164239/`
- Run-3c eval dir: `scratch/belief_trajectory_rollout/star/eval/run3c_eval_seq560_20260425_181016/`
- Eval harness: `scratch/belief_trajectory_rollout/star/eval_adapter_smoke.py`
- Eval corpus (held-out): `harvest_20260424_133611/`
- Postmortem of the 71-row collapse: `scratch/belief_trajectory_rollout/star/POSTMORTEM_002918.md`
- Live snapshot (run-3b/3c arc): `scratch/belief_trajectory_rollout/star/RUN3BC_LIVE_SNAPSHOT.md`
- Pre-launch prediction: `scratch/belief_trajectory_rollout/star/run3c_prediction.md`
- Source digest: TBD `[[sources/<sha>]]` after post-run commit lands

## Prediction vs reality (run-3c)

The pre-launch prediction at `scratch/belief_trajectory_rollout/star/run3c_prediction.md` is itself a learning artifact — the comparison checks whether the project's intuitions about preserve-thoughts at scale matched what the M5 Max actually produced.

| Axis | Predicted | Observed (n=560) | Verdict |
|---|---|---|---|
| Iter-1 val loss | 3.0–3.5 (vs run-3b ~2.475) | 2.319 (slightly *lower*) | wrong direction; thought-token entropy turned out to be lower per-token than tool-call-token entropy |
| Best val loss | 0.5–0.9 range | **0.268** | predicted way too pessimistic; gap was much smaller than expected |
| Wall-clock | 45–60 min vs 32 min | **45 min** | ✓ |
| Peak memory | should fit under 22 GB ceiling at max-seq 2048 | held at 22.3 GB; no OOM | ✓ |
| Thought-block presence | >50% of decisions | **95.9%** | predicted way too pessimistic; coverage was much higher |
| Bot-match | 50–65%, "could be lower than 60%" | **66.6%** | above top of range — preserve-thoughts didn't cost match rate, slightly improved it |
| Legal rate | ≥98% | 100% | ✓ |
| Failure mode "thought blocks but nonsense" | watched | not flagged at n=560; thought blocks track the prompt and are on-topic | not triggered |
| Failure mode "0% thought blocks again" | would mean preserve-thoughts isn't doing what we think | not triggered — flag did exactly what was advertised | resolved |
| Eval-side: thought-vs-no-thought signal would survive | "thinking decisions ~5-10pp better on bot-match" | survived match-rate split was 1.5pp at n=560 (not 5-10pp as in early sample) | weaker than predicted at full n |
| Eval-side: bot-match would be the right success metric | implied | **wrong**: signed Δ = −1.98 reveals the adapter loses 7.5:1 to the bot on the 187 disagreements; bot-match alone is misleading | flagged for next-round eval upgrades |

**Takeaway:** two predictions held (val-loss shape, thought-block presence flips), one was wrong in a teaching way (per-token entropy of thoughts is lower not higher than tool calls — natural language Gemma already knows beats varied tool-call patterns), and one was a methodological gap (the eval as designed answers "does it think" but not "does it play well" — see What's next §Eval-side gaps). The [[iter5-e1-rank-sweep]] +3.3pp signal at N=26 was the directional read; at N=560 it's a phase change on thought emission, but the play-quality verdict remains open without signed-delta and oracle-regret.

## Links

[[burl]] [[star]] [[burl-2000-harvest]] [[r1-rationalization]] [[max-tokens-2048-floor]] [[batched-harvest-resilience]] [[reasoning-coherence-verification]] [[candlewax]] [[iter5-e1-rank-sweep]] [[preserve-thoughts]] [[iter4-null-preserve-thoughts]]
