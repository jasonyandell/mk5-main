---
title: Burl STaR Run-3 (filter-only, strict pool)
kind: experiment
first_seen: 02d9096
last_updated: 02d9096
status: active
---

## Goal

Filter-only [[star]] pass on the 1062-row strict pool from [[burl-2000-harvest]], rank-8 lr-3e-5 1-epoch on M5 Max. Isolates one variable: does [[burl]] learn good reasoning when shown only Burl-already-correct traces, before adding [[r1-rationalization]]? Run-2 (the prior 71-row attempt) collapsed to loss 0.10 and looped on history strings — postmortem at `scratch/belief_trajectory_rollout/star/POSTMORTEM_002918.md` traced it to a tiny corpus crossed with rank-16 + lr-1e-4. Run-3 is the conservative redo on a 10× larger corpus with a capacity-bounded rank.

## Recipe

Corpus: 1062-row strict pool from [[burl-2000-harvest]] (`ALL_AGREE_CORRECT` + `BURL_ALONE_FIXES` + `BOTH_FIX` + `BURL_INDEPENDENT_RIGHT` + `BURL_FOLLOWS_PI_RIGHT`), with `--min-assistant-chars 300` strip — yields ~755 train + ~169 val rows. Pre-built at `scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/`. Token cap follows [[max-tokens-2048-floor]]; the underlying harvest's per-wave resilience is documented at [[batched-harvest-resilience]].

Hyperparameters: rank=8, lr=3e-5, 1 epoch, `--steps-per-eval 50`, `--early-stop-val-rise 0.02 --early-stop-patience 2`. Best-checkpoint snapshot lands the lowest-val-loss params, not the final iteration. Trainer: `burl/train/star_mlx.py`.

Eval: held-out sequential 560 (`harvest_20260424_133611/`), `D_required_first`, `max_tokens=2048`, batch=6 via `eval_adapter_smoke.py`. Re-tag against `per_decision_eval_k200.jsonl` and compare bucket distributions to the unadapted 560 baseline.

Full plan and rationale: `burl/STAR_RUN3_PLAN.md`.

## Pre-flight findings

Scout pass on 2026-04-25 caught that the prior session's corpus had been sourced from the held-out eval harvest — training on the eval set would have invalidated all downstream comparisons. Corpus rebuilt from `harvest_batched_20260425_072910/`. See [[burl-2000-harvest]] "Footgun caught (2026-04-25)" once landed.

## Training

TBD — wall-clock, val-loss curve summary (best step, plateau shape), best-checkpoint step, and whether early-stop tripped (and at which eval).

## Eval (held-out 560)

TBD — mean Burl regret vs the 0.517 deployable Q-mean baseline, `matches_bot` delta vs unadapted ~52%, `BURL_BREAKS_CONSENSUS` bucket-count delta, illegal-commit count, forced-commit rate.

## Verdict

TBD — clean win / partial win / negative, scored against the success criteria in `burl/STAR_RUN3_PLAN.md` §"Success criteria":

- training survives without collapse and best-checkpoint fires;
- strict-pool sanity eval picks gold-bucket play >90%;
- held-out regret drops below 0.517 with `matches_bot` up and `BURL_BREAKS_CONSENSUS` down;
- no new failure modes (illegal commits stay 0, forced-commit rate stays ≤12%).

## What's next

TBD. If filter-only is clean: [[r1-rationalization]] on the 299-row `BURL_BREAKS_CONSENSUS` bucket, gated by [[reasoning-coherence-verification]] (don't ship rationalization without a verifier in the loop, per the [[candlewax]] workstream). If filter-only is null or negative: corpus-filter or hyperparameter iteration before assuming the recipe is wrong.

## Pointers

- Plan: `burl/STAR_RUN3_PLAN.md`
- Trainer: `burl/train/star_mlx.py`
- Corpus builder: `scratch/belief_trajectory_rollout/star/build_filtered_corpus.py`
- Training corpus: `scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/`
- Adapter: `scratch/belief_trajectory_rollout/star/adapters/run3_<timestamp>/` (TBD)
- Eval harness: `scratch/belief_trajectory_rollout/star/eval_adapter_smoke.py`
- Eval corpus (held-out): `harvest_20260424_133611/`
- Postmortem of the 71-row collapse: `scratch/belief_trajectory_rollout/star/POSTMORTEM_002918.md`
- Source digest: TBD `[[sources/<sha>]]` after post-run commit lands

## Links

[[burl]] [[star]] [[burl-2000-harvest]] [[r1-rationalization]] [[max-tokens-2048-floor]] [[batched-harvest-resilience]] [[reasoning-coherence-verification]] [[candlewax]] [[iter5-e1-rank-sweep]]
