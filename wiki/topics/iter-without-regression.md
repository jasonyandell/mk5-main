---
title: Iter without regression
kind: topic
first_seen: 2026-04-26
last_updated: 2026-04-26
status: superseded
---

## What it is

Harvest-2 + run-4 (the 2026-04-26 overnight cycle) is the **first Burl iteration that landed without a new bug, pathology, or regression**. Every prior iteration introduced one:

- **Iter-0** ([[burl-iter0-adapter]]): baked in eq-shy from the Layer-1 corpus — 60% bot-match vs Layer-1's 70%.
- **Iter-1** ([[burl-iter1-adapter]]): trimmed primer collapsed commit discipline; 5/10 retry-exhausted on the held-out eval despite zero retry-exhausted during corpus generation.
- **Iter-5** ([[iter5-e1-rank-sweep]]): rank-128 sweep collapsed to a degenerate adapter; rank-32 was the cliff.
- **The 71-row run** (Run-2, postmortem at `scratch/belief_trajectory_rollout/star/POSTMORTEM_002918.md`): loss collapsed to 0.10 and the adapter looped on history strings — too-small corpus crossed with rank-16 + lr-1e-4.
- **Run-3 (pre-run-3b)** ([[burl-star-run3]]): three consecutive launch failures (parent-shell-killed, bare-python ModuleNotFoundError, MLX OOM at iter 487) — three full-cycle losses before the first adapter wrote.

Harvest-2 + run-4 cleared all of the above with **no analogous regression**: the harvest survived a 5h+ wall, the trainer survived a probability-of-resume crash, the eval ran end-to-end, the adapter loaded standalone. The strategic bucket distribution shift from harvest-1 → harvest-2 is +13 strict-pool decisions (+1.2%), basically noise — the recipe stayed put.

## Why this is the foundation, not the ceiling

The "iter without regression" milestone means **the recipe + infra is now stable enough to iterate on**. Future iterations can A/B specific changes without the noise floor of "did the trainer crash" or "did the rollout truncate" obscuring the signal:

- Comparing run-4 to run-3c is the first STaR-shaped iter-N-vs-iter-(N−1) comparison the project has ever had with both runs trained from corpora generated under the same recipe.
- Recipe variations (rank, lr, [[r1-rationalization]] enrichment, [[ls-mixture]]) become cleanly testable without the prior-iteration confounds.
- The [[backwards-curriculum]] ratchet becomes possible because we trust the iter-N base — moving the rollout target from trick 6 to trick 5 is now an A/B against a stable reference, not an A/B against "did the harness work this time."

The 4000 total decisions across the two Burl harvests (1995 in harvest-1 + 2000 in harvest-2) is **tiny** — Zeb's training run plateaued after hundreds of thousands of games. "No regression" doesn't mean "no plateau"; it means "the tools work." The right comparison is `iter ∈ {1, 2}` vs Zeb's `iter ∈ {0, 1, 2, ..., N}` — the project is two ratchets in on a workstream that needs many more before claiming saturation.

## What had to land for this milestone

The infra prerequisites that quietly enabled the regression-free cycle:

- **[[batched-harvest-resilience]]** (per-wave atomic writes + quarantine ledger + OOM fallback) — harvest-1 v2 was the first 2000-decision harvest to land cleanly. Harvest-2 reused the pattern.
- **[[batched-eval-resilience]]** (the same pattern ported to eval) — the n=180 base eval and run-3c eval both survived mid-run interruptions thanks to per-wave incremental writes + `--resume-dir`.
- **[[resumable-checkpointing]]** + crash-snapshot save in `star_mlx.py` — the run-3 OOM that lost a healthy adapter at iter 487 would now leave a recoverable snapshot.
- **[[preserve-thoughts]]** as a confirmed phase change — without it, run-3b emitted thoughts on 0% of decisions, making the iter-1 → iter-2 comparison meaningless. Defaulted ON.
- **[[regret-eval]]** ported to Burl — bot-match alone undersold every prior iteration's signal; the regret reframe revealed run-3c's true −39% regret reduction vs naked-Burl on paired n=180.
- **[[commit-discipline-collapse]]** named and explained — the FORCED_COMMIT inflation discovered in run-3c was originally framed as a regression; the regret-based diagnosis showed it's a decision-shape cost, not a play-quality cost. Cap-bump (8 → 12) recovered training-corpus yield without sacrificing eval quality.

Each of these is a small piece of plumbing. Together they gave the recipe enough stability that the iter-2 cycle could land "boringly" — which is the actual goal at this stage of the project.

## Forward implications

- **R1-rationalization on the 299 base-harvest BBC bucket** ([[r1-rationalization]]) is the next training-experiment candidate. Now testable as an A/B against a stable iter-2 base.
- **Corpus enrichment with FORCED_COMMIT-as-negative** (`scratch/belief_trajectory_rollout/star/CORPUS_ENRICHMENT_PLAN.md`) is staged as insurance if iter-3 reverts to commit-discipline collapse.
- **Backwards-curriculum ratchet to trick 5** becomes meaningful because iter-2 → iter-3 will have an apples-to-apples reference once the rollout target shifts.
- **Capacity scaling experiments** (rank 8 → 16 → 32) can now A/B without the prior-iteration confound of "did the trainer survive."

**None of these three ran.** The project pivoted to a perf-sprint ([[perf-on-the-table]]), then [[burl-chat]]/[[burl-lab]]/[[burl-microscope]], then [[champion]]/[[jud]]. The "iter without regression" milestone held at the infrastructure level named above, but the forward implications it was meant to unlock were never exercised.

## Caveat

This page is celebratory but bounded. Two iterations on a self-improvement loop is not a saturation result; it's not even a trend. The right read is "the tools work, the project can now iterate, and the next 8–10 iterations are where the real signal lives." Don't overclaim ceiling; don't overclaim plateau; don't overclaim breakthrough. The milestone is **stability**, full stop.

## Pointers

- Origin experiments: [[burl-2000-harvest]], [[burl-star-run3]], [[burl-harvest-2]] (covers harvest-2 + run-4)
- Diagnosis that surfaced this framing: `scratch/belief_trajectory_rollout/star/STAR_EVAL_REPORT_2026-04-26.md` + `FORCED_COMMIT_DIAGNOSIS_2026-04-26.md`

## Links

[[burl]] [[star]] [[burl-iter0-adapter]] [[burl-iter1-adapter]] [[iter5-e1-rank-sweep]] [[burl-star-run3]] [[burl-2000-harvest]] [[batched-harvest-resilience]] [[batched-eval-resilience]] [[resumable-checkpointing]] [[preserve-thoughts]] [[regret-eval]] [[commit-discipline-collapse]] [[backwards-curriculum]] [[r1-rationalization]] [[ls-mixture]]
