---
title: Commit-discipline collapse
kind: topic
first_seen: 622816a
last_updated: pending-this-ingest
status: active
---

## What it is

A failure mode discovered in [[burl-star-run3]] (post-hoc rescore + diagnosis 2026-04-26): when a [[star]] training corpus is filtered to only Burl-already-correct decisions, the resulting LoRA adapter learns the strategic-exploration tool-call sequence verbatim but loses the discrimination signal for *when to stop exploring and commit*. The adapter probes and explores past the turn cap, and the harness force-commits on its behalf.

The full framing matters: **strict-pool training disrupts when-to-commit timing without disrupting what-to-commit quality, because the harness's force-fallback (`forced: highest-E[Q] probed play` / `oracle scan`) substitutes for the missing commit signal and lands near-oracle picks.** The adapter still plays well — it just stops self-determining its commit timing. The cost is decision shape and training-corpus yield, not play quality.

Distinct from [[primer-tradeoff]] (where commit discipline depends on the rules-primer scaffold) and [[commit-discipline]] (the design decision to keep the primer for that reason). Those are *prompt-time* failure modes; this is a *training-time* one.

## Evidence

Same-harness same-corpus same-eval on the held-out 560:

- **Base (no adapter):** 12% FORCED_COMMIT (or ~17% in batched harness, early signal n=18).
- **run-3b** (preserve-thoughts OFF): **32.3% FORCED_COMMIT** at n=130 partial.
- **run-3c** (preserve-thoughts ON): **34.1% FORCED_COMMIT** at n=560.
- **Paired comparison on the same 130 indices: 32.3% vs 32.3% — identical.**

`--preserve-thoughts` is independently confirmed *not* to be the cause; both adapter variants show the same inflation. The driver is the strict-pool corpus filter.

Failure shape: 134 of 158 newly-forced run-3c decisions hit exactly n_turns=8 (the default cap). Zero bailed; zero degenerate completions. The adapter is doing legitimate strategic exploration (`belief_trajectory` 1.86×/dec vs base 1.22×; `probe_worst_case` 1.69× more often) but never narrowing to a commit. See `scratch/belief_trajectory_rollout/star/FORCED_COMMIT_DIAGNOSIS_2026-04-26.md` Q1/Q2/Q3.

Counterintuitive sub-finding: among `n_legal=2` decisions (two trumps to choose between), force rate at `eq_gap >= 5` is **47%** vs `eq_gap < 1` at 36%. Wider eq_gap (one obviously-better play) makes the failure *more* likely — the adapter explores both anyway, even when the answer is clear.

## Why it doesn't cost play quality

The 191 forced run-3c decisions have **mean regret 1.60 vs the 369 not-forced 2.46**. The harness's force-fallback scans the adapter's probed plays and picks the highest-E[Q] one — which is often near-oracle because the adapter probed the right candidates. Even on the 158 newly-forced subset (where naked Burl committed naturally), adapter forced regret (1.81) **beats naked-Burl natural-commit regret on those same indices (2.15)**.

So the harness substitutes for the missing commit signal. Play quality is preserved (slightly improved, even, on the newly-forced subset). What's lost is:

1. **Decision shape / interpretability** — a forced commit is a harness pick, not a model pick. Reasoning chains end mid-thought, the model never gets to "decide." Bad for trace inspection and for any downstream training that wants to learn from clean decision traces.
2. **Training-corpus yield** — FORCED_COMMIT decisions are by-construction filtered *out* of the strict pool (no model-emitted commit_play to train on). At 34% force rate on a 2000-decision harvest, ~680 decisions become unusable for training; usable strict-pool yield drops from ~1700 (cap=12, 15% force) to ~1300 (cap=8, 34% force).

## Implication for STaR keep rule

The [[k1-grading]] keep rule selects on decision *correctness* (Δ ≥ 0). It is silent on commit *timing*. A row where Burl over-explored for 7 turns and then luckily committed to a near-oracle play passes K1, becomes part of the training corpus, and the trained model learns "explore for 7 turns, then commit." A row where Burl committed at turn 3 with the same correctness also passes K1. Both teach exploration, neither teaches commit timing.

Two paths forward:

- **Surface-level fix:** bump rollout turn cap (8 → 12) to push the failure-mode threshold past where most natural commits happen. Doesn't *teach* discipline but recovers training-corpus yield. Cheap; harness-side. Adopted for [[burl-harvest-2]] (see [[burl-star-run3]]'s "What's next" §"FORCED_COMMIT inflation diagnosis (closed)").
- **Training-side fix:** add explicit negative supervision (FORCED_COMMIT rows formatted as "do NOT do this; instead, commit now"). See `scratch/belief_trajectory_rollout/star/CORPUS_ENRICHMENT_PLAN.md`. ~30 min implementation, drop-in compatible with `star_mlx.py` SFT loop. Insurance in case cap-bump alone isn't enough.

Adjacent: [[ls-mixture]] (the arxiv-2505.03469 short/long trace blend) and [[eq-gate-star]] (gating on E[Q] delta, not just Δ ≥ 0) are both staged training-side workstreams that touch this failure mode.

## Pointers

- Diagnosis: `scratch/belief_trajectory_rollout/star/FORCED_COMMIT_DIAGNOSIS_2026-04-26.md`
- Enrichment plan (Format B): `scratch/belief_trajectory_rollout/star/CORPUS_ENRICHMENT_PLAN.md`
- 100-row dry-run: `scratch/belief_trajectory_rollout/star/enriched_FC_100rows_dryrun.jsonl`
- Source experiment: [[burl-star-run3]]
- Companion eval framework: [[regret-eval]] §"Ported to Burl evaluation" (the regret reframe is what surfaced the "no play-quality cost" refinement; without it the headline would still be "FORCED_COMMIT inflation is bad" rather than "FORCED_COMMIT inflation is a yield cost not a quality cost")

## Links

[[burl]] [[star]] [[burl-star-run3]] [[k1-grading]] [[ls-mixture]] [[eq-gate-star]] [[commit-discipline]] [[primer-tradeoff]] [[preserve-thoughts]] [[regret-eval]]
