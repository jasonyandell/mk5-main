---
title: Regret Eval (E[Q] lost vs oracle's best legal)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-07-13
status: complete
---

## Overview

Regret eval is the primary quality metric for [[gus]]. For each held-out decision: `regret = oracle_best_eq - student_chosen_eq`. Averaged over all held-out decisions, it measures how many E[Q] points the student loses relative to the oracle's best legal play (2a09050).

## Why better than bot-match

Texas 42 is full of near-tie positions where multiple actions have E[Q] within 0.1 Q-points of each other — the plays are strategically equivalent. Bot-match counts "picked the wrong one of a tied pair" as a miss; regret counts it as zero loss. Regret captures actual strategic cost, not nominal agreement with one particular tie-breaking choice (2a09050).

**Companion metric: near-ties rate** — fraction of bot-mismatches where the student's pick is within 0.5 Q-points of the oracle's best. In practice, 70–75% of mismatches are near-ties (2a09050):

| Adapter | Bot-match | Mean regret | Near-ties rate |
|---|---|---|---|
| v1_full_1000g | 65.4% | 2.16 Q-pt | 73.4% |
| v2_voids_1000g | 63.6% | 2.22 Q-pt | 72.0% |

2.16 Q-points of mean regret on a ±42 Q-point scale ≈ 2.5% of range lost per decision. The 35% "bot-mismatches" are overwhelmingly near-tie alternative choices, not strategic blunders (2a09050).

## Decision-hardness correlation

High-regret decisions correlate with high E[Q] spread (oracle max − oracle min across legal actions). The student's highest-regret decisions (0, 4, 8, 11, 12, 16) are the same decisions with highest strategic spread (a50c9ef):

| Decision | E[Q] spread | Student regret |
|---|---|---|
| 0 (first play) | 13.2 | 4.0 |
| 4 | high | 8.64 |
| 8 | high | 4.12 |

Uniform-random picking at decision 0 would give ~6.6 regret; the student achieves 4.0, doing real inference with zero observable information (a50c9ef).

End-game decisions (24-27): 0 regret, 100% bot-match. Perfect — forced play or near-forced (2a09050).

## Bimodal distribution of regret

Regret is NOT normally distributed (f0139a3, PRACTICALITIES receipt #5):

- **73% of decisions have exactly 0 regret** — the student matches the oracle's argmax.
- **6% blunder tail (regret > 5 Q-pts)** drives essentially all of the 1.39 mean regret.
- The remaining ~21% is spread across small non-zero regret (near-ties and modest misses).

Implication: mean regret is misleading as a scalar. The action is in the blunder tail — reducing the 6% blunder rate is worth more than shaving the middle distribution. The [[v-pi-decoupling]] finding explains where those blunders come from; the [[consistency-regularizer]] is the targeted fix (f0139a3, b007cf3).

Near-tie rate (70–75%) confirms: most "mismatches" are genuinely equivalent plays. The real ceiling is on the 6% blunder tail (f0139a3).

## Comparison to K1 grading

[[k1-grading]] in LEM/Burl asks "did the model beat the bot?" (binary). Regret asks "by how much did the model miss the oracle's best?" (continuous). Regret is more informative for a supervised-distillation student where every legal play can be scored by the oracle (2a09050).

## Ported to Burl evaluation (2026-04-26)

The same regret framework now scores Burl adapters as well as gus students. The post-hoc rescorer at `scratch/belief_trajectory_rollout/star/star_eval_report.py` joins existing Burl eval rows to the per-decision K=200 oracle data (`diagnostic/per_decision_eval_k200.jsonl`) and computes:

- `oracle_regret = oracle_best_eq − adapter_chosen_eq` (the gus-side metric, applied per-decision)
- `signed_delta = adapter_eq − bot_eq` (the rollout-bot reference, signed not absolute — exposes the wins-vs-losses split that bot-match alone hides)
- `k1_pass = signed_delta ≥ 0` (the [[k1-grading]] keep rule, made post-hoc-able)
- `near_tie_rate = fraction with regret ≤ 0.5` (the gus-side near-ties metric)

### Why this was needed

The Burl side started with bot-match as the primary success metric. [[burl-star-run3]] caught the trap at scale: run-3c bot-match 66.6% on the held-out 560 looked like a win, but signed-delta = −1.98 revealed that among the 187 disagreements with the bot, the adapter splits 22 wins / 165 losses (7.5:1 lossy). Bot-match was confounding "agreed with the bot" (which can be wrong) with "played well." The gus precedent — Texas 42 has many near-tie positions, so any binary match metric is noisy — applied directly to Burl evaluation.

### Burl-side reading

On the held-out 560 (cross-harness comparison via `corpus_index_k200.jsonl`):

| Player                  | n   | Mean oracle regret |
|-------------------------|----:|-------------------:|
| π (gus policy head)     | 560 | 0.551              |
| Q-mean                  | 560 | 0.518              |
| Naked Burl (no adapter) | 560 | 2.295              |
| Run-3c (preserve-thoughts adapter) | 560 | **2.165** |

**FINAL paired n=180 (in-distribution batched harness, same indices both runs, landed 2026-04-26 ~07:00):**

| Player           | n   | Mean oracle regret |
|------------------|----:|-------------------:|
| Naked Burl       | 180 | 3.132              |
| Run-3c           | 180 | **1.915** (−39%)   |

**Run-3c is meaningfully better at Texas 42 than naked-Burl, not just better at emitting thought blocks.** A 39% relative reduction in oracle regret on the same 180 decisions, much larger than the cross-harness 5.7%. Per-bucket on the in-distribution paired sample: BIW −6.21, BOTH_FIX −5.73, FORCED_COMMIT −3.15, AAW −1.67, BBC −0.48, **AAC −0.36** (the cross-harness "AAC tax" was a harness artifact, not the adapter — the adapter is *slightly better* than base on the easy cases too). The in-distribution paired test is the cleanest possible adapter-vs-base comparison.

### Adjacent Burl-side findings the regret reframe surfaced

- **`commit-discipline-collapse`** ([[commit-discipline-collapse]]) — refined to "decision-shape cost not play-quality cost" once the regret metric showed FORCED_COMMIT decisions (1.60 mean regret) outperform not-forced (2.46) — the harness's force-fallback picks near-oracle plays from the adapter's probes, so forcing isn't a strategic loss, it's a self-determination loss.
- **K=1 keep rule** is uniformly more permissive than zero-regret (run-3c k1_pass 70.5% vs match_oracle 65.4%) — the keep rule passes near-ties as wins, matching the gus near-ties philosophy.

### Implementation pointers

- Tracked rescore + metric contract: `burl/eval/star_metrics.py`
- Tracked CLI wrapper: `burl/eval/star_eval_report.py`
- Original run-3 report builder: `scratch/belief_trajectory_rollout/star/star_eval_report.py`
- Rescore output: per-eval `star_rescore.json` + `star_rescore_rows.jsonl` alongside each eval dir
- Comparison report: `scratch/belief_trajectory_rollout/star/STAR_EVAL_REPORT_2026-04-26.md`
- Triangle script (used to fold the n=180 base eval into the report, reported above): `scratch/belief_trajectory_rollout/star/fold_base_into_report.py`
- Origin experiment: [[burl-star-run3]]
- Diagnosis spinoff: `scratch/belief_trajectory_rollout/star/FORCED_COMMIT_DIAGNOSIS_2026-04-26.md`

Promotion note (2026-04-26): the tracked metric layer deliberately reads
actual `thinking` events from per-decision `events.jsonl`; `belief_trajectory`
usage is not a proxy for thought-block emission.

## Links

[[gus]] [[burl]] [[burl-star-run3]] [[expected-q-value]] [[k1-grading]] [[pimc]] [[v-pi-decoupling]] [[consistency-regularizer]] [[commit-discipline-collapse]] [[preserve-thoughts]]
