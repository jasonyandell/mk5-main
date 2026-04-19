# iter-2 coverage test — eval writeup

**Author**: corpus-chef (T9)
**Date**: 2026-04-19
**Scope**: train iter-2 on the 118-row blended corpus, eval N=10 held-out @ `max_retries=7`, compare to iter-1@r7 and spike v2.

## TL;DR

**iter-2 regressed vs iter-1@r7 on both primary metrics.** Same recipe, 4× more rows (118 vs 30) — but bot-match dropped from 88.9% → 66.7% (−22 pp) and mean_eq_delta widened from −0.16 → −0.94 (6× worse). Commit discipline held at 9/10; tool idiom stayed structural with a modest uptick in `eq_outcome_distribution` (2 vs iter-1's 1). Net: **more rows did not help; the blend likely hurt by diluting iter-1's higher-quality post-trim rollouts with iter-0's pre-trim content**. Recommend team-lead not merge iter-2 to main and treat the result as one data point arguing *corpus quality > corpus size* at this N.

## Three-way comparison — same N=10 held-out, identical seeds

| metric | spike v2 (base + commit_play) | iter-1 @ r7 (**winner**) | **iter-2 @ r7** |
|---|---|---|---|
| n_attempted | 10 | 10 | 10 |
| n_completed | 10 | 9 | **9** |
| n_retry_exhausted | 0 | 1 | **1** |
| legal_rate | 100% | 100% | **100%** |
| first_legal_rate | — | 90% | **90%** |
| bot_match_rate | **88.9%** | **88.9%** | **66.7%** |
| mean_eq_delta | −1.92 | **−0.16** | **−0.94** |
| p_eq_geq_bot | 88.9% | 88.9% | **66.7%** |
| empty_tool_rollout_rate | — | 10% | **0%** |
| mean_tokens_in | 13,600 | 32,382 | 31,310 |
| mean_tokens_out | 1,500 | 4,134 | **4,766** (+15% vs iter-1) |
| wall_time_seconds | — | 1,245 | 2,087 |
| estimated_usd | — | $0.28 | **$0.46** |

**Key read**: iter-1@r7 remains the winning adapter. iter-2 sits between iter-0 (60% / −3.33) and iter-1 (88.9% / −0.16). Not a regression to iter-0 territory, but not a coverage win either.

## Tool histogram — structural idiom preserved, eq usage slightly up

| tool | spike v2 | iter-1@r7 | **iter-2** |
|---|---|---|---|
| trump_declared | 0 | 9 | **7** |
| is_legal | 15 | 16 | **17** |
| is_trump | 0 | 7 | **9** |
| eq_outcome_distribution | 15 | 1 | **2** |
| unseen | — | 1 | **0** |

iter-2 inherited iter-1's structural-reasoning idiom (`trump_declared` + `is_trump` dominating `eq_outcome_distribution`). The 2× uptick in eq-tool usage (2 calls vs iter-1's 1) is in the right direction but nowhere near spike v2's 15 — the blended corpus is still mostly iter-0/iter-1 content that learned the structural path.

## Training run

- adapter: `jasonyandell/gemma-4-e2b-texas42-burl-iter2`
- corpus: `scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl` (118 rows: 79 long + 38 synthetic short + 1 natural short)
- recipe: 3 epochs, lr 1e-4, rank 16, batch 2 × grad_accum 4, bf16, sdpa (identical to iter-1)
- GPU: B200, 82s wall time, 45 steps, mean_token_accuracy 2.9% → 47.9%
- loss trajectory: **53.87 → 7.70** (end-of-run), train_loss mean 19.77 (dominated by the first 5 warmup steps)
- Modal app: https://modal.com/apps/jasonyandell/main/ap-5LlSsyP36i1vKUuo3aK95K
- estimated cost: ~$0.09 (B200 $4/hr × 82s)

Loss descent looked healthy — this is not a "training didn't converge" failure; the adapter learned a consistent policy that just happens to generalize worse on the held-out N=10.

## Decision-level results (N=10)

| # | bot_play | burl_play | match | tools | burl_eq | bot_eq | Δ | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 21 | 21 | ✅ | 1 | +9.88 | +9.88 | 0 | |
| 2 | 23 | — | ✗ retry-exhausted | 8 | N/A | +21.22 | — | 8 tool calls but never produced a legal `commit_play` |
| 3 | 15 | 15 | ✅ | 4 | −18.22 | −18.22 | 0 | |
| 4 | 22 | 24 | ✗ | 1 | −0.61 | +0.79 | −1.40 | 1-tool commit — went too fast |
| 5 | 4 | 4 | ✅ | 7 | +21.58 | +21.58 | 0 | |
| 6 | 9 | 9 | ✅ | 4 | +2.36 | +2.36 | 0 | |
| 7 | 15 | 15 | ✅ | 4 | −4.41 | −4.41 | 0 | |
| 8 | 25 | 25 | ✅ | 2 | +28.56 | +28.56 | 0 | |
| 9 | 0 | 15 | ✗ | 2 | −21.58 | −18.33 | −3.25 | picked a same-eq-class domino but wrong one |
| 10 | 22 | 25 | ✗ | 2 | −21.67 | −17.86 | −3.81 | near-miss; both bad, but bot less bad |

**Summary**: 6 bot-matches, 1 retry-exhausted, 3 near-misses (all with |Δ| < 4 eq points). The near-misses cluster on decisions with small eq_gaps (D4: 1.40, D9: 3.25, D10: 3.81), which is the regime where the decision is genuinely close and a slightly-different pruning at the tool-call stage swings the choice.

## Interpretation

**Headline**: more rows did not help; the blend likely hurt. Three plausible stories, in order of how much weight I give them:

### 1. (Most likely) Corpus-quality dilution
iter-1's 30 rows came from iter-0-with-trimmed-primer rollouts, already filtered by K1. iter-2's 79 long rows = 50 iter-0 (pre-trim, the 60% adapter) + 29 iter-1 (post-trim). That's majority iter-0 content. **The blend trained ~63% of long rows on a source we already know regressed to 60% bot-match** — the iter-0 rollouts that Phase 4 iter-0 explicitly called out as the regression driver. Adding more of those does not help and can hurt: Gemma sees more examples of "structural reasoning that still lands on the wrong play".

### 2. (Medium likely) Variance at N=10
N=10 held-out @ 88.9% has a ~±15 pp standard error; 66.7% is inside 2σ of 88.9%. The three near-misses are all within eq-gap of 4 points — on a different seed those could flip. Running the same adapter on a second held-out set would tell us; this budget didn't cover that.

### 3. (Least likely but notable) The schema-surprise finding re-prices the blend
The iter-2 chat-template investigation pinned that `strip_thinking()` drops all `<|channel>thought` blocks before tokenization — so the blend's "length reduction" is invisible to the trainer. What iter-2 actually trained on is ~79 unique long rows with ~38 synthetic near-dups (short variants of longs post-strip have ~identical token streams up to whitespace). So iter-2 is effectively 79-unique-rows × 3 epochs = 237 row-passes, vs iter-1's 30-rows × 3 = 90 row-passes. More passes on lower-quality content ≠ win.

**Corroborating data**: `mean_tokens_out` went UP 15% (4,766 vs 4,134) — iter-2 talks *more*, not less, despite the "blend reduces verbosity" framing of T4. That framing was already re-priced in the iter-2 launch doc once the chat-template strip was discovered; this eval confirms the re-priced story (blend = coverage, not verbosity fix) — and shows the coverage lever alone did not pay off.

## What I would NOT conclude

- **"The blender is broken"** — no. It did exactly what T4 designed: 33% short ratio, tool-call envelopes preserved, `commit_play` preserved, synthetic rows marked. T4 tests all pass. The blend just isn't the right intervention for the verbosity problem, and adding lower-quality iter-0 content dilutes rather than augments. That's a data issue, not a code issue.
- **"Training failed"** — no. Loss descent 53.87 → 7.70, token-accuracy 2.9% → 48%, commit discipline held at 9/10 (same as iter-1), no retry-count spikes (mean 0.0). The adapter converged cleanly; it just converged to a weaker policy than iter-1's.
- **"iter-1@r7 is no longer the winner"** — iter-1@r7 remains the winning adapter. iter-2 is explicitly worse on every primary metric.

## Suggested next levers (not in scope for T9)

1. **Re-harvest a fresh iter-2 raw corpus from a cleaner source** (e.g. the spike-v2 prompt shape team-lead has flagged) rather than blending iter-0 + iter-1 outputs. This is the hypothesis most likely to unlock progress: the blender isn't broken, the *input* to the blender was.
2. **Rules-as-tools (T1/T7)** and **EQ-gate (T2/T5)** are orthogonal levers that the blend didn't touch — those were the designs specifically aimed at correctness, not verbosity.
3. **If re-training with a cleaner corpus**, keep the 0.33 blend ratio; the LS-Mixture ablation at 0.0 / 0.25 / 0.33 / 0.50 is probably a second-order concern until the content source is fixed.
4. **Before the next real iter-2 run**, consider bypassing the chat template via an SFTTrainer `formatting_func` so thought blocks actually reach the trainer — the schema surprise is the root cause of the blend being invisible. Out of scope for T6/T9, but a real lever if the team wants training-time verbosity control.

## Cost & budget

| item | estimate | actual |
|---|---|---|
| training (B200, 82s, 45 steps) | $0.10 | $0.09 |
| eval (10 decisions @ r7, Modal L4-equivalent) | $0.25 | **$0.46** |
| **total** | $0.35 (or $0.60 with slack) | **$0.55** |

Eval cost ran ~$0.21 over the $0.25 line-item but stayed **inside the $0.60 total authorization** with ~$0.05 unused. The overrun is explainable: iter-2's mean_tokens_out is 15% higher than iter-1's, and several decisions did multi-tool chains (up to 8 on the retry-exhausted case). No additional authorization needed for this run; if we re-fire with a cleaner corpus, budget $0.50 for eval alone to stay safe.

## Artifacts

- `scratch/burl_p5_iter2_prep/iter2_train.log` — training stdout + RESULT blob
- `scratch/burl_p5_iter2_prep/iter2_eval.log` — eval stdout
- `scratch/burl_p5_iter2_prep/move4_iter2_eval/summary.json` — eval metrics
- `scratch/burl_p5_iter2_prep/move4_iter2_eval/traces.jsonl` — per-decision traces
- `scratch/burl_p5_iter2_prep/move4_iter2_eval/report.md` — auto-generated grading table + sampled traces
- adapter (private HF): `jasonyandell/gemma-4-e2b-texas42-burl-iter2`
- wandb run: `gemma-4-e2b-texas42-burl-iter2`, global_step=45, final loss=7.70
- Modal training app: https://modal.com/apps/jasonyandell/main/ap-5LlSsyP36i1vKUuo3aK95K
