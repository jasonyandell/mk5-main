# iter-5 E1 — LoRA capacity sweep under untruncated thoughts

**Author**: analyst (T9)
**Date**: 2026-04-19
**Scope**: Four-way A/B eval of base Gemma 4 E2B and three `preserve_thoughts=True` LoRA adapters (rank 16 / 64 / 128) trained on the iter-3-rules 26-row corpus with `max_seq_length=4096` (no truncation). N=10 held-out decisions, `max_retries=7`, rules-as-tools preamble.

## TL;DR

**The scientific question changed mid-flight.** The iter-4-thoughts byte-identical null result was rediagnosed as a training-data artifact: `max_seq_length=2048` had been truncating exactly the thought-bearing rows in the corpus, so the "preserve thoughts" recipe never saw the thoughts it was supposed to learn from. E1 is the first preserve-thoughts adapter family trained on untruncated thought content. The ITER4_PLAN §1 hypothesis (a) capacity-bound vs (b) base-reflex dominance is answered differently than originally framed: **divergence appears at rank 16**, so we are no longer asking "does any thought-gradient move the output?" but "how does rank allocate that gradient as it scales?"

The answer is unflattering for the scale-it-up reflex: **the dose-response curve is monotone bad above rank 16**. Base 66.7% → rank-16 70.0% → rank-64 55.6% → rank-128 0.0%. rank-16 is the only adapter that improves on base; rank-64 regresses below base; rank-128 collapses catastrophically (10/10 retry-exhausted, zero tool calls, 16k rambling chars of output per decision). On a 26-row corpus with MLX-LM's ungraduated LR×rank interaction and no gradient clipping, more capacity does not help — it destroys the policy.

**Recommendation**: rank-16 preserve-thoughts on untruncated data is the new baseline for the preserve-thoughts lineage. The next lever is **corpus size**, not rank. E3 (iter-5-rules-scaled, rank 16-32 × N=100 rollouts) is the right follow-up; E2 Candlewax remains independent and unaffected.

## Setup

Four configurations, same N=10 held-out decisions (`burl/eval/data/move4_decisions_n50.jsonl`), same rules-as-tools preamble, `max_retries=7`, temperature 0.6:

| condition | adapter | rank | notes |
|---|---|---|---|
| base | none | — | Gemma 4 E2B with rules-as-tools preamble, no SFT |
| rank16 | `e1/rank16` | 16 | MLX-LM, 26 rows, `preserve_thoughts=True`, `max_seq_length=4096` |
| rank64 | `e1/rank64` | 64 | MLX-LM, same corpus & recipe except rank |
| rank128 | `e1/rank128` | 128 | MLX-LM, same corpus & recipe except rank |

All three adapters trained on the iter-3-rules corpus of 26 rows (23 rollout_win + 3 EQ-gate self-corrects, after the `preserve_thoughts` path was fixed to not strip thinking blocks at SFT). The old iter-4-thoughts run used Modal/UnSloth with a 2048-token ceiling that silently truncated exactly the rows bearing thought content — this E1 sweep is the first time the "preserve thoughts" recipe has actually exposed untruncated thought tokens to the loss.

## Headline metrics

| metric | base | rank16 | rank64 | rank128 |
|:---|---:|---:|---:|---:|
| n_attempted | 10 | 10 | 10 | 10 |
| n_completed | 9 | **10** | 9 | **0** |
| n_retry_exhausted | 1 | **0** | 1 | **10** |
| legal_rate | 100% | **100%** | 100% | **0%** |
| first_legal_rate | 90% | **100%** | 90% | **0%** |
| bot_match_rate | 66.7% | **70.0%** | 55.6% | 0% |
| mean_eq_delta | −3.149 | **−2.834** | −5.594 | 0.000 (degenerate) |
| p_eq_geq_bot | 66.7% | 70.0% | 55.6% | 0% |
| mean_retry_count | 0.00 | 0.00 | 0.30 | 0.00 |
| empty_tool_rollout_rate | 0% | 0% | 50% | 100% |
| mean_tokens_in (chars) | 20,728 | 17,783 | 16,175 | 88,196 |
| mean_tokens_out (chars) | 2,246 | 1,027 | 2,159 | 16,353 |
| wall_time (s) | 81 | 65 | 177 | 1,523 |
| total tool calls | 26 | 34 | 13 | 0 |
| tool errors | 0 | 4 | 4 | — |

`rank128` does not produce a well-defined `mean_eq_delta`: all ten decisions retry-exhausted, so there is no committed play to grade. The `0.000` entry is the summary default, not a comparable number.

### Tool histograms

| tool | base | rank16 | rank64 | rank128 |
|:---|---:|---:|---:|---:|
| trick_winner_if | 17 | 26 (+4 mis-cased `TrickWinnerIf`) | 6 | 0 |
| is_legal | 5 | 0 | 4 | 0 |
| eq_outcome_distribution | 2 | 0 | 1 | 0 |
| contract_progress | 2 | 0 | 0 | 0 |
| what_beats_what | 0 | 4 | 2 | 0 |

Two load-bearing SFT effects at rank 16:

1. **`trick_winner_if` got reinforced** — 1.7 calls per attempted decision pre-SFT, **3.0 per decision post-SFT (rank 16)**. This is the same direction iter-3-rules showed, slightly stronger.
2. **`eq_outcome_distribution` was zeroed out** — base called it twice across the eval; rank-16 never calls it. A regression worth naming. Candlewax (E2) is the orthogonal lever that targets this surface directly.
3. **Casing drift** — rank-16 emits `TrickWinnerIf` (PascalCase) four times, which the tool dispatcher doesn't match. Four wasted calls per 30 = 13% tool-call leakage. The base model doesn't do this; SFT on 26 rows with preserved thoughts introduced the typo. Small, but present.

## Per-decision final plays

Same 10 decisions, same seed, plays across conditions. `*` = retry-exhausted (no commit). `✓` in "match" = burl_final_play equals bot_play.

| # | seed/decl/narr | legal | bot | bot_eq | base | rank16 | rank64 | rank128 |
|---|---|---|---:|---:|---|---|---|---|
| 0 | 900000/0/1 | 13,21 | 21 | −4.89 | * | **21** ✓ | 13 (Δ=−22.0) | * |
| 1 | 900000/0/2 | 5,20 | 5 | −22.75 | 5 ✓ | 5 ✓ | 5 ✓ | * |
| 2 | 900000/0/3 | 0,23 | 23 | −5.19 | 0 (Δ=−1.0) | 0 (Δ=−1.0) | 0 (Δ=−1.0) | * |
| 3 | 900000/1/0 | 22,24 | 22 | +17.12 | 22 ✓ | 22 ✓ | 22 ✓ | * |
| 4 | 900000/1/3 | 0,11 | 11 | +2.58 | 0 (Δ=−11.9) | 0 (Δ=−11.9) | 0 (Δ=−11.9) | * |
| 5 | 900000/2/2 | 10,26 | 10 | −8.48 | 10 ✓ | 10 ✓ | 10 ✓ | * |
| 6 | 900000/2/3 | 9,27 | 9 | +8.94 | 9 ✓ | 9 ✓ | 9 ✓ | * |
| 7 | 900000/3/0 | 24,25 | 25 | +12.08 | 24 (Δ=−15.5) | 24 (Δ=−15.5) | 24 (Δ=−15.5) | * |
| 8 | 900000/3/2 | 5,20 | 5 | −26.50 | 5 ✓ | 5 ✓ | 5 ✓ | * |
| 9 | 900000/3/3 | 0,16 | 0 | +6.45 | 0 ✓ | 0 ✓ | * (3 retries) | * |

Cross-condition play divergence (final play differs; treating retry-exhausted as a distinct outcome):

| pair | decisions that differ |
|:---|:---:|
| base vs rank16 | 1/10 (D0: base retry-exhausted, rank16 commits bot-match) |
| base vs rank64 | 2/10 (D0, D9) |
| rank16 vs rank64 | 2/10 (D0, D9) |
| rank16 vs rank128 | 10/10 |
| rank64 vs rank128 | 9/10 |

**Critical read: rank-16 is NOT byte-identical to base on this set.** On D0 the base model retry-exhausts (genuine protocol failure — 6 tool calls, no commit) while rank-16 commits `21` in three turns for a 0.0 eq-delta. That is a net gain the SFT produced. On the other eight decisions where both commit, plays match — but the *paths to commit* differ substantially (rank-16 uses far fewer tokens of output, 1027 vs 2246 mean, and different tool mixes).

This is the central finding: **once training data is untruncated, rank-16 preserve-thoughts SFT does produce a behaviorally distinct adapter from base**, unlike the iter-4-thoughts byte-identical null. The original hypothesis "more rank unlocks the thought-gradient signal" is not the right question anymore — the signal was already there at rank 16, it was just blocked at SFT time by the 2048-token ceiling.

## Why rank-64 and rank-128 collapse

rank-64 loses half its traces to `empty_tool_rollout_rate=50%` and regresses below base accuracy (55.6% vs 66.7%). The traces show three failure modes:

1. **Malformed tool JSON at rank 64** — the adapter emits `trick_winner_if({'domino_id': 0, 'trick_winner_seat': 3, ...})` with hallucinated kwargs that the tool dispatcher rejects (`unexpected keyword argument 'trick_winner_seat'`). Four of rank-64's thirteen tool calls failed this way. D9 exhausted retries by repeatedly re-emitting the same malformed call.
2. **Premature commits** — decisions 1 and 9 reached commit on zero or one tool call (D1: 0 tool calls, 1119 chars of think, straight to commit; D9: 3 retries, never produced a clean tool call). The policy is abandoning its own tool-use loop.
3. **Reasoning-trace rot** — the think blocks in rank-64 traces are noisy; the thought stream interleaves `<|tool_call>` tokens with malformed JSON, which the dispatcher can't resolve.

rank-128 is a complete collapse: zero valid tool calls across 10 decisions, all retry-exhausted, producing 16k chars of incoherent token-salad per decision ("1. 0-1), 02-0. 25.0. 1. (2), seats 0 in Play,0,00, 0/2 is the 5..."). The policy is no longer the Gemma base — it has been destroyed by the update.

The simplest hypothesis: **MLX-LM lacks gradient clipping**, and at rank 64 / 128 the LoRA subspace is large enough that the unclipped update on a 26-row corpus pushes the adapter weights far outside the base's effective manifold. rank-16 squeaks by because the subspace is small; rank-64 hits a regime where most tool-call calls still work but the output format drifts; rank-128 collapses the sampling distribution entirely.

## Interpretation — which hypothesis does the data support?

ITER4_PLAN §1 framed this as (a) capacity-bound (scaling rank unlocks the signal → invest in preserve-thoughts) vs (b) base-reflex dominance (outputs stay identical at all ranks → abandon preserve-thoughts, redirect to environment shape). Neither holds in its original form, because both are conditional on the byte-identical iter-4-thoughts result that we now know was a truncation artifact.

**What the data actually supports**:

1. **Untruncated preserve-thoughts works at rank 16.** rank-16 diverges from base cleanly (D0 goes from exhaust → match), pushes bot_match from 66.7% to 70.0%, and sharpens the output (mean out chars 2246 → 1027). The preserve-thoughts recipe is not dead; iter-4-thoughts was a methodology artifact, not a scientific null.
2. **"More capacity" is NOT the next lever.** Hypothesis (a) as originally written is actively disconfirmed: scaling rank from 16 to 64 regresses below base, and 128 collapses the policy. On this corpus size (26 rows) with this training stack (MLX-LM, no gradient clipping), rank 16 is at or near the sweet spot.
3. **This is the first preserve-thoughts adapter that earned its keep.** iter-4-thoughts at rank 16 on the (secretly truncated) Modal path was byte-identical to iter-3-rules. E1's rank-16 adapter on the same recipe at `max_seq=4096` produces a distinct, better-than-base policy. The difference is training-time exposure to the thought content, not rank.

Hypothesis (a) would have predicted monotone improvement with rank. Hypothesis (b) would have predicted byte-identical outputs at all ranks. The actual shape — rank-16 helps, rank-64+ hurts monotonically — fits neither and points at a third mechanism: **LoRA-LR × rank × corpus-size joint instability** on the MLX-LM stack. This is a training-methodology finding, not a "more thoughts = better" or "thoughts are reflex-locked" finding.

## Recommended next step

**Do not run iter-5-hybrid as originally specified** (rank + N=100). The ITER4_PLAN recipe assumed hypothesis (a) was either right or falsifiable by rank alone. It's neither. The E1 data says the next lever is **more data at current rank**, not more rank at current data.

Specifically:

- **E3 revised (top priority)**: rank-16 preserve-thoughts adapter on N=100 rollouts with iter-3-rules shape + EQ-gate. Same MLX-LM recipe that produced E1's rank-16 winner. Target: does the behavior gap over base widen when the adapter sees 4× the thought-bearing traces? If yes, the preserve-thoughts lineage has a real scaling curve; if no, we've saturated at N=26 and corpus-quality (not quantity) is the next lever.
- **E2 Candlewax (unaffected)**: E2 is about environment-shape, orthogonal to this capacity finding. The 0% `eq_outcome_distribution` rate at rank-16 (regression from base's 2 calls) is independent motivation for E2 — if the tool surface itself is illegible, no SFT recipe at any rank will teach Gemma to reach for it.
- **Before scaling rank again, fix the training stack**: gradient clipping in `burl/train/star_mlx.py` is the blocking issue for any future rank > 16 experiment. Without it, this collapse curve is reproducible by construction.
- **Cross-link to T13 Candlewax writeup**: the `eq_outcome_distribution` regression at rank-16 (2 base calls → 0 adapter calls) is evidence that Candlewax's redesign matters regardless of how well the SFT lineage scales. Independent work, mutually informing.

## Out of scope (what this eval does NOT tell us)

- **Is rank-16 preserve-thoughts better than iter-3-rules?** Not tested here. Both were trained on the same 26-row corpus; iter-3-rules hit 90% bot-match at N=10 on a different seed-set via the Modal recipe, and E1's rank-16 hit 70% on this seed-set. Seeds and eval configurations differ; a head-to-head at matched seeds is a cheap follow-up.
- **Would gradient clipping save rank-64/128?** Plausible, but not tested. Adding clipping to `star_mlx.py` is a ~1-hour change and would let us re-test the scaling question without methodology noise.
- **Does N=10 hold at N=30?** The 3.3-pp lead of rank-16 over base (70% vs 66.7%) is one decision's worth of signal. A larger held-out set (N=30 or more) would pin whether rank-16's win is real or within-noise.

## Budget

- Training: three MLX-LM runs on M5 Max local, ~30 min wall each, zero marginal cost (no cloud spend).
- Eval: four N=10 runs on local MLX serve, 81+65+177+1523 = 1,846 seconds (31 minutes) of total wall time. Zero marginal cost.
- Total E1: **$0.00 cloud spend**.

## Artifacts

- `burl/eval/results/e1/base/{summary.json,traces.jsonl,report.md}`
- `burl/eval/results/e1/rank16/{summary.json,traces.jsonl,report.md}`
- `burl/eval/results/e1/rank64/{summary.json,traces.jsonl,report.md}`
- `burl/eval/results/e1/rank128/{summary.json,traces.jsonl,report.md}`
- `burl/train/star_mlx.py` — training script (note: no gradient clipping at time of writing; fix is a prerequisite for any future rank > 16 experiment)
- Adapter weights: local only (not pushed to HF), three LoRA checkpoints under the training output directory

## Reading list for a returning session

1. This file — the capacity-sweep result and reframing.
2. `burl/experiments/iter4_thoughts_eval_writeup.md` — the byte-identical null result that this sweep reopens.
3. `burl/experiments/iter3_rules_eval_writeup.md` — the current winner adapter, same corpus as E1 but Modal/UnSloth recipe.
4. `burl/ITER4_PLAN.md` §1 — the original hypothesis framing (now partially superseded; §2 Candlewax and §3 Pareto still hold).
5. `burl/experiments/iter5_e2_candlewax_eval_writeup.md` (T13, in progress) — the orthogonal environment-shape track, cross-links to the `eq_outcome_distribution` regression finding here.
