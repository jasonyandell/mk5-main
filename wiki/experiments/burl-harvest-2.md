---
title: Burl harvest-2 + run-4 — first STaR self-sharpening test
kind: experiment
first_seen: 74464e9
last_updated: 74464e9
status: active
---

## What

A 2000-decision [[burl]] harvest collected on fresh seeds from `gus/data/corpus_train_chunk_4100-4199.pt` with the [[burl-star-run3]] **run-3c adapter** in the rollout loop (the [[preserve-thoughts]] winner from iter-1). Same `D_required_first` variant, same batched harness from [[burl-2000-harvest]], `--turn-cap 12` (bumped from cap=8 used in harvest-1, per the FORCED_COMMIT diagnosis). Output: `scratch/belief_trajectory_rollout/harvest_batched_20260426_031338/`.

Followed by **run-4 train**: filter-only SFT on the harvest-2 strict pool with the same recipe as run-3c (rank=8, lr=3e-5, 1 epoch, preserve-thoughts ON). Adapter: `scratch/belief_trajectory_rollout/star/adapters/run4_20260426_163325/`.

Followed by **run-4 eval**: same 180 paired indices `gi=0..179` as the [[burl-star-run3]] in-distribution eval, batched k=6, `--turn-cap 12`, `max_tokens=8192`, three-way fold against base@180 and run-3c@180 in `STAR_EVAL_REPORT_2026-04-26.md` §"Run-4 fold".

## Why

The first [[star]] **self-sharpening** test for Burl: does run-3c rolling out generate a meaningfully different corpus than base-Burl rolling out, and does training on that sharper corpus produce a better adapter? This is the iter-1 → iter-2 question the project has been pointing at since [[iter-without-regression]].

If filter-only iter-2 closes the regret gap further, the simple loop "harvest → SFT → harvest with new adapter → SFT" is a viable curriculum. If iter-2 plateaus or regresses, the next move is changing the loss target ([[r1-rationalization]] on `BURL_BREAKS_CONSENSUS`, FORCED_COMMIT-as-negative).

## Headline result — strategic distribution unchanged, play quality plateaued

**The harvest is essentially indistinguishable from harvest-1 at the bucket level.** Strict pool moved 1062 → 1075 (+13 decisions, +1.2%). What did change: row count moved 2686 → 3893 (+45%) because run-3c is a chattier rollout (~3.16 → ~3.62 rows/decision). FORCED_COMMIT dropped 10.9% → 9.2% (the cap-bump from 8 → 12 working as intended).

**Run-4 eval is statistically tied with base on play quality and regresses ~1.05 in regret vs run-3c.** What it does deliver is clean commit discipline (FC 9.4% vs run-3c's 33.9%, comparable to base's 11.1%). See "Run-4 result" below.

So the bottom line: **filter-only iter-2 produced an adapter that essentially matches base on regret while repairing the [[commit-discipline-collapse]] that run-3c introduced.** That repair is real, but it is not "iter-2 beats iter-1 on play quality."

## Bucket comparison (harvest-1 vs harvest-2, n=2000 each)

| Bucket                      | harvest-1 (n) | harvest-2 (n) | Δ (dec) | Class |
|-----------------------------|--------------:|--------------:|--------:|-------|
| ALL_AGREE_CORRECT           |          860  |          873  |    +13  | gold (trivial) |
| BURL_ALONE_FIXES            |           14  |           13  |     −1  | gold (sharp) |
| BOTH_FIX                    |           52  |           37  |    −15  | gold |
| BURL_INDEPENDENT_RIGHT      |           88  |          103  |    +15  | gold (sharp) |
| BURL_FOLLOWS_PI_RIGHT       |           48  |           49  |     +1  | gold |
| **STRICT POOL TOTAL**       |     **1062**  |     **1075**  |   **+13** | — |
| → non-trivial gold          |          202  |          202  |      0  | — |
| BURL_BREAKS_CONSENSUS       |          299  |          295  |     −4  | loss (sharp [[r1-rationalization]]) |
| BURL_INDEPENDENT_WRONG      |          148  |          165  |    +17  | loss |
| ALL_AGREE_WRONG             |          100  |          110  |    +10  | loss |
| BURL_PARROTS_PI_WRONG       |           57  |           50  |     −7  | loss |
| QMEAN_ALONE_FIXES           |           37  |           31  |     −6  | loss |
| BURL_PARROTS_QMEAN_WRONG    |           41  |           45  |     +4  | loss |
| BURL_DRIFTS_FROM_PI         |           37  |           46  |     +9  | loss |
| FORCED_COMMIT               |          219  |          183  |    −36  | guarded (cap=12 win) |
| **ILLEGAL**                 |            0  |          800  |  **+800** | **regression** |

**The ILLEGAL row is a regression flag worth honest framing.** Harvest-2 has 800 decisions (28.6%) with no `trace_summary.json` on disk — i.e. 800 decisions silently failed to produce a usable trace. Harvest-1 had zero. The bucket counts above are normalized to the 2000 decisions that did succeed, so the strategic distribution comparison is apples-to-apples on the surviving corpus. But the harness yield regressed from 100% → 71.4%. Possible drivers: longer turn cap × longer effective sequences × cap-bump path causing OOM-quarantine without ledger entry, or run-3c's chattier rollout occasionally exceeding `max_tokens` mid-thinking-block (the [[burl-2000-harvest]] v1 contamination pattern). Not yet root-caused; **noted as the harvest-2 footgun for follow-up**.

The non-trivial gold count is **identical at 202 rows** — the structural distribution of "where Burl is sharper than the bot consensus" is essentially fixed by the seed pool, not by the rollout policy. Same goes for `BURL_BREAKS_CONSENSUS` (299 → 295). Run-3c-as-rollout did not preferentially surface the kinds of decisions that filter-only training cares about.

## Run-4 result — three-way paired n=180

Eval: `scratch/belief_trajectory_rollout/star/eval/run4_eval_n180_cap12_run4_20260426_163325_20260426_181558/`. Wall 45.6 min batched. Rescored via `burl.eval.star_eval_report` against the K=200 belief-sampled E[Q] oracle ([[regret-eval]]).

| Metric                  | base@180 | run-3c@180 | run-4@180 | r3c−base | r4−base | r4−r3c |
|-------------------------|---------:|-----------:|----------:|---------:|--------:|-------:|
| k1_pass (Δ ≥ 0)         |   60.6%  |    69.4%   |   62.8%   |   +8.9pp |  +2.2pp | −6.7pp |
| match_oracle            |   55.6%  |    63.3%   |   58.9%   |   +7.8pp |  +3.3pp | −4.4pp |
| match_bot               |   58.9%  |    65.6%   |   58.3%   |   +6.7pp |  −0.6pp | −7.2pp |
| mean_signed_delta       |  −3.045  |   −1.829   |  −2.880   |   +1.216 |  +0.166 | −1.051 |
| **mean_oracle_regret**  |  **3.13**|  **1.92**  |  **2.97** |   −1.216 |  −0.166 | **+1.051** |
| **forced_commit_rate**  |  **11.1%**| **33.9%** |  **9.4%** |  +22.8pp | −1.7pp  | **−24.4pp** |
| thought_block_rate      |     —    |    ~95%    |   92.8%   |     —    |    —    |    —   |

Per-decision regret comparison:
- run-4 vs run-3c: 27 better, 43 worse, 110 same — net 16 worse than run-3c
- run-4 vs base: 39 better, 24 worse, 117 same — net 15 better than base, but +0.166 mean regret delta is well within noise

## Reading

Filter-only iter-2 is a **two-axis result**:
1. **Play quality (regret)**: tied with base, regresses ~1.05 vs run-3c. The "iter-2 produces meaningfully better adapter than iter-1" hypothesis is not supported.
2. **Commit discipline (FC rate)**: clean — back to base-comparable 9.4%. The [[commit-discipline-collapse]] that run-3c introduced is **repaired without retraining the loss target**.

The cleanest interpretation: filter-only SFT on the same-shape corpus, no matter which policy generated the rollouts, asymptotes near base play quality. Run-3c's regret win was carrying a hidden cost (FC inflation) that the rescorer caught only post-hoc. Run-4 trades the regret win back to recover commit discipline; it does not extract additional signal beyond what run-3c already learned and forgot.

The "[[iter-without-regression]]" milestone holds at the *behavioral* level (no new pathology — thought-block emission held at 92.8%, schema clean, FC repaired) — but it does not hold at the *play-quality* level. Iter-2 was not a strict improvement.

## What this implies for next steps

The plateau strongly motivates **changing the loss target**, not just iterating filter-only on the same shape:

1. **[[r1-rationalization]] on `BURL_BREAKS_CONSENSUS` (n=295 in harvest-2, 299 in harvest-1)**: condition the model on the oracle answer and learn the *reasoning* that gets there. The sharpest STaR-shaped target the project has.
2. **FORCED_COMMIT-as-negative**: include FC decisions in training with a negative signal (counterfactual: the bot would not have forced). Aimed at the FC-vs-regret tradeoff that run-3c and run-4 expose as a real two-axis Pareto.
3. **Ratchet curriculum to trick 5**: harvest D_trick5_first to get away from the trick-0 bias of the current corpus.
4. **Investigate the harvest-2 ILLEGAL=28.6% regression** before trusting any harvest-3 numbers.

## Caveat

Only ~4000 total decisions across two harvests. Zeb plateaued only after hundreds of thousands of games. We may be at the edge of what filter-only SFT can extract, and the iter-2 plateau here may not generalize to "iter-N plateau" once the loss target changes. But the data we have so far says the simple "harvest → SFT → re-harvest → SFT" loop does not compound at this scale.

## Open questions

- Does run-3c-rolled-out at trick 5 / trick 0 produce different bucket distributions, or is the harvest-2 distribution-parity a property of the seed pool independent of rollout policy?
- What drives the harvest-2 ILLEGAL=28.6% regression — cap-bump path OOM, run-3c chattiness blowing `max_tokens`, or something in the harvest harness path?
- Is the run-3c regret win on this paired n=180 sustained at larger N, or is the −1.05 vs run-4 partly a paired-180 sampling artifact? (Run-3c was eval'd at N=560 originally; its regret on the wider set was ~2.30, vs 1.92 on this paired 180 — the first 180 indices may be a slightly easier subset for the FC-heavy run-3c policy.)
- Does R1-rationalization on the 295 BBC rows compound on top of run-4, or does the loss target change need to start from run-3c (FC-and-all)?

## Pointers

- Harvest dir: `scratch/belief_trajectory_rollout/harvest_batched_20260426_031338/`
- Harvest summary: `…/HARVEST_SUMMARY.md`
- Run-4 adapter: `scratch/belief_trajectory_rollout/star/adapters/run4_20260426_163325/`
- Run-4 eval: `scratch/belief_trajectory_rollout/star/eval/run4_eval_n180_cap12_run4_20260426_163325_20260426_181558/`
- Run-4 rescore JSON: `…/star_rescore.json`
- Three-way fold writeup: `scratch/belief_trajectory_rollout/star/STAR_EVAL_REPORT_2026-04-26.md` §"Run-4 fold"

## Links

[[burl]] [[star]] [[burl-2000-harvest]] [[burl-star-run3]] [[preserve-thoughts]] [[commit-discipline-collapse]] [[regret-eval]] [[learned-by-playing]] [[iter-without-regression]] [[r1-rationalization]] [[batched-eval-resilience]] [[batched-harvest-resilience]]
