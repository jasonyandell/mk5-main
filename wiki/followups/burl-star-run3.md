# burl-star-run3 — audit 2026-07-07

## Corrections

- Page said the min-300 corpus "yields ~755 train + ~169 val rows"; the corpus manifest says 820 train + 205 val decisions = 2686 train + 665 val assistant rows (evidence: scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/manifest.json).
- Page said run-3b eval had "113/560 decisions completed"; the eval dir holds 130 parseable trace_summary.json files (decisions 0–129), and the rescore report used all 130 — 113 was only the live-snapshot count (evidence: scratch/belief_trajectory_rollout/star/STAR_EVAL_REPORT_2026-04-26.md "Inputs" note; the n=113 metrics 60.2%/2.89 kept as the labeled snapshot values).

## Verified (spot-checked, no change)

- run-3b best val 0.238 @ iter 749, 32 min; run-3c best val 0.268 @ iter 949, 45 min (train.log JSON tails in adapters/run3b_*/ and run3c_*/).
- run-3c full-560 eval: 373/560 match (66.6%), legal 100%, mean signed Δ −1.984, |Δ| 2.111 (eval summary.json); thought-block presence 537/560 confirmed by grepping `<|channel>thought` in all 560 transcript.live files.
- Rescore table (k1_pass 70.5%, regret 2.165 vs base 2.295, FORCED_COMMIT 191, near_tie 74.1%) matches STAR_EVAL_REPORT_2026-04-26.md; paired n=180 table matches base_vs_run3c_paired_n180.md; FC diagnosis (134/158 at n_turns=8, 2.04× belief_trajectory, n_legal=2 33% vs 23%) matches FORCED_COMMIT_DIAGNOSIS_2026-04-26.md.
- star_mlx.py has --resume/checkpoint_state.json and best_on_crash crash snapshot as described; all Pointers paths exist (scratch/ artifacts live in the main repo tree, gitignored).

## Follow-ups

- The "1062-row strict pool" phrasing conflates decisions and rows throughout the Recipe section; a cheap cleanup pass could standardize on decisions vs assistant-rows terminology. (Second pass confirms: pool is 1062 *decisions* pre-filter — manifest `n_decisions_total` 1025 + `decisions_emptied_by_filter` 37.)

## Review (second pass, 2026-07-07)

- Verified — corrections stand.
- Correction 1 re-derived from `scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/manifest.json`: 1025 decisions (37 emptied), 820/205 train/val decisions, 2686/665 train/val rows — all exact. The original "~755 train + ~169 val rows" matches nothing in the manifest.
- Correction 2 re-derived: `find` counts exactly 130 `trace_summary.json` in `eval/run3b_eval_seq560_20260425_164239/` spanning decisions 0–129 (`decision_130/` exists but has no trace_summary); rescore report table gives run-3b @ 130 match_bot 58.5%, mean_abs_delta 2.978; the 113/60.2%/2.89 figures trace to `RUN3BC_LIVE_SNAPSHOT.md` line 9, confirming "113 was the live-snapshot count."
- Spot-re-checked the Verified section: train.log JSON tails (best_val 0.2378 @ 749 / 0.2679 @ 949, 1934s/2725s), run-3c summary.json (373/560, legal 560/560, −1.9837/2.1106), rescore report (70.5%, 2.165 vs 2.295, FC 191, near_tie 74.1%), `base_vs_run3c_paired_n180.md` (3.1316 → 1.9153), `FORCED_COMMIT_DIAGNOSIS_2026-04-26.md` (134/158 @ n_turns=8, 322/158 = 2.04 belief_trajectory calls, n_legal=2 33% vs 23%), and `burl/train/star_mlx.py` (`--resume`/`checkpoint_state.json`/`best_on_crash`). All match.
- Minor, not amended: page says run-3b/3c "early-stopped at iter 849/1149" while the logs print "Iter 850"/"Iter 1150" at the stopping eval — an off-by-one internal to the trainer's own logging (it also prints best @ 749 for an eval labeled Iter 750), so the page's convention is as defensible as the log's.
