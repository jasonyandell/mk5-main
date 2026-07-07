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

- The "1062-row strict pool" phrasing conflates decisions and rows throughout the Recipe section; a cheap cleanup pass could standardize on decisions vs assistant-rows terminology.
