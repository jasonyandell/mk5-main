Reviewed against code on 2026-07-07 — no issues found.

Verified: harvest-1/harvest-2 bucket tables match `HARVEST_SUMMARY.md` in `scratch/belief_trajectory_rollout/harvest_batched_20260425_072910/` and `.../20260426_031338/` exactly; the run-4 three-way table matches `STAR_EVAL_REPORT_2026-04-26.md` §"Run-4 fold" verbatim; 3893 rows = 3118 train + 775 val in the run-4 `train.log`; adapter/eval/rescore paths all exist; `burl/eval/star_eval_report.py` exists in-repo. Note: `scratch/` artifacts live in the main checkout (gitignored), not this worktree — verified against `/Users/jason/code/mk5-main/scratch/`.

- The page's ILLEGAL=0 correction is right, but the raw `HARVEST_SUMMARY.md` still says "ILLEGAL 800 (28.6%)" — a cheap fix is the page's own suggestion #4 (tagger denominator), or a one-line addendum in the summary files to stop future re-confusion.
- The N=560 vs paired-180 discrepancy for run-3c regret (2.30 vs 1.92) is still unresolved; a cheap probe is rescoring run-3c on indices 180–559 only.
