## Corrections

- Page said the eval had "9 completed for base/rank-16/64"; the writeup's completion counts are base 9/10, rank-16 10/10, rank-64 9/10, rank-128 0/10 — rank-16 was the only condition to complete all 10 (evidence: burl/experiments/iter5_e1_capacity_eval_writeup.md, "Headline metrics" table).

## Follow-ups

- Source discrepancy worth reconciling: the E1 writeup and the ceca203 commit message say the old truncation ceiling was `max_seq_length=2048`, while [[decisions/sft-max-seq-length]] and the iter-4 pages say TRL's default 1024. The page follows the 1024 account; one of the two source docs is wrong.
- The eval artifacts listed in the writeup (`burl/eval/results/e1/{base,rank16,rank64,rank128}/`) are not committed to the repo — headline numbers are verifiable only against the writeup and commit message, not raw summary.json.
- The "verified on 147 rows — rank-64 still diverges at LR peak" claim traces only to the ceca203 commit message; no writeup or artifact for that 147-row run exists in the repo. A cheap probe: rerun rank-64 on the larger corpus after adding gradient clipping to `burl/train/star_mlx.py`.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. The completion-count fix matches the "Headline metrics" table in `burl/experiments/iter5_e1_capacity_eval_writeup.md` (n_completed: base 9, rank16 10, rank64 9, rank128 0); the original page's "9 completed for base/rank-16/64" was indeed wrong for rank-16, and the edit touched nothing else on the page.
- Follow-ups all check out: the 2048-vs-1024 discrepancy is real (writeup + ceca203 say 2048; fix commit edf86e9 and `wiki/decisions/sft-max-seq-length.md` say TRL default 1024); `burl/eval/results/e1/` is absent from the repo; "147 rows" appears only in the ceca203 commit message; `burl/train/star_mlx.py` has no gradient clipping, so the probe suggestion is still live.
