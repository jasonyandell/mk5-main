Reviewed against code on 2026-07-07 — no issues found.

Verified: MarginNet/ValueBidder classes and the `margin[...]` CLI registration exist as described; round-0 A/B (−1.44 [−1.88, −0.95], 36/128, +5.66 pts/hand, offense 75.2%, made 49.9%/75.4%), canonical r4 (−0.07 [−0.66, +0.49], 65/128, +7.01), definitive 512 (−0.01 [−0.28, +0.25], 258/512, +7.23), the round table (r1 −0.31 … r4 +0.22, notrump 47.5%→1.5%, gap +0.259), ECE 0.046 (N=568), A2 numbers (−1.82 / −0.38 with offense/made shifts), and the addendum's head_8 +0.38 [+0.09, +0.67] all match the JSON artifacts in champion/evidence/jud_v0/.

- `scratch/jud-v0/loop/run_loop.py` is absent from this worktree because scratch/ is gitignored, but exists in the main checkout — consider mirroring the loop script into champion/evidence/jud_v0/ (or forge/) so the experiment's driver survives scratch cleanup.
- The predicted-exceedance curve vs oracle MAE claims (0.020 vs 0.118) were not independently recomputed; they rest on step2b_report.md — a cheap probe would be a small script that re-derives both MAEs from margin_net_eval.json + optimism_gap.json.
