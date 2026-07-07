# w42-jud-v1 — audit

Reviewed against code on 2026-07-07 — no issues found.

Verified: JP1 −4.371 [−4.547,−4.193] and point margin −6.103 (`ab_r0_full512_summary.json`); JP2 −1.09 [−1.496,−0.691]; loop trajectory −4.37/−4.38/−4.16/−3.72/−4.04 and offense 64%→49%, made 32.8%→39.7% (`loop_metrics.json`); playdiag r0 −3.35 vs r4 −3.44; JS1 −1.16 [−1.55,−0.77], made 62.7%; JS2 −1.05, gain +0.11; JS3 play −1.41, full −1.43 [−1.68,−1.16]; r5 CE 2.09 and 600 games/6612 hands (`jud_v1_predictions.md`); Zeb-protocol −1.39/30.9% and −2.73/15.2%; gus pilot +0.59 [−0.19,+1.39]. Net dims (350=63+28+259, 512×512→43) and JudPlay greedy sign-flip mechanism confirmed in `champion/jud_net.py` / `arena/jud_play.py`. Commits f550205/9d30b25/e596205/3ac03de all exist as described.

## Follow-ups

- The per-trick MAE 8.6→3.5 figure is quoted from `jud_build_report.md`; the underlying eval JSON (`jud_net_eval_round0.json`) could be linked directly for provenance.
- A cheap next probe before v2: JS2 showed worlds aren't the constraint — a leaf-bias probe (same search, oracle leaf vs jud leaf, small n) would quantify exactly how much of the remaining −1.16 gap is leaf vs rollout policy.
