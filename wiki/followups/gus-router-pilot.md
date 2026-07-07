# gus-router-pilot — audit 2026-07-07

## Corrections

- Page said the no-oracle prerequisite could be met "by running K=50+ at inference"; source says K=100+ — K=50 is the setting that already failed (PIMC-Q-K50 regret 1.47 vs 1.39 baseline), and the receipt estimates K=100+ gives at best ~2x improvement over K=50 (evidence: commit a09ef43 diff to gus/PRACTICALITIES.md, receipt 14).

## Verified

- Regret table (1.15/0.56/0.49 oracle; 1.39/1.47/1.48 pimc-q; 1.48/1.55/1.69 next-best), 560 held-out decisions, 1.39 baseline, detector trained on 30% of chunks 0-299, 7-fixed/6-introduced blunder breakdown, mid-game concentration (decisions 4, 8, 10) — all match commit messages eba5103/a09ef43 and gus/PRACTICALITIES.md receipt 14. gus/eval/router.py exists.

## Follow-ups

- The page's 3-row table omits the 10%/15% flag rows present in receipt 14 (0.97/0.82 oracle); could add for completeness.
- Detector precision at flag time is the real lever (6 false-positive-induced blunders); a cheap probe is re-running the router with a precision-tuned detector threshold rather than flag% quotas.

## Review (second pass, 2026-07-07)

- The K=50+ → K=100+ correction stands: original page's "K=50+ at inference" contradicted the source (K=50 is the failed setting, regret 1.47 vs 1.39 baseline), and the replacement text matches receipt 14 verbatim intent ("K=100+ at inference (cheap but only 2× improvement over K=50 at best)"). Evidence: `git show a09ef43 -- gus/PRACTICALITIES.md`.
- The added training clause ("loss penalizing cross-world Q variance") also matches receipt 14 ("change train_v2_voids loss to query Q_head on K worlds and penalize cross-world variance").
- Amended one citation: the Implication paragraph retained "(commit message @ a09ef43)" but the K=100+ and ~2× figures come from the receipt-14 file content in that commit's diff, not the commit message (which says only "much bigger K at inference"). Changed to "(PRACTICALITIES receipt 14 @ a09ef43)", matching house citation style (cf. wiki/experiments/gus-lamir1-piopp.md, gus-belief-calibration-diagnostic.md).
- Independently re-verified the "Verified" list against eba5103's commit message and receipt 14: regret table rows (1.15/0.56/0.49 · 1.39/1.47/1.48 · 1.48/1.55/1.69), 560 decisions, 1.39 baseline, detector on 30% of chunks 0-299, 7-fixed/6-introduced, mid-game concentration (dec 0-12, esp. 4/8/10), end-game 24-27 never flagged, `gus/eval/router.py` exists (615 lines, added in eba5103). All hold.
- Both Follow-ups suggestions kept: the 10%/15% rows (0.97/0.82 oracle) exist in receipt 14 and are absent from the page's table; the precision-tuned-threshold probe is sensible and was never run as such — later router work (`gus/eval/*qmean_router.py`, commit 3ad63f9f) routes on Q-mean instead, a different signal than a precision-tuned GBM detector threshold.
