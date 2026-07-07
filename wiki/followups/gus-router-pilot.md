# gus-router-pilot — audit 2026-07-07

## Corrections

- Page said the no-oracle prerequisite could be met "by running K=50+ at inference"; source says K=100+ — K=50 is the setting that already failed (PIMC-Q-K50 regret 1.47 vs 1.39 baseline), and the receipt estimates K=100+ gives at best ~2x improvement over K=50 (evidence: commit a09ef43 diff to gus/PRACTICALITIES.md, receipt 14).

## Verified

- Regret table (1.15/0.56/0.49 oracle; 1.39/1.47/1.48 pimc-q; 1.48/1.55/1.69 next-best), 560 held-out decisions, 1.39 baseline, detector trained on 30% of chunks 0-299, 7-fixed/6-introduced blunder breakdown, mid-game concentration (decisions 4, 8, 10) — all match commit messages eba5103/a09ef43 and gus/PRACTICALITIES.md receipt 14. gus/eval/router.py exists.

## Follow-ups

- The page's 3-row table omits the 10%/15% flag rows present in receipt 14 (0.97/0.82 oracle); could add for completeness.
- Detector precision at flag time is the real lever (6 false-positive-induced blunders); a cheap probe is re-running the router with a precision-tuned detector threshold rather than flag% quotas.
