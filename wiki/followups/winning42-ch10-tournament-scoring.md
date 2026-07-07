# winning42-ch10-tournament-scoring — audit 2026-07-07

## Corrections

- Page said genuine mark-vs-point flips were "3.6% of decisions (10/280)"; the Wave 1 source says 12/280 = 4.3% genuine flips, with 10 of the 12 detector-endorsed (evidence: wiki/experiments/w42-bookval-v1-wave1-mark-utility-transform.md lines 18-21, 70).

## Verified clean

- Claim-ledger metrics (94.5% early terminal, 4.3805 mean tricks saved, 84.989% partial erasure, 41 severity values, 23.814062 trick reduction, 15% advancement disagreement) all match `w42/phase4_scoring_objective_tests/claim_summary.csv` and `summary.json`.
- Wave 2 numbers (259,618 rows, 14,000 paired decisions, 56/60/70/71/71% flip ladder, bid=84 plateau via shared threshold_q=42, sixes 74.8%, bidder 74.6%) match wiki/experiments/w42-bookval-v1-wave2-ch10-action-level.md and -bid-aware-atlas.md.

## Unverifiable

- The book source path `scratch/winning42/winning42.with_figures.md` (lines 4243-4331) does not exist in this worktree — scratch/ is gitignored and the file lives only in the main working copy. Not treated as an error.

## Follow-ups

- The Wave 1 "10 genuine flips" vs "12 genuine flips" distinction (detector-endorsed subset vs all positive-mark-gain flips) is easy to conflate; a one-line glossary note on the Wave 1 page could prevent future drift.
- The mark-ladder flip table stops at bid=84; a cheap probe would extend to 126/168 to confirm the threshold_q-not-multiplier story holds when threshold_q saturates.
