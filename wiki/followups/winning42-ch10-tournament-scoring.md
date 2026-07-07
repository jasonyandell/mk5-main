# winning42-ch10-tournament-scoring — audit 2026-07-07

## Corrections

- Page said genuine mark-vs-point flips were "3.6% of decisions (10/280)"; the true count is 12/280 = 4.3% genuine flips, with 10 of the 12 detector-endorsed (primary evidence: recomputation from `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/action_mark_ev_scalars.csv` — 65 flips, 12 with mark_gain > 0, mean cost 1.32 pts, mean gain 0.030 marks; the endorsed subset of 10 is in `detector_endorsed_flips.csv`, mean cost 1.46, mean gain 0.032. The Wave 1 wiki page lines 18-21 agrees but was edited in the same audit pass, so it is corroborating, not primary).
- Page said the genuine flips "concentrate at mid-hand"; they concentrate early-hand — 10 of the 12 genuine flips are `score_bucket=early_hand` (same CSV recomputation; the Wave 1 page summary likewise says the effect is "clearest for early-hand decisions"). The mid-hand enrichment finding applies to all 65 nominal flips, not the genuine subset. Fixed in second-pass review.

## Verified clean

- Claim-ledger metrics (94.5% early terminal, 4.3805 mean tricks saved, 84.989% partial erasure, 41 severity values, 23.814062 trick reduction, 15% advancement disagreement) all match `w42/phase4_scoring_objective_tests/claim_summary.csv` and `summary.json`.
- Wave 2 numbers (259,618 rows, 14,000 paired decisions, 56/60/70/71/71% flip ladder, bid=84 plateau via shared threshold_q=42, sixes 74.8%, bidder 74.6%) match wiki/experiments/w42-bookval-v1-wave2-ch10-action-level.md and -bid-aware-atlas.md.

## Unverifiable

- The book source path `scratch/winning42/winning42.with_figures.md` (lines 4243-4331) does not exist in this worktree — scratch/ is gitignored and the file lives only in the main working copy. Not treated as an error.

## Follow-ups

- The mark-ladder flip table stops at bid=84; a cheap probe would extend to 126/168 to confirm the threshold_q-not-multiplier story holds when threshold_q saturates.
- The artifact README at `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/README.md` (lines 44-56) still says 10 genuine flips (15.4%) / 55 zero-gain (84.6%) with mean gain 0.044 — stale relative to its own `action_mark_ev_scalars.csv` (12/53, mean gain 0.030). Out of scope for a wiki-only pass; the README should be regenerated or annotated.

## Review (second pass, 2026-07-07)

- The headline correction stands, now grounded in primary evidence: recomputing top-1 flips from `w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/action_mark_ev_scalars.csv` gives exactly 65 flips, 12 with positive mark_gain (12/280 = 4.3%), 53 zero-gain (81.5%), mean cost 1.3188, mean gain 0.0295; `detector_endorsed_flips.csv` contains 10 of the 12 (mean cost 1.459, mean gain 0.0322). The first pass cited only the Wave 1 wiki page, which was edited in the same audit commit (45a7e35) — circular; the raw CSV settles it, including against the stale artifact README (which says 10/55).
- Amended the page: "concentrate at mid-hand" → "concentrate early-hand (10 of 12)" with bidder/partner count-safe lines and setter lead-pressure opening plays. Evidence: `score_bucket` on the 12 recomputed genuine flips (10 early_hand, 2 mid_hand; seats: 5 bidder, 3 bidder_partner, 4 setter). The first pass preserved this pre-existing error while editing the same sentence. Mid-hand enrichment is a property of all 65 nominal flips (via ch10-early-terminal 1.21x), not the genuine subset.
- Re-verified all "Verified clean" numbers against artifacts: `w42/phase4_scoring_objective_tests/summary.json` (early_terminal_rate 0.945, mean_tricks_saved 4.3805, 41 severity values, timed_advancement_disagreement_rate 0.15) and `claim_summary.csv` (partial_erasure 0.84989, trick_reduction 23.814062); Wave 2 ladder/plateau/sixes/bidder numbers match `wiki/experiments/w42-bookval-v1-wave2-ch10-action-level.md`. All stand.
- Dropped the "glossary note on the Wave 1 page" follow-up: the Wave 1 page already carries the disambiguating parenthetical "(10 of the 12 are detector-endorsed…)" in its summary and the 12-vs-53 split in its findings — the suggestion is already done. Kept the 126/168 ladder probe (no 126/168 flip data exists in either Wave 2 page). Added the stale-README follow-up.
