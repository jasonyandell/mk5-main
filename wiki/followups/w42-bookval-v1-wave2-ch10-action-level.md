Reviewed against code on 2026-07-07 — no issues found.

All headline numbers verified against `w42/book_validation_v1/wave2/probes/t42-8na4_ch10_action_level/` (summary.json, action_flip_rates.csv, multiplier_strategic_effect.csv, slice_breakdown.csv): flip rates per bid, 95% CIs, seat-role and declaration breakdowns, seeds 9000-9049, 14,000 decisions, 2,000 bootstrap iterations all match.

Follow-up suggestions:

- The declaration table on the page lists 7 of the 10 declarations (blanks 69.3%, ones 68.4%, threes 68.5% at bid=84 are omitted); could add them for completeness — the "all above 68%" claim still holds.
- summary.json's caveat that at bid=42 p_make=0 for all actions (mark_ev = -mm always, argmax by ties) is a sharp detail worth surfacing on the page — the zero-exception mark_ev==p_make agreement at bid 42/84 is partly tie-breaking, not discrimination.
- A cheap next probe: flip-rate vs trick number (early vs late decisions) to see whether threshold_q sensitivity concentrates in the endgame.
