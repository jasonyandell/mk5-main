# Follow-ups: w42-bookval-v1-wave1-cross-ai-agreement

Reviewed against code and artifacts on 2026-07-07.

## Corrections

- Page said the 11 four-way-split decisions appear "predominantly in no-trump and doubles-suit late-trick positions with 4–5 legal actions"; `divisive_decisions.csv` (n_distinct_picks==4) shows 3× twos, 3× sixes, 2× fives, 2× doubles-suit, 1× no-trump, at trick_idx 0–3 with 4–7 candidates (evidence: w42/book_validation_v1/wave1/t42-m2i7_cross_ai_agreement/divisive_decisions.csv).
- Page said `ch05_setter_pressure_regime` "fires on 58% of setter decisions and has a 58% conflict rate with EV"; per_action_source_picks.csv shows it fires on 100% of setter action rows and 60.2% of fired rows are non-EV-best (evidence: per_action_source_picks.csv, recomputed).
- Page said `p_make`/`threshold_mass` agree with EV "only 58.9%" on the 280-decision sub-corpus; recomputed from Wave 1.1 action_utility_scalars.csv (seed 9430): p_make 74.6% argmax with row-order tie-break (91.1% with ties), threshold_mass_low 45.7% (58.9% with ties); cvar_10 82.5% and robust_q25 81.8% were confirmed exact and tie-free (evidence: w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker/action_utility_scalars.csv). The original 58.9% is exactly threshold_mass_low's tie-inclusive agreement — the original page misapplied that one lens's figure to both lenses. The threshold-mass-is-the-outlier conclusion holds under every tie convention (91.1 vs 58.9 with ties; 74.6 vs 45.7 row-order; 62.1 vs 36.8 unique-argmax-only).
- Page said "Twenty decisions have detector endorsing 5-5 when EV picks a different domino"; 20 is the count within the top-100 divisive set only — corpus-wide it is 915 decisions (evidence: divisive_decisions.csv detector_pick=='5-5' == 20; per_action_source_picks.csv == 915).

## Verified

- Headline agreement rates (78.3 / 43.7 / 47.0 / 39.5), divisiveness counts (18,378 / 8,529 / 1,082 / 11), per-family table, Gus proxy note, artifact list, and reproduce path all match summary.json / agreement_matrix.csv / per_claim_family_agreement.csv / run_cross_ai_agreement.py.
- Pattern 1 (reckless_count: 2,300 fires, mean regret 9.18) and pattern 2 (called_non_double non-EV 85.7% ≈ 86%) recomputed exactly; the regret-31.03 no-trump 4-way case matches divisive_decisions.csv row 1 (EV 6-0, Gus 1-1 regret 1.33, dist 5-1).

## Follow-ups

- The Wave 1.1 summary.json reports N=211 decisions (branch_atlas_scaled_v0 + v1, seeds 9420-9421 included) while the m2i7 join uses the 280-decision seed-9430 subset; a one-line note reconciling the two Ns would prevent future confusion.
- summary.json's `dist_lens_decisions: 754` does not match the page's 280/539 framing; worth checking what that field actually counts.
- A cheap next probe: recompute the four-source matrix on the 915 detector-endorses-5-5 decisions to see whether the 5-5 over-generalization is regime-specific or uniform.

## Review (second pass, 2026-07-07)

All four corrections independently re-derived from the primary artifacts; three stand exactly, one refined:

- Correction 1 (4-way splits) confirmed exact: divisive_decisions.csv rows with n_distinct_picks==4 give decl_name {twos:3, sixes:3, fives:2, doubles-suit:2, no-trump:1}, trick_idx 0–3, n_candidates 4–7. Original page's "predominantly no-trump/doubles-suit late-trick, 4–5 legal actions" was indeed wrong (evidence: w42/book_validation_v1/wave1/t42-m2i7_cross_ai_agreement/divisive_decisions.csv).
- Correction 2 (setter regime) confirmed exact: 35,133/35,133 setter action rows carry ch05_setter_pressure_regime (100%); 21,133 of those are non-EV-best (60.15%) (evidence: per_action_source_picks.csv, recomputed).
- Correction 3 (dist-lens rates) refined: 74.6/45.7 are argmax with row-order tie-break, not strict-unique argmax (which gives 62.1/36.8); p_make ties at max in 83/280 decisions, threshold_mass_low in 65/280, EV/cvar_10/robust_q25 are tie-free. Amended the page to state the tie convention and added threshold_mass_low's tie-inclusive 58.9% — which is precisely the original page's number, so the original error was misattributing tml's tie-inclusive figure to both lenses. The auditor's outlier conclusion survives all conventions (evidence: t42-ybo6_distribution_lens_reranker/action_utility_scalars.csv, recomputed).
- Correction 4 (5-5 counts) confirmed exact: 20 detector_pick=='5-5' rows in the top-100 divisive set (all with ev_pick != 5-5); 915 distinct decision keys corpus-wide with 5-5 detector-endorsed and not EV-best (evidence: divisive_decisions.csv + per_action_source_picks.csv).
- Verified section re-checked: headline rates (0.7834/0.4368/0.4702/0.3952, N 28,000/9,667/9,667/539), divisiveness counts (18,378/8,529/1,082/11), per-family table (per_claim_family_agreement.csv exact), reckless_count 2,300 fires @ mean regret 9.18, called_non_double 85.7% non-EV, the regret-31.03 no-trump row (EV 6-0, Gus 1-1 @ 1.327, dist 5-1), and the Gus proxy 64.5% (w42/joined_claim_row_model_table/model_metrics.csv, public_features_only 0.6446) all confirmed.
- Follow-up suggestions all kept: the N=211-vs-280 discrepancy is real (ybo6 summary.json N_decisions=211 across seeds 9420/9421/9430 while seed 9430 alone has 280 decision_ids in action_utility_scalars.csv), and dist_lens_decisions=754 in m2i7 summary.json genuinely differs from the page's 280/539 framing.
