# Audit: w42-bookval-v1-wave1-hidden-threat-impact-ranker

Reviewed against code and artifacts on 2026-07-07.

## Corrections

- Page said trump-count tiles appear as load-bearing in fours in 8 decisions; artifact says 12 (blanks 6 and sixes 2 were correct) (evidence: w42/book_validation_v1/wave1/t42-c2y9_hidden_threat_impact_ranker/per_decision_top_k_tiles.csv, rank=1 rows with tile_category=trump_count).

## Follow-ups

- The "Highest-impact single-decision tile occurrences" list (mirrored from the artifact README) is not a strict impact ranking: the actual #2 is 4-4 held by bidder_partner in a fours declaration at 52.3 (a trump_double), which both README and page omit; the 4-4/no-trump entries are ranks 3 and 5 (47.5, 49.1). Not corrected since the page faithfully mirrors the artifact README; a cheap fix would be to regenerate the list from inferable_vs_not.csv sorted by max_impact_score.
- summary.json reports detector_correlation_rows: 24 while the shipped detector_correlation.csv has 113 data rows (page is right); worth noting the summary.json field is stale if anyone consumes it programmatically.
- The page's enrichment table "Rate in Top-5" column is a renormalized version of summary.json's rate_in_topk (consistent scaling; enrichment ratios identical) — fine, but a one-line denominator note on the page would prevent future confusion.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Independently recounted rank=1 trump_count rows in per_decision_top_k_tiles.csv: fours 12, blanks 6, sixes 2 (total 20, consistent with summary.json tile_category_distribution_top1 trump_count 0.0617 × 324). The auditor's 8→12 fix is correct and the single-line edit damaged nothing else.
- Follow-up bullets independently re-derived and confirmed: (1) inferable_vs_not.csv sorted by max_impact_score gives #2 = 4-4/bidder_partner/fours/trump_double at 52.3, omitted by both README and page; (2) summary.json N.detector_correlation_rows = 24 vs 113 data rows in detector_correlation.csv; (3) page's rate columns are summary.json values scaled by a constant ~1.895, enrichment ratio 1.325 ≈ page's 1.33x.
