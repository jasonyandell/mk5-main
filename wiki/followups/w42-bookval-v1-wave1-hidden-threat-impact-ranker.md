# Audit: w42-bookval-v1-wave1-hidden-threat-impact-ranker

Reviewed against code and artifacts on 2026-07-07.

## Corrections

- Page said trump-count tiles appear as load-bearing in fours in 8 decisions; artifact says 12 (blanks 6 and sixes 2 were correct) (evidence: w42/book_validation_v1/wave1/t42-c2y9_hidden_threat_impact_ranker/per_decision_top_k_tiles.csv, rank=1 rows with tile_category=trump_count).

## Follow-ups

- The "Highest-impact single-decision tile occurrences" list (mirrored from the artifact README) is not a strict impact ranking: the actual #2 is 4-4 held by bidder_partner in a fours declaration at 52.3 (a trump_double), which both README and page omit; the 4-4/no-trump entries are ranks 3 and 5 (47.5, 49.1). Not corrected since the page faithfully mirrors the artifact README; a cheap fix would be to regenerate the list from inferable_vs_not.csv sorted by max_impact_score.
- summary.json reports detector_correlation_rows: 24 while the shipped detector_correlation.csv has 113 data rows (page is right); worth noting the summary.json field is stale if anyone consumes it programmatically.
- The page's enrichment table "Rate in Top-5" column is a renormalized version of summary.json's rate_in_topk (consistent scaling; enrichment ratios identical) — fine, but a one-line denominator note on the page would prevent future confusion.
