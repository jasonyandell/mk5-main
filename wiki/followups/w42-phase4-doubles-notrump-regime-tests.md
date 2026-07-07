Reviewed against code on 2026-07-07 — no issues found. All bucket numbers, coverage counts, and CI-crosses-zero claims match `w42/phase4_doubles_notrump_regime_tests/summary.json` and `bucket_summary.csv` exactly.

- Could surface the dual_suit_65 bucket (n=44, +0.148, 75.0% NT-preferred) — it's in summary.json but absent from the page's bucket table.
- A cheap next probe: rerun with non-greedy (sampled) play to check whether the flat high-double-control result is policy-greediness-sensitive.
