Reviewed against code on 2026-07-07 — no issues found.

All headline numbers verified against `w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker/summary.json` (64.9% = 137/211, per-utility disagree rates, EV gaps, n_alt_dominates_ev=0), detector rates against `detector_explained_disagreements.csv`, top-5 decisions against `top_disagreements.csv`, SHAs against `manifest.json`, and mechanism (Q>=18 make cut, CVaR_10 = mean of lowest 10%, robust_q25 = 0.25 quantile) against `run_distribution_lens_reranker.py`.

- The per-detector table on the page is a curated subset (10 of 31 tags); several high-n tags (hidden_threat_large_impact n=179, wide_shelf_gap n=158) are omitted — a footnote noting the subset would prevent misreading it as exhaustive.
- Cheap next probe: rerun on a multi-mark bid (84) corpus, where mark_ev decouples from p_make and the degenerate 4-way tie in the disagreement matrix breaks.
- `n_alt_dominates_ev = 0` is within-sample only; a multi-seed replication (even 2-3 more base seeds) would test whether EV's small point-advantage over tail lenses generalizes.
