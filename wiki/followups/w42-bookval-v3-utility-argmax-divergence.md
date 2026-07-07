Reviewed against code on 2026-07-07 — no issues found.

- Disagreement table, void/preserve confusion table, headline 41.2% [36.8%, 45.6%], 29/500 = 5.8% joint pattern, runtime 207.9s, sha256 prefix, and gate verdict text all match `w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/{summary.json,disagreement_matrix.csv,void_subset_confusion.csv,manifest.json}`; utility formulas match `analyze.py`.
- Cheap next probe: rerun on a higher-bid subset (bid ≥ 84, mm ≥ 2) to actually exercise the mark_ev ≠ p_make regime the caveat says this corpus cannot test.
- Per-snapshot resampling variance (caveat 3) is checkable for ~$0: rerun 50 snapshots with a different world seed and count argmax flips.
