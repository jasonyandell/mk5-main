Reviewed against code on 2026-07-07 — no issues found.

All headline and per-slice numbers match `w42/book_validation_v1/wave2/probes/t42-8kbh_pounce_high_bid/summary.json` and `slice_breakdown.csv`; Wave 2.E comparison figures match `probes/t42-ntbe_pounce_window_bid30/summary.json`; the script implements the pounce/decline paired contrast as described (lowest-cost tile selection, n_samples=100/arm).

- The atlas `.pt` files (`bid_aware_atlas/eq_pdf_seeds*_bid*_v2.pt`) are not in the repo — only manifest/CSVs — so the "n_samples=200 per decision" claim was verified via manifest, not the tensors themselves.
- The page flags the "setter team led" subsample (39.6% pounce better, N=280) as worth a narrower probe; that follow-up appears not to have run — a cheap next probe would filter to setter-partner-led tricks with count currently on the table.
- Caveat 1 (compute per-position mark_ev to cross-check E[Q]) remains open and would resolve the tension with Wave 2.B.2 directly.
