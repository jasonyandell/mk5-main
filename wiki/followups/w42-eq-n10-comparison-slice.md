Reviewed against code on 2026-07-07 — no issues found.

- All boss-table numbers match `w42/eq_n10_comparison_slice/summary.json` and `w42/raw_public_state_baseline/metrics.json` exactly (0.117969 / 90.00% / 95.00% / n=560, raw rows 2.471069 and 2.833552, deltas included).
- Mechanism description confirmed: `EQNWrapper` takes first N per-world Q rows; `evaluate_eq_n` means them, masks illegal actions, argmaxes, and scores regret vs full `e_q` (w42/raw_public_state_baseline.py:105, :229).
- Both commit hashes (`8df0c3b1`, `3a5575fc`) exist in the repo; cross-referenced `gus-strategy-tags-probe` figures (0.191, 0.167) also match that page.
- Cheap next probe: multiple random world subsamples for E[Q] N=10 to bound boss variance, as the page's Next Checks already suggests — still unrun.
