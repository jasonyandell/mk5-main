# Follow-ups: w42-bookval-v2-utility-lens-synthesis

Reviewed against code on 2026-07-07 — no issues found.

Verified: all per-probe and per-claim table values match `w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/per_probe_utility_verdicts.csv` and `per_claim_utility_summary.csv`; probe-bead mapping and bootstrap params (n=2000, seed=42) match `manifest.json`; the superseded-by header's 41-44% divergence range and the ADOPT → ADOPT-DEFERRED downgrade match the Wave 4.0 page and campaign tracker; all five listed artifact paths exist in the repo.

## Follow-ups

- The page's "Missing utilities problem" flags ch04-low-trump-trap as a priority re-probe under p_make; the Wave 4.x campaign never ran it — still a cheap open probe.
- robust_q25 remains missing across all Wave 2 probes here, yet Wave 4.1 found it the second-best lens; backfilling q25 on the existing snapshots would close that gap cheaply.
