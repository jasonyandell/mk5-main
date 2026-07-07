Reviewed against code on 2026-07-07 — no issues found.

All headline numbers, N breakdown, slice table, claim-ledger impact, artifact paths, and the snapshots.jsonl SHA256 verified against `w42/book_validation_v1/wave2/probes/t42-ntbe_pounce_window_bid30/summary.json` and `paired_contrasts.csv` (500 data rows).

- Finding 3's "high-pip count tiles (5-5 or 6-4)" inspection detail was not independently re-derived from the CSV (the CSV's count_bucket column doesn't name tiles); harmless narrative, but a one-line pointer to the 5 snap_idx values would make it reproducible.
- A cheap next probe: rerun the paired contrast with p_make (threshold-probability) deltas instead of the p_set proxy, since the p_make-vs-E[Q] divergence is the page's central mechanism but is inferred, not measured.
- Subgroup slices with N=5 (count=10pts, phase late) drive two bolded findings; pooling with a second 500-snapshot draw would confirm or kill the 10-pt "pounce burns position" effect cheaply.
