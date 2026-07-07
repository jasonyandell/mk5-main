Reviewed against code on 2026-07-07 — no issues found.

- `burl/eval/belief_calibration.py` exists and implements exactly the hidden-vs-played separation the page describes (hidden_mask excludes own hand + already-played; hidden-only top-1 computed per state).
- Headline numbers (72% all-28 vs ~39% hidden-only, Brier 0.224, ECE 0.067) match the source digest `wiki/sources/d9baf3b.md`; no raw JSON calibration output is checked into `burl/eval/results/` (only perf artifacts), so the exact figures are cited from the digest, not re-derivable in-repo.
- Cheap follow-up: check in the calibration run's numeric output (or a one-line CSV) alongside the plots so the 39%/0.224/0.067 figures are reproducible from the repo.
