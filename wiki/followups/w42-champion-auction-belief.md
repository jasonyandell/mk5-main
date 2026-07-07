Reviewed against code on 2026-07-07 — no issues found.

- Headline numbers (+2.59pp mean, per-corpus deltas, 11/11 positive, shuffled −0.39pp vs real +2.63pp, arena −0.29 CI [−0.92, +0.36]) all match `scratch/champion-run/run24_RESULTS.txt` (main checkout; gitignored, as the page notes).
- Code claims verified: `gus/model/auction.py::auction_feature_vector` is the 28-dim per-relative-seat + 10-wide decl one-hot exactly as described; `BidsEncoder`/`StudentTransformerFullVoidsAuction` in `gus/model/student.py`; `--auction`/`--shuffle-bids` flags and `--emit-snapshots` all present; all four commits exist on `forge`.
- Cheap next probe: rerun the arena A/B with a non-oracle (net:wp or jud) play policy — the page's own thesis predicts the belief gain should only show marks value when play is imperfect; that would turn the interpretation from inference into measurement.
