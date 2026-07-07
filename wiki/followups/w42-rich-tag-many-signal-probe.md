Reviewed against code on 2026-07-07 — no issues found.

Verified: all artifact paths under `w42/rich_tag_many_signal_probe/` exist; headline metrics (rich best/final 1.970 regret, 65.54% match, 75.54% near-tie, 11.43% tail; v0 best 2.000; raw final 2.834/17.32% tail; E[Q] N=10 0.118/90%) match `metrics.json`; all nine concept-bucket rows match `bucket_metrics_best.csv` (`rich:*` buckets); script defines 48 global + 48 action rich signal names; W&B run id `3xyy2dxr` and command args match `run.json`/`manifest.json`; commit `f746b93` exists; source corpora present at `/Users/jason/code/mk5-main/gus/data/`.

- W&B run `3xyy2dxr` (and superseded `c5x85xfx`) not verified remotely.
- Cheap next probe: the page's own multi-seed suggestion is the right one — the +0.03 regret delta over v0 is within single-seed noise.
- The `action:identity`/`slot`/`pip_pressure`/`hand_shape` buckets in `bucket_metrics_best.csv` are identical rows (n=381) — the bucket assignment collapses several action tags to the same mask; worth a note if bucket slices are ever used as evidence.
