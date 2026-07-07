Reviewed against code on 2026-07-07 — no issues found.

The design landed essentially as written: `GameStateTensor.from_snapshot` (forge/eq/game_tensor.py:280), `generate_eq_from_snapshots` (forge/eq/generate/pipeline.py:335), `--snapshot-file` and `--bid-values` in forge/eq/generate/cli.py, `w42/book_validation_v1/wave2/run_bid_aware_atlas.py`, and the snapshot corpora under `w42/book_validation_v1/wave2/snapshots/`.

## Follow-ups

- Corpus naming drifted slightly from the design: `pounce_window_high_bid/` shipped as `pounce_window/`, and extra corpora exist (`reentry_preservation_v2/`, `void_creation_follow/`). Harmless for a design record, but a one-line "as-built" postscript would help future readers.
- Page is still `status: active` (last_updated 2026-05-03) while waves 3-5 directories now exist; consider flipping status to done/superseded and linking the wave-2 result pages.
