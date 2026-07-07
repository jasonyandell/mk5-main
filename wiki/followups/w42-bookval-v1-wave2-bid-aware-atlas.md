# Followups: w42-bookval-v1-wave2-bid-aware-atlas

Reviewed against code and artifacts on 2026-07-07. Full-sweep numbers all verified: divergence table, per-bid row counts (259,618 total), wall time (1939.9 s), power-analysis CIs, and bid=30 validation 10/10 all match `manifest.json` / `power_analysis.csv` / `validation_check.csv` exactly.

## Corrections

- Page said `mark_multiplier = bid // 42 for bid >= 84`; code applies `bid // 42` for bid >= 42 (yielding 1 at bid=42, 2 at 84) (evidence: w42/book_validation_v1/wave2/run_bid_aware_atlas.py, `mark_multiplier`). Same numeric results, wrong stated bound.
- Artifacts table said `bid_aware_actions.csv` was "last run: seed 9430, 7 bids, 5550 rows"; the on-disk CSV is the full-sweep corpus, seeds 9000–9049, 259,618 rows (seed 9430 rows are not in the CSV per the manifest `rejoin_note`) (evidence: w42/book_validation_v1/wave2/bid_aware_atlas/manifest.json, `wc -l bid_aware_actions.csv` = 259,619).
- Artifacts table listed `eq_pdf_seeds9000-9004_bid*.pt` / `eq_pdf_seeds9430-9430_bid*.pt` as if present; no .pt files exist in the repo, and the full sweep used batched filenames `eq_pdf_seeds9000-9009_bid*.pt` etc. — replaced with a note that filenames + SHA256s live in the manifest join logs.

## Not verifiable

- Smoke-run tables (first divergence table, 63.2%/57.5% change rates, n=280) — smoke artifacts were overwritten by the full sweep; left as-is.
- `w42/book_validation_v1/wave2/bid_aware_atlas/README.md` carries the same stale "5550 rows" description (not edited — out of scope for this audit).

## Follow-ups

- Cheap fix: update the artifact README.md's row-count line to match the full-sweep corpus.
- The .pt joint-world files exist only locally (if at all); if the corpus matters for reproduction, consider archiving them or noting explicitly where they live.
