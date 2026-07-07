# Followups: w42-bookval-v1-wave2-bid-aware-atlas

Reviewed against code and artifacts on 2026-07-07. Full-sweep numbers all verified: divergence table, per-bid row counts (259,618 total), wall time (1939.9 s), power-analysis CIs, and bid=30 validation 10/10 all match `manifest.json` / `power_analysis.csv` / `validation_check.csv` exactly.

## Corrections

- ~~Page said `mark_multiplier = bid // 42 for bid >= 84`; code applies `bid // 42` for bid >= 42~~ **Retracted by second pass.** The code (w42/book_validation_v1/wave2/run_bid_aware_atlas.py:69-72) is literally `if bid >= 84: return bid // 42; return 1` — the original page's ≥ 84 bound was correct, and the first-pass edit misdescribed the code (numerically equivalent, but the wrong condition). The page now reads `1 for bid < 84 (including bid=42); bid // 42 for bid ≥ 84`, matching the code exactly while closing the original page's unstated 42–83 gap.
- Artifacts table said `bid_aware_actions.csv` was "last run: seed 9430, 7 bids, 5550 rows"; the on-disk CSV is the full-sweep corpus, seeds 9000–9049, 259,618 rows (seed 9430 rows are not in the CSV per the manifest `rejoin_note`) (evidence: w42/book_validation_v1/wave2/bid_aware_atlas/manifest.json, `wc -l bid_aware_actions.csv` = 259,619).
- Artifacts table listed `eq_pdf_seeds9000-9004_bid*.pt` / `eq_pdf_seeds9430-9430_bid*.pt` as if present; no .pt files exist in the repo, and the full sweep used batched filenames `eq_pdf_seeds9000-9009_bid*.pt` etc. — replaced with a note that filenames + SHA256s live in the manifest join logs.

## Not verifiable

- Smoke-run tables (first divergence table, 63.2%/57.5% change rates, n=280) — smoke artifacts were overwritten by the full sweep; left as-is.
- `w42/book_validation_v1/wave2/bid_aware_atlas/README.md` carries the same stale "5550 rows" description (not edited — out of scope for this audit).

## Follow-ups

- Cheap fix: update the artifact README.md's row-count line to match the full-sweep corpus.
- The .pt joint-world files exist only locally (if at all); if the corpus matters for reproduction, consider archiving them or noting explicitly where they live.

## Review (second pass, 2026-07-07)

- **Amended correction 1 (mark_multiplier):** the first pass changed a correct page statement into a false one. `run_bid_aware_atlas.py:69-72` reads `if bid >= 84: return bid // 42` / else `return 1` — the original page's "bid // 42 for bid ≥ 84" matched the code; the auditor's "for bid ≥ 42" did not (numerically equivalent for all bids, but the wrong code condition). Page line now states `1 for bid < 84 (including bid=42); bid // 42 for bid ≥ 84`, and the correction bullet above is retracted accordingly. Evidence: w42/book_validation_v1/wave2/run_bid_aware_atlas.py:69-72 (and `run_full_sweep_batched.py` imports the same function).
- **Correction 2 verified:** on-disk `bid_aware_actions.csv` is 259,619 lines (259,618 rows + header), seeds run 9000–9049 (checked via `awk` on column 1), and `manifest.json` `rejoin_note` confirms seed 9430 rows are not in the CSV.
- **Correction 3 verified:** no `.pt` files exist under `w42/book_validation_v1/wave2/bid_aware_atlas/` (dir contains only 5 files); `manifest.json` `join_log`/`join_with_validation` record 42 batched filenames (`eq_pdf_seeds9000-9009_bid30_v2.pt` … `eq_pdf_seeds9430-9430_bid*.pt`) with SHA256s, matching the page's replacement text.
- **Follow-up suggestions verified:** the README stale-row-count fix is real (`bid_aware_atlas/README.md:99` still says "seed 9430, 7 bids, 5550 rows"); both suggestions kept.
