# w42-phase2-decision-table — audit 2026-07-07

## Corrections

- Page said the v0 table "reads `forge/analysis/results/data/eq_pdf_v3_sample.jsonl`" as if present; that file is untracked and no longer exists in the repo (evidence: `git log` shows no history for the path; sha256 preserved in `w42/phase2_decision_table/manifest.json`). Rephrased to past tense with a provenance note.

All headline numbers verified clean against `w42/phase2_decision_table/summary.json`: 140 states / 346 actions, 68 scalar-EV omission, 118 multi-peak decisions, 87/309/259/177 action tags, 133/128/114 actual-action agreement, all detector counts, 1000 samples/PDF, 12 examples. CSV row counts confirmed on disk. Script CLI (`--input`, default path) matches the Commands section.

## Follow-ups

- The build is unreproducible without the source JSONL; a cheap fix is to check the 5-game sample (or a regeneration command) into the repo, or note where it can be regenerated from.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Independently confirmed: `forge/analysis/results/data/eq_pdf_v3_sample.jsonl` absent with zero git history (`git log --all` on the path is empty); sha256 present in `w42/phase2_decision_table/manifest.json`; all page numbers match `w42/phase2_decision_table/summary.json` (140/346, 68, 118, 87/309/259/177, 133/128/114, all nine detector counts, 1000 samples/PDF); CSVs have 346 and 140 data rows on disk; `examples.json` has 12 entries; script default input and `--input` flag match the Commands section (`build_phase2_decision_table.py` lines 44, 884). Edit touched only the Data Slice paragraph; nothing nearby damaged. Follow-up suggestion kept — the source sample is still missing, so reproducibility remains open.
