# w42-claim-ledger — audit 2026-07-07

## Corrections

- Page said the ledger "was never centrally assembled into one populated ledger file"; a populated 64-row central ledger exists at `w42/statistics_claims_ledger/claims.csv` (with `manifest.json`/`summary.json`, generator `build_statistics_claims_ledger.py`, git sha `e55a6f8`, exact same 5-value status vocabulary), which [[w42-phase2-statistics-claims-ledger]] assembled and phase 4 carried to closure as canonical (evidence: w42/statistics_claims_ledger/, wiki/experiments/w42-phase2-statistics-claims-ledger.md).

Verified clean: schema/template paths exist and match the page's field/vocabulary description exactly; six per-bead delta/update files exist (5 `claim_ledger_delta.json` + 1 `claim_ledger_update.json`); commit `489c1fd7` exists; the jud v1 "registered predictions" quote matches wiki/experiments/w42-jud-v1.md line 28.

## Follow-ups

- claims.csv uses a slightly different column set (family, evidence_mode, measurement, result, next_check) than the schema's entry fields; a one-line note on the field mapping between the JSON schema and the CSV ledger would prevent confusion.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Re-derived from primary evidence: `w42/statistics_claims_ledger/claims.csv` has 64 data rows; `w42/statistics_claims_ledger/manifest.json` confirms generator `build_statistics_claims_ledger.py`, git sha `e55a6f89de...` (short `e55a6f8`), claim_count 64, and the identical 5-value status vocabulary; `w42/claim_ledger.schema.json` status enum matches the page table; the six per-bead files are 5 `claim_ledger_delta.json` + 1 `claim_ledger_update.json`; commit `489c1fd7` exists; the jud v1 quote matches `wiki/experiments/w42-jud-v1.md`; and both `wiki/experiments/w42-phase2-statistics-claims-ledger.md` and `wiki/experiments/w42-phase4-final-claim-audit.md` (plus `w42/phase4_final_claim_audit/audit_summary.json` finding 64 ledger rows) support "phase 2 assembled, phase 4 carried to closure as canonical." The follow-up suggestion (schema↔CSV field mapping note) is sensible and not yet done; kept.
