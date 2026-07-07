## Corrections

- Page said the audit "reran the seven phase-4 validation scripts"; the audit's artifact does not record any rerun — its `validation` block contains only file-parse counts and ledger/board alignment checks, so the page now states the rerun is unrecorded rather than asserting it happened or didn't. Note the artifact's `scope` field ("Audit only; ... Write scope limited to w42/phase4_final_claim_audit/") constrains *write* scope only and the per-worker `validate_outputs.py` scripts are read-only, so scope alone cannot rule a rerun in or out. The later independent audit did rerun all seven validators, 7/7 pass. (evidence: w42/phase4_final_claim_audit/audit_summary.json; w42/phase4_laydown_rule_accounting/validate_outputs.py; w42/book_validation_v1/wave1/t42-1nmm_independent_audit/README.md)

## Verified

- 64 ledger rows / 64 unique claim ids / 64 completion-board rows, 0 no-evidence-no-blocker claims, 39 unresolved rows (37 bounded_by_generator_or_scope + 2 ledger_review_candidate_bounded: ch02-bid-only-enough, ch07-protected-one-off-shape-frequency), 13 CSVs + 15 JSONs parsed, and all 8 overclaim guards match `suspect_overclaims` (evidence: w42/phase4_final_claim_audit/audit_summary.json, unresolved_or_blocked_claims.csv).

## Follow-ups

- Of the two ledger-review candidates, ch02-bid-only-enough was already reconciled — promoted to `supported` in wave 2.G (commit 6918aa94, w42/statistics_claims_ledger/claims.csv). Only ch07-protected-one-off-shape-frequency remains at `underpowered` with an open review recommendation; a cheap next step is a one-pass status review for that single row.
- The audit script itself (`run_audit.py`, named in w42/book_validation_v1/wave1/t42-1nmm_independent_audit/README.md) is not checked into the repo — only its outputs are; if reproducibility matters, capture the script alongside the artifacts. (Mitigated but not mooted by t42-1nmm's independent-parse reproduction.)

## Review (second pass, 2026-07-07)

- Amended the page's Validation paragraph: the first pass correctly removed the unsupported "reran the seven phase-4 validation scripts" claim (the artifact's `validation` block records only parse counts and the ledger/board join), but its replacement over-asserted the negative — it claimed the audit "parsed ... rather than rerunning the validation scripts" citing the `scope` field, which constrains write scope only, and `validate_outputs.py` is read-only, so a rerun cannot be ruled out from the artifact. Page now says the rerun is unrecorded. Evidence: w42/phase4_final_claim_audit/audit_summary.json (validation + scope fields), w42/phase4_laydown_rule_accounting/validate_outputs.py.
- Added to the page: the later independent audit reran all seven validators (7/7 pass) and confirmed the counts and all 8 overclaim risks. Evidence: w42/book_validation_v1/wave1/t42-1nmm_independent_audit/README.md.
- Re-derived the Verified section independently: 64/64/64 rows, 0 no-evidence-no-blocker, 39 unresolved (37 bounded_by_generator_or_scope + 2 ledger_review_candidate_bounded), 13 CSVs + 15 JSONs, 8 page overclaim-guard bullets map 1:1 to the 8 `suspect_overclaims` entries. All confirmed. Evidence: w42/phase4_final_claim_audit/audit_summary.json, w42/phase4_final_claim_audit/unresolved_or_blocked_claims.csv.
- Amended follow-up #1: ch02-bid-only-enough was already reconciled (context-limited → supported, wave 2.G, commit 6918aa94); only ch07-protected-one-off-shape-frequency remains open (still `underpowered` in w42/statistics_claims_ledger/claims.csv).
- Verified follow-up #2 stands: no `run_audit.py` anywhere in the repo (find), and t42-1nmm's README names it as the prior audit's script, confirming only outputs were archived.
