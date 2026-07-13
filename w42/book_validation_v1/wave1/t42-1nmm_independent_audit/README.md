# Independent Audit — Wave 1 / t42-1nmm

## Question

Is the existing 64-row phase-4 claim audit reproducible from artifacts? Where does an independent rebuild disagree with `w42/phase4_final_claim_audit/audit_summary.json`?

## Slice

All 64 claims in `w42/statistics_claims_ledger/claims.csv` and `w42/phase4_claim_completion_board/completion_board.csv`. All phase-4 worker artifacts cited in the prior audit.

## N

- 64 claims audited
- 28 cited artifact files checked for existence and row/key counts
- 7 validate_outputs.py scripts rerun
- 8 prior overclaim risks independently re-derived

## Method

Independent parse — not a rerun of `run_audit.py`. Steps:

1. Parse 64-row ledger and completion board, verify ID set symmetry and status alignment.
2. Check existence of all 28 files cited in `parsed_file_checks` of the prior audit.
3. Check all artifact paths cited in `evidence_artifacts` field of each claim row.
4. Compare each phase-4 worker `claim_statuses` or `claim_conclusions` against the ledger.
5. Rerun all 7 `validate_outputs.py` scripts.
6. Independently re-derive the 8 overclaim risks by inspecting artifacts directly.
7. Sanity-check the 12 context-limited blockers in `unresolved_or_blocked_claims.csv`.

## Status

Paired contrast. Evidence mode: artifact path verification + worker-vs-ledger comparison.

## Metric

- Artifact existence: 28/28 cited files present, 0 missing
- Ledger-board alignment: 0 status mismatches, 0 ID set differences
- Validation reruns: 7/7 pass
- Disagreements with prior audit: 3 rows (2 status, 1 taxonomy)
- Overclaim risks confirmed: 8/8

## Findings

**The prior audit holds up.** No drift in artifacts since 2026-05-03. Two status divergences found where phase-4 worker evidence justifies a status promotion that the ledger did not absorb:

1. **ch10-point-system-skill-signal**: ledger=`underpowered`, independent=`context-limited`. Phase-4 scoring worker ran 4 heuristic policy pairs and observed measurable point/mark separation. Small-N but not zero evidence.

2. **ch10-timed-marks-advancement-objective**: ledger=`not-yet-tested`, independent=`context-limited`. Phase-4 scoring worker ran 160 synthetic timed round-robin trials; advancement disagreement rate=0.15.

3. **ch10-tournament-speed-tradeoff**: No status change (both=`context-limited`). Worker used non-standard status string `supported-for-generated-trace-proxy` not in AGENTS.md taxonomy. Prior overclaim risk confirmed.

All 8 prior overclaim risks independently confirmed. All 12 context-limited blockers still hold.

## Caveats

- This wave does not propose ledger-status changes; documents findings for orchestrator reconciliation.
- Status divergences are based on phase-4 worker evidence produced on 2026-05-03; no new external data collected.
- `ch03-laydown-correctness`: 6 passing fixture proofs now exist; the not-yet-tested label may understate the fixture-substrate scope.

## Command

```
# Rebuild artifacts (do not rerun the prior audit script):
python3 -c "import json; ..."  # see manifest.json for exact inputs
```

## Artifacts

| File | Description |
|------|-------------|
| `independent_audit.json` | Full 64-row verdict with per-claim consistency checks |
| `audit_diff.csv` | 3 rows of disagreement with prior audit |
| `validation_rerun_results.csv` | 7 phase-4 scripts, all pass |
| `overclaim_risk_diff.csv` | 8 prior overclaims re-derived independently |
| `summary.json` | Headline numbers |
| `manifest.json` | Input SHA fingerprints |
