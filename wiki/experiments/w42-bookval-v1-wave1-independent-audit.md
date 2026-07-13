---
kind: experiment
title: W42 Book Validation v1 — Wave 1 Independent Audit
bead: t42-1nmm
epic: t42-4zi6
wave: wave1
status: complete
created: 2026-05-03
first_seen: 2026-05-03
last_updated: 2026-07-11
---

## Summary

Wave 1 of the W42 book validation campaign independently rebuilt the 64-row phase-4 claim audit to verify whether the baseline established by [[w42-phase4-final-claim-audit]] is reproducible and trustworthy before downstream waves propose claim-status changes.

**Headline:** The prior audit holds up. All cited artifacts exist, all validation scripts pass, and the board/ledger are internally consistent. Two status divergences were found where phase-4 worker evidence justifies promotions not yet absorbed by the ledger. All 8 prior overclaim risks are independently confirmed.

## Inputs

- [[w42-phase4-claim-completion-board]] — 64-row claim ledger surface
- [[w42-phase2-statistics-claims-ledger]] — master 64-row ledger
- [[w42-phase4-final-claim-audit]] — prior audit verdict and overclaim list
- All `w42/phase4_*/summary.json` and worker artifacts cited in the prior audit

## Method

Independent parse — not a rerun of `run_audit.py`. Steps taken:

1. Verified all 28 files cited in `parsed_file_checks` exist on disk with matching row/key counts.
2. Verified all artifact paths in `evidence_artifacts` for every claim row exist on disk (0 missing).
3. Compared board-ledger ID sets (0 differences) and status field alignment (0 mismatches).
4. Compared each phase-4 worker `claim_statuses` or `claim_conclusions` against the ledger.
5. Reran all 7 `validate_outputs.py` scripts (7/7 pass, exit code 0).
6. Independently re-derived the 8 overclaim risks by inspecting worker artifacts directly.
7. Sanity-checked all 12 context-limited blockers in `unresolved_or_blocked_claims.csv`.

## Findings

### Artifact Baseline

- 28/28 cited files exist. 0 missing.
- 64/64 ledger artifact paths exist. 0 broken.
- Board/ledger: 0 ID set differences, 0 status mismatches.
- Baseline counts match: {supported: 23, underpowered: 21, context-limited: 12, not-yet-tested: 6, contradicted: 2}.

### Validation Scripts

All 7 phase-4 `validate_outputs.py` scripts rerun and pass:

| Script | Result |
|--------|--------|
| phase4_84_dynamic_seed_tests | PASS |
| phase4_bidding_count_exposure_tests | PASS |
| phase4_doubles_notrump_regime_tests | PASS |
| phase4_laydown_rule_accounting | PASS |
| phase4_scoring_objective_tests | PASS |
| phase4_sequence_handshape_tests | PASS |
| phase4_claim_completion_board | PASS |

### Status Divergences (3 rows)

Two rows where independent evidence supports a status promotion not in the ledger:

**ch10-point-system-skill-signal** (ledger=`underpowered`, independent=`context-limited`): The phase-4 scoring worker (t42-br7n.3) ran 4 heuristic policy pairs over 4000 generated hands and observed different point/mark separation under the two objectives. This is small-N policy-population evidence — not oracle or human — but it crosses the underpowered threshold. Advancement disagreement rate=0.15 in the timed pool.

**ch10-timed-marks-advancement-objective** (ledger=`not-yet-tested`, independent=`context-limited`): The phase-4 scoring worker ran 160 synthetic timed round-robin trials with a trick-budget model. The advancement disagreement rate of 0.15 between point-leader and mark-leader goes beyond not-yet-tested. Bounded: synthetic round-robin, no real bracket or clock.

**ch10-tournament-speed-tradeoff** (ledger=`context-limited`, independent=`context-limited`): No status change. The phase-4 scoring worker used a non-standard status string `supported-for-generated-trace-proxy` not in the AGENTS.md taxonomy. The underlying evidence (23.8 trick reduction per match, marks vs points) is real but the ledger `context-limited` is the correct taxonomy label. Prior overclaim risk confirmed.

### Overclaim Risk Assessment (8/8 Confirmed)

All 8 prior suspect overclaims independently confirmed by artifact inspection:

| Claim | Risk Confirmed |
|-------|---------------|
| ch02-bid-only-enough | Yes — synthetic auction Monte Carlo, not full policy |
| ch07-protected-one-off-shape-frequency | Yes — frequency only, no make/set rollout |
| ch10-tournament-speed-tradeoff | Yes — tricks proxy, no wall-clock data |
| ch09-doubles-trump-candidate-4plus | Yes — same-hand simulation shows NT parity in most 4+ buckets |
| ch03-laydown-correctness | Partially — 6 fixture proofs exist but broad corpus not wired |
| ch04-safe-partner-count-donation | Yes — before-closure CI crosses zero; only closure slice is strong |
| ch05-pounce-count-before-certainty | Yes — phase-4 proxy CI crosses zero; Gus corpus is primary support |
| ch02-strong-trump-bad-risk-trap | Yes — trap make rate=0.71 (high); risk is bid-ceiling, not make-rate |

**Additional note on ch03-laydown-correctness:** Six passing fixture proofs now exist (phase4_laydown_rule_accounting). The ledger `not-yet-tested` understates the fixture-substrate scope. This is a genuine candidate for a bounded status note or split.

### Context-Limited Blocker Sanity Check

All 12 context-limited blockers in `unresolved_or_blocked_claims.csv` remain valid. No new evidence in the current artifact set eliminates any of the stated requirements (full policy counterfactuals, arbitrary late-state injection, wall-clock tournament data).

## Disagreements Summary

See `audit_diff.csv` (3 rows) for per-disagreement evidence and recommended actions.

## Caveats

- This wave documents evidence for orchestrator reconciliation. No ledger edits are proposed.
- The two status divergences are based on phase-4 worker evidence produced on 2026-05-03 in the same session as the prior audit; no new external data was collected.
- The `supported-for-generated-trace-proxy` status string in the scoring objective worker is a taxonomy violation that should be cleaned up.

## Outputs

All outputs under `w42/book_validation_v1/wave1/t42-1nmm_independent_audit/`:

- `independent_audit.json` — full 64-row verdict with per-claim consistency checks
- `audit_diff.csv` — 3 disagreement rows
- `validation_rerun_results.csv` — 7/7 pass results
- `overclaim_risk_diff.csv` — 8 overclaim re-derivations
- `summary.json` — headline numbers
- `manifest.json` — input SHA fingerprints

## Related Pages

- [[w42-phase4-final-claim-audit]]
- [[w42-phase4-claim-completion-board]]
- [[w42-phase2-statistics-claims-ledger]]
- [[w42-book-claim-synthesis-and-ai-directions]]
