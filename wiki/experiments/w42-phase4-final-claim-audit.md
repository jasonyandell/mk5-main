---
title: w42 Phase4 Final Claim Audit
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] ran an independent final audit after the `t42-br7n` phase-4 worker
artifacts landed. The audit writes
`w42/phase4_final_claim_audit/audit_summary.json` and
`w42/phase4_final_claim_audit/unresolved_or_blocked_claims.csv`.

The verdict is that phase 4 can close conservatively. The audit found 64 ledger
rows, 64 unique claim ids, 64 completion-board rows, and no claim with both no
evidence and no blocker. The remaining 39 rows are bounded, not ownerless: 37
have explicit generator or scope bounds, and 2 are ledger-review candidates
(`ch02-bid-only-enough` and `ch07-protected-one-off-shape-frequency`).

This does not mean every book claim is broadly true. It means every claim is now
either evidenced on a named slice, contradicted on a named slice, bounded by an
explicit generator/data requirement, or ready for ledger-reconciliation review.

## Overclaim Guards

The audit calls out these tempting overclaims to avoid:

- do not promote broad `bid-only-enough` beyond generated auction-pressure and
  bid-margin evidence;
- do not treat 84 shape frequency as action-value proof;
- do not treat generated trick savings as human wall-clock tournament evidence;
- do not simplify Chapter 9 to "four-plus doubles means doubles-trump";
- do not apply laydown correctness beyond the fixture proof checker until saved
  snapshots and Burl claims are wired in;
- do not broaden partner count donation beyond exact closure/current-control
  slices;
- do not broaden pounce-before-certainty beyond the operationalized row-local
  labels;
- do not turn the strong-trump trap into a simple make-rate penalty.

## Validation

The audit parsed 13 CSV files and 15 JSON summaries and checked that the
64-row ledger and 64-row completion board align exactly (no rows on either
side of the join). Whether it also reran the seven per-worker
`validate_outputs.py` scripts is not recorded in its artifact —
`audit_summary.json`'s validation block documents only file-parse counts and
the ledger/board join. The later independent audit
(`w42/book_validation_v1/wave1/t42-1nmm_independent_audit/`) reran all seven
validators (7/7 pass) and confirmed the audit's counts and all eight
overclaim risks.

## Links

[[w42]] | [[w42-phase4-claim-completion-board]] |
[[w42-phase4-sequence-handshape-tests]] |
[[w42-phase4-84-dynamic-seed-tests]] |
[[w42-phase4-doubles-notrump-regime-tests]] |
[[w42-phase4-laydown-rule-accounting]] |
[[w42-phase4-scoring-objective-tests]] |
[[w42-phase4-bidding-count-exposure-tests]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 1 correction applied in place and independently re-verified; second pass amended 1.

- Of the two ledger-review candidates, `ch02-bid-only-enough` was later reconciled — promoted to `supported` in wave 2.G (commit 6918aa94, `w42/statistics_claims_ledger/claims.csv`). Only `ch07-protected-one-off-shape-frequency` remains `underpowered` with an open review recommendation; a cheap next step is a one-pass status review for that single row.
- The audit's script (`run_audit.py`, named in `w42/book_validation_v1/wave1/t42-1nmm_independent_audit/README.md`) is not checked into the repo — only its outputs are; if reproducibility matters, capture the script alongside the artifacts. Mitigated but not mooted by t42-1nmm's independent-parse reproduction.
