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

The audit parsed 13 CSV files and 15 JSON summaries, reran the seven phase-4
validation scripts, and checked that the 64-row ledger and 64-row completion
board align exactly.

## Links

[[w42]] | [[w42-phase4-claim-completion-board]] |
[[w42-phase4-sequence-handshape-tests]] |
[[w42-phase4-84-dynamic-seed-tests]] |
[[w42-phase4-doubles-notrump-regime-tests]] |
[[w42-phase4-laydown-rule-accounting]] |
[[w42-phase4-scoring-objective-tests]] |
[[w42-phase4-bidding-count-exposure-tests]]
