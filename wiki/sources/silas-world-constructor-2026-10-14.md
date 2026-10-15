---
title: Silas world constructor — source bundle and native integration
kind: source
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Source

Jason supplied [silas_world_constructor.zip](https://drive.google.com/file/d/1m4PG44GmlIrw1imUixb2DbP6kTGM3Z9c/view),
then requested native integration in an independent worktree and a rerun of the
lessons and record checks. The bundle contains `construct_42.py`, `README.md` and
`results.json`, reviewed against repository snapshot `76254ede`. Its standard-library
Python prototype counts labelled allocations, partitions them by resolved ownership,
and samples a uniform integer rank over the accepted completion counts. The local
rerun reproduced its JSON exactly before integration.
Bundle SHA-256: `32c6194288b869357f4241b78b42530d393af3ea7629e90ba86fa603dc5db49a`.

## Native artifacts

| path | role |
|---|---|
| `silas/src/physics.rs` | shared three/four-seat allocation DP; public checked `PreparedFiber` |
| `silas/src/construct.rs` | `WorldSpace`, `OpeningSpace`, disjoint cells and uniform integer draws |
| `silas/src/frame.rs` | checked ownership reads including the varying opening viewer |
| `silas/src/bin/silas.rs` | `construct` command and replayable JSON receipts |
| `silas/examples/construction/four-fives.silas` | fresh-hand family in the existing language |
| `silas/tests/construct.rs` | independent bijections, source counts, refusals and record replay |
| `silas/tests/book_odds.rs` | constructor parity with uniform-fiber lesson probabilities |
| `silas/tests/cli.rs` | deterministic receipts, frame reconstruction and CLI refusals |

The two source counts are test assertions: the fixed four-fives hand admits
138,625,344 of 399,072,960 hidden worlds (33/95) in ten cells; the fresh four-fives
family admits 1,843,717,075,200 full deals in 100 cells. Its 13,300 viewer hands
satisfy an ownership shape, not the whole criterion for bidding 84. Test fixtures,
not a new strategic study; no play-strength result is claimed.

## Verification contract

The complete [[silas]] suite reruns the book rules at their positions and seatings,
the odds and posterior readings, the bidding proofs, and the pinned encodings.
`book_fixture` checks all 39 diagram states (36 numbered hands, two samples and a
later state), 450 tile appearances, all narrated lines and annotation records,
including the known source conflicts. The constructor additionally completes every
legal prefix of the eight full deals at all four seatings and viewer chairs, then
replays each completion from every seat. Hand 33 stops before the recorded illegal
24th play; the source fixture is preserved.

The full repository gate is `SILAS_FULL_PROOF=1 python -u scripts/check`; the flag
asks the bidding proofs to enumerate all hands. The integration is rebased onto
main's reading-policy support and its fence typing fix; neither is reimplemented
here. The README's opening and construction examples are synchronized with their
executable test.

At integration code commit `c4697e6e` (main base `07b5484a`), the full gate is
GREEN: 17 suites pass, zero fail; Forge is unavailable without CUDA and the gate
excludes retired Burl. Silas passes 485 tests across 37 binaries plus doctests.
The language check compiles all 110 sources with zero failures and one existing
warning. All 19 book policies run through both `allowed` and `decide` over all
205 available narrated frames of 33 hands, with matching provenance, legal
allowed moves and any selected move in the allowed set. The two records whose
continuations are refused remain bounded at their last valid prefix. Another
300 CLI receipts (100 fixed, 100 fresh opening, 100 later-position worlds) pass
construction validation and an independent tile-partition check.

The new reading-based policy exposes a separate diagnostic defect: `coverage`
recompiles funnel guards without the policy's registered reading predicates.
[Issue #164](https://github.com/jasonyandell/mk5-main/issues/164) tracks it outside
this integration. The 18 other policies pass CLI coverage; all 19 pass the
compiled-policy replay described above, which does not use the defective
diagnostic path. Verification logs and receipt files remain in this worktree's
`scratch/construction-verification/`.

## Boundaries

Uniform physical completion is not a policy posterior. Opening construction has
an explicit standard-predicate fragment, keeps the auction and declaration fixed,
and does not synthesize histories. A budget refusal is not impossibility. Exact
counts do not make estimates from finite draws exact. New strategic findings and
unrelated opportunities are outside this integration. See [[silas]] and the
[[book-authority]].
