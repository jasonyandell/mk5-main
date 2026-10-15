---
title: "Demon: contesting an opponent's point bid"
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: active
---

**STRATEGY-FUSION FENCE.** This is the next development phase of the fenced
[[the-demon]] after [[demon-vs-zeb-2026-10-15]]'s completed attempt 1 failed
the joint game-win criterion. It does not change that trial's result.

## Question and implemented policy

Can a lower threshold for contesting an opponent's point bid improve whole
game wins against pinned native match-aware Zeb 0.4.0 within the same runtime?

The existing `pimc:64:0.875` bidder requires the same optimistic double-dummy
make fraction whether opening or overcalling. Passing over an opponent leaves
that opponent holding the contract. [[match-aware-bidding-2026-10-14]] finds
this distinction consequential for Zeb; its calibration does not transfer
automatically to PIMC's different forecast.

The implemented experimental spec is
`pimc-over:<worlds>:<opening-cutoff>:<opponent-cutoff>`. It lowers the threshold
only when an opponent holds a bid and a point raise is legal. It keeps the
minimum legal point raise, partner pass, nine straight declarations, common
worlds and declaration tie rules. There are no mark overcalls and no extra
forecast requests. Thresholds are experimental DD decision parameters,
not calibrated imperfect-information make probabilities. Existing participant
specs retain their behavior and bytes; new specs get distinct identities.

## Fixed development protocol

All arms use 100 play worlds, 64 bidding worlds, uniform play sampling and
exact margin ties. The native Zeb source, shipped match-aware bidder, runtime
accounting and machine are the same as attempt 1. Every game is complete
straight 42 to seven marks, both seatings of each duplicate pair.

| Arm | Bidder | Duplicate pairs | Games |
|---|---|---:|---:|
| Control | `pimc:64:0.875` | 32 | 64 |
| Over 0.50 | `pimc-over:64:0.875:0.5` | 32 | 64 |
| Over 0.25 | `pimc-over:64:0.875:0.25` | 32 | 64 |

All three use fresh seed base `0xD4200000`, with pair index added. No early
stopping or discarded games; infrastructure failures are preserved and audited.
One serial pair runs in each shared `scripts/cpu-slot` batch. The existing
reporter gives 95% whole-pair bootstrap intervals (40,000 resamples, seed 666).
These are development comparisons, never confirmation evidence.

Selection occurs only after all three arms finish. Among arms whose CPU and
wall ratio upper bounds are at most 1.20, select the highest game-win fraction;
ties go to higher marks/game, then lower total CPU. If no arm fits the runtime
bar, continue development without confirmation. No threshold is changed
mid-arm. The control may win selection.

## Next confirmation

Before any attempt-2 game, freeze the selected source, exact participant,
binary and harness hashes. Use **512 fresh duplicate pairs / 1,024 games per
opponent**, seed base `0xD4300000`, against both pinned native Zeb policies.
The larger size accounts for attempt 1's uncertainty and the stricter
alpha-spending threshold. Attempt 2 uses alpha `1/120` (99.1667% two-sided
intervals), 40,000 whole-pair bootstrap resamples, seed 666. Both game-win lower
bounds must exceed 50%; both CPU and wall ratio upper bounds must be at most
1.20 for each opponent. Faster is acceptable. All games finish before judgment;
no attempt-1 or development games enter the confirmation. Every attempt is
retained regardless of its outcome. There is no promotion yet.

## Status

The policy is implemented; development games have not started. Boundary tests
cover every seat orientation, opponent versus partner, no point raise left,
unchanged existing specs and malformed parameters. Whole-game tests check
legality and determinism. The protocol is fixed before measurement; #165 remains
open until the full strength/runtime objective is achieved.

## Links

[[the-demon]] · [[demon-vs-zeb-2026-10-15]] · [[tournament]] ·
[[match-equity]] · [[strategy-fusion-fence]]
