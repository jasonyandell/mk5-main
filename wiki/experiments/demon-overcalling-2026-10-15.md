---
title: "Demon: contesting an opponent's point bid"
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: complete
---

**STRATEGY-FUSION FENCE.** This is the next development phase of the fenced
[[the-demon]] after [[demon-vs-zeb-2026-10-15]]'s completed attempt 1 failed
the joint game-win criterion. Attempt 2 completes the objective against both
pinned native Zeb 0.4.0 bidders on Apple M5 Max: its game-win intervals exceed
chance and its total seat runtime is lower. Attempt 1's result is unchanged.

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

## Completed development and selection

All 192 games finished. The full audit checks 1,941 hands and 5,626 candidate
bid calls against their recorded forecasts, plus seeds, straight declarations,
request boundaries, source pins and whole-seat cost sums. All three arms fit
the runtime bar. The fixed rule selects **Over 0.50**. This sample does not
establish that 0.50 is the optimal threshold or superior to 0.25.

Intervals are 95% whole-pair bootstrap intervals; chance is 50%, marks baseline
is zero, and runtime ratios divide Demon by Zeb.

| Arm | Wins–losses | Win percentage [interval] | Marks/game [interval] | CPU ratio [interval] | Wall ratio [interval] |
|---|---:|---:|---:|---:|---:|
| Control | 39–25 | 60.94 [50.00, 71.88] | 1.234 [0.500, 1.984] | 0.6071 [0.5932, 0.6210] | 0.6077 [0.5938, 0.6216] |
| Over 0.50 | 49–15 | 76.56 [68.75, 84.38] | 2.109 [1.516, 2.703] | 0.6296 [0.6155, 0.6438] | 0.6306 [0.6165, 0.6449] |
| Over 0.25 | 47–17 | 73.44 [65.62, 81.25] | 1.984 [1.328, 2.641] | 0.6325 [0.6182, 0.6470] | 0.6331 [0.6187, 0.6477] |

[All development results and raw evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/86a619cdf34106bbe274151b6e354d0797b91aa9/demon/2026-10-15-vs-zeb/development-2/results)
include all three reports, full audit and selection, games, costs, traces,
source reuse receipts and executable scripts. Selection, audit, reports, a raw
game and manifest were freshly downloaded and byte-verified.

## Frozen confirmation attempt 2

The selected source, exact participant, binary and harness hashes were
[publicly frozen before play](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/2e22b25e532ffa62e15d680efc28fb9d82577af2/demon/2026-10-15-vs-zeb/confirmation-attempt-2/freeze).
The trial uses **512 fresh duplicate pairs / 1,024 games per
opponent**, seed base `0xD4300000`, against both pinned native Zeb policies.
The larger size accounts for attempt 1's uncertainty and the stricter
alpha-spending threshold. Attempt 2 uses alpha `1/120` (99.1667% two-sided
intervals), 40,000 whole-pair bootstrap resamples, seed 666. Both game-win lower
bounds must exceed 50%; both CPU and wall ratio upper bounds must be at most
1.20 for each opponent. Faster is acceptable. All games finish before judgment;
no attempt-1 or development games enter the confirmation. Every attempt is
retained regardless of its outcome. These criteria were fixed before play.

The selected record is `demon-pimc-n100-bidc102a137-straight-2e233d29`, source
`2e233d29387c2aea0404adffadcaffac3a845a38`, bidder
`pimc-over:64:0.875:0.5`. The native binary SHA-256 is
`1c2ccee87a02069ef4e14d957e97c6042007694abd79058bf8d3f1b73721494e`.
The selection SHA-256 is
`7dfe0a0af7c2a7b3b36c7c5e24edfcd20bff63118c1ba3f249c3f74e5de9328b`;
the plan SHA-256 is
`69ac62f7cab9a5202c420de1d795c15f31e58471a1d5c60bc2c1b59f43352567`.
The plan pins all runtime workspace and npm files; each pair verifies them.
Registry metadata still identifies Zeb 0.4.0 as latest before this freeze.

Measurement includes each seat's startup, bidding, declaration, play, close,
IPC and Python adapter on the Apple M5 Max. Node's waited native descendant
CPU is included. Build, install and the common referee are excluded. The
fixed interleaved schedule runs at most three serial pairs through shared
CPU slots. Preflight checks both actual runner setups without creating seats:
zero confirmation games had started when the freeze was published.

## Completed confirmation

All **2,048 games** finished, with no dropped games, restarts or infrastructure
errors. Both opponents qualify. Intervals below are the predeclared **99.1667%
whole-pair bootstrap intervals**; chance is 50%, marks baseline is zero, and
runtime ratios divide Demon by Zeb (parity is 1).

| Native Zeb 0.4.0 bidder | Wins–losses | Win percentage [interval] | Marks/game [interval] | CPU ratio [interval] | Wall ratio [interval] |
|---|---:|---:|---:|---:|---:|
| Match-aware, pip-only | 667–357 | 65.14 [61.52, 68.85] | 1.813 [1.535, 2.095] | 0.6788 [0.6737, 0.6839] | 0.6796 [0.6745, 0.6847] |
| Standard, nine declarations | 678–346 | 66.21 [62.70, 69.63] | 1.316 [1.080, 1.551] | 0.2853 [0.2835, 0.2870] | 0.2854 [0.2837, 0.2872] |

The full audit covers **21,412 hands and 62,242 candidate bid decisions**:
straight declarations, seed schedule, frozen source hashes, seat-visible
request boundaries, each bid against its recorded forecast, whole-seat cost
sums, raw/report totals and the predeclared verdict. All checks pass.

These are complete straight-42 games to seven marks, with native play and
forecasting on both sides. Zeb uses source
`29f519953467f6aacd78bd17700f3c67fda6b4eb`, 160 worlds, seed 0, shipped soft
reading and exact margin ties. Its JavaScript bidding policies are unmodified.
Package/native parity covers 203 requests, 27 adapter cases and four complete
games (47 hands). The npm registry still identified 0.4.0 with the same
integrity and shasum after the trial. This is evidence for those pinned
opponents on this machine, not for other processors or future Zeb releases.

### Runtime detail

The runtime criterion is total seat cost, including the operations described
above. Across the match-aware trial Demon used 6,362.39 CPU seconds and
6,366.23 wall seconds; Zeb used 9,372.81 and 9,367.89. Against standard Zeb the
totals were 7,031.43 / 7,035.65 for Demon and 24,649.73 / 24,650.95 for Zeb.
CPU and wall have independent clocks; CPU includes waited native descendants.

Descriptive per-call latency percentiles in milliseconds (p50 / p95):

| Trial | Operation | Demon | Zeb |
|---|---|---:|---:|
| Match-aware | Bid | 163.750 / 259.058 | 240.944 / 280.518 |
| Match-aware | Play | 0.294 / 85.915 | 0.581 / 69.246 |
| Match-aware | Startup | 6.173 / 7.088 | 48.329 / 50.684 |
| Standard | Bid | 164.247 / 259.609 | 697.105 / 795.073 |
| Standard | Play | 0.313 / 88.985 | 0.518 / 69.754 |
| Standard | Startup | 6.115 / 7.161 | 48.360 / 50.573 |

Demon's play p95 is higher in both trials. Lower total seat runtime does not
mean every operation is faster. All operation distributions, including
declaration and close, are retained in the reports.

## Evidence and registered participant

The complete CC0 results, raw games, costs, traces, reports, audit, frozen
protocol and executable source are public in [[demon-zeb-attempt-2]]. Both
reports, verdict, audit, a raw game and manifest were freshly downloaded and
byte-verified. The dataset card records this result alongside the failed
attempt and earlier declaration-domain correction.

The exact selected participant is registered in the existing tournament store:
[immutable participant record](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/blob/93553b19f17719caff725b8aa70afe33876aa69c/participants/demon-pimc-n100-bidc102a137-straight-2e233d29.json).
Its freshly downloaded canonical bytes match the frozen selection; SHA-256
`0c05f48ca2c73008f8faba83e5f69cf5d8490cc0a453acb19546c6393314fee2`.
`DEMON_OVERCALL_N100` appends that record to `tournament/participants.py`'s
roster without changing any existing participant. These research games are
not relabeled as standing-ladder faceoffs or Bradley–Terry ratings.

Final roster integration passes `scripts/check --fast`: fence 8,
py-tournament 44, wiki and reuse, with no failed or unavailable suites.
A separate byte check verifies equality to the frozen selected and registered
record, straight-game eligibility, all three source trees and preservation of
every existing roster record.

The policy is merged at [[2e233d29]]. The freeze was
[recorded on issue 165](https://github.com/jasonyandell/mk5-main/issues/165#issuecomment-6189308298)
before launch. Boundary tests
cover every seat orientation, opponent versus partner, no point raise left,
unchanged existing specs and malformed parameters. Whole-game tests check
legality and determinism. The policy's `scripts/check --fast` is GREEN: fence 8,
py-tournament 44, wiki and reuse. The native binary and all three Rust trees
are unchanged from attempt 1; its SHA-256 remains `1c2ccee8…`. The public-tree
rebuild matches all 185 held-out replies and node counts; the source-reuse
receipt verifies the same immutable Rust trees under source `2e233d29`.

[Public frozen protocol](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/fc266b135b6830785edb9b725c8cfae8bc41d478/demon/2026-10-15-vs-zeb/development-2/protocol)
includes executable configuration, runners, source patch and check receipts.
The plan and manifest were freshly downloaded and verified. The source and
protocol were recorded [before play](https://github.com/jasonyandell/mk5-main/issues/165#issuecomment-6188236020).
The selected overcall rule improves this measured opponent comparison without
additional forecast requests. It does not establish an optimal cutoff or a
PIMC ceiling; the strategy-fusion fence remains unchanged.

## Links

[[the-demon]] · [[demon-vs-zeb-2026-10-15]] · [[tournament]] ·
[[match-equity]] · [[strategy-fusion-fence]]
