---
title: "The Demon's next form: declaration evidence and PIMC bidding"
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-15
status: complete
---

**STRATEGY-FUSION FENCE.** The [[the-demon]] remains a sanctioned tournament
opponent only. No values or labels from this work enter Walt, Zeb or Silas policies.

**Scope correction, 2026-10-15:** the PIMC bidder could select declaration 8,
the nello-only suit convention. It did so in 87 of the bidder confirmation's
5,620 hands. The reported result is for a ten-declaration experimental arena,
**not a valid straight-42 benchmark**. Its tournament promotion is withdrawn;
the immutable record and raw evidence remain. [[demon-vs-zeb-2026-10-15]] holds
the audit, corrected declaration restriction and replacement trials.

## Candidates and invariants

The uniform player of [[demon-power-2026-10-14]] remains available with its
original immutable participant records. The new capabilities are opt-in:

- **Declaration evidence.** Reconstruct the declaring seat's original hand
  in each lawful world by adding its public plays back to its remaining tiles.
  A native port of the arena heuristic predicts its pip declaration. Matching
  hands receive integer likelihood `r`, other hands 1; `r=32` is the candidate.
  This is a categorical model with probabilities `r/(r+6)` and `1/(r+6)`.
  The declaring viewer already knows its hand and gains no evidence from its
  own choice. Non-pip declarations are unmodeled and leave the prior uniform.
- **Direct posterior samples.** Uniform rank proposals use the fiber's count
  DP, accepted with probability likelihood/`r`. Accepted draws are IID under
  the stated model; no world has zero support. If the whole fiber fits the
  budget, exact integer-weighted make and margin sums replace sampling.
  The response's `mass` is the denominator; `worlds` counts solves. With `r=1`,
  the original distinct uniform sampler and decisions are preserved.
- **Contract forecasts.** From one seat's seven tiles, solve the same sampled
  hidden deals under all ten declarations. Each value is an exact double-dummy
  make-or-set outcome. Declaration ties prefer the arena heuristic, then the
  lowest declaration id. `pimc-declare:N` changes declarations only;
  `pimc:N:cutoff` also bids the minimum legal point raise when the best forecast
  reaches the stated cutoff, and passes over a partner's standing bid.

These are model choices, not guarantees of strength. The declaration likelihood
is a hypothesis about a player's behavior. The contract forecast is optimistic;
the cutoff is a decision parameter, not a calibrated win probability. Bidding
mode changes willingness, partner-pass behavior and declaration jointly.

Source: `tournament/demon/src/{hand_model,sampling,bidding,lib,main}.rs` and
`tournament/demon_fenced_seat.py`. The native heuristic is checked against the
arena implementation. Posterior weights and root values are held to exhaustive
minimax; sampling is checked against enumerated likelihoods. Every mode plays
legal, repeatable complete games through the arena referee.

## Evaluation protocol

The baseline is the frozen distinct-uniform 100-world Demon at `228812ff`.
All arms retain 100 play worlds and the make-then-margin decision rule.
Declaration reading retains the heuristic bidder. Bidding candidates retain
uniform play, use 16 forecast worlds, and compare declaration-only behavior
with cutoffs 0.75 and 0.875. The bidding development arms share duplicate deals.

Development selects a bidder by paired marks/game. The selected bidder and the
declaration reader are then frozen for separate 256-pair confirmations on fresh
seeds. The primary statistic is win fraction against 50%; marks/game is secondary.
Intervals resample whole duplicate pairs. Every completed game and native process
cost is retained. These are fixed-play-budget comparisons, not matched-time tests.

## Completed results

Within the ten-declaration experimental arena, the selected bidder beat the prior 100-world Demon:
**324–188 in 512 fresh games**, win fraction **63.28% [59.38%, 67.19%]** against
the 50% chance baseline, marks/game **+1.020 [+0.760, +1.283]**. Its play still
uses 100 distinct uniform worlds. The tested setting is `pimc:16:0.875`:
14 of 16 forecast deals must make. The source pin is [[4454f603]].

All intervals below are 95% duplicate-pair bootstrap intervals, 20,000 resamples,
seed 666. Marks are candidate minus baseline per game. The three bid development
arms used the same deals; their best paired marks selected the frozen bidder
before its independent confirmation. Reader development and confirmation use
the heuristic bidder on both sides. Combined development uses the selected PIMC
bidder on both sides and changes only the reader.

| Arm | Games | Wins–losses | Win fraction, 95% interval | Marks/game, 95% interval |
|---|---:|---:|---|---|
| Reader, development | 96 | 52–44 | 54.17% [48.96%, 59.38%] | +0.260 [−0.042, +0.594] |
| Declarations only, development | 48 | 22–26 | 45.83% [35.42%, 56.25%] | −0.167 [−0.938, +0.625] |
| Bid cutoff 0.75, development | 48 | 28–20 | 58.33% [45.83%, 70.83%] | +0.563 [−0.334, +1.438] |
| Bid cutoff 0.875, development | 48 | 32–16 | 66.67% [56.25%, 77.08%] | +1.146 [+0.333, +1.958] |
| Selected bidder, confirmation | 512 | 324–188 | 63.28% [59.38%, 67.19%] | +1.020 [+0.760, +1.283] |
| Reader, confirmation | 512 | 270–242 | 52.73% [49.61%, 55.86%] | +0.162 [−0.020, +0.340] |
| Reader plus selected bidder, development | 96 | 45–51 | 46.88% [38.54%, 55.21%] | −0.042 [−0.542, +0.448] |

All 1,360 games finished; no game was dropped and neither confirmation stopped
early. The reader did not establish a gain. Combining it with the selected
bidder did not improve development marks, so the plan's conditional combination
confirmation did not run. The selected bidder retained uniform play. These
results compare against the named Demon baselines; broader strength against
Walt, Zeb or other opponents remains unmeasured for this form.

## Cost and validation

The bidder confirmation contains 5,620 hands. Candidate native CPU totals
1,475.998 seconds versus 753.279 for the baseline: **1.9594×**, or 2.883 versus
1.471 seconds per complete game. Peak native RSS is 39,174,144 versus 32,915,456
bytes. Reader confirmation uses 819.341 versus 796.258 native CPU seconds
(1.0290×); combined development uses 273.646 versus 269.836 (1.0141×).

These are native process user+system CPU totals from `wait4`, including contract
forecasts but excluding Python runner and bidder overhead. They are single-run
resource observations, not latency guarantees or matched-time strength results.
The initial batches ran serially; later batches ran up to three independent
pairs through the shared CPU-slot wrapper, as recorded in `execution-note.md`.
No seeds, policies or trial sizes changed with scheduling.

- Original uniform decisions match the frozen baseline on 560 replay requests,
  including all values and node counts.
- The public pinned build matches the frozen research binary on 560 uniform
  and 560 reader play requests plus 20 all-declaration forecasts, every response
  field except elapsed milliseconds.
- The native declaration heuristic matches Python on 2,000 random hands.
  Weighted exhaustive endgames, posterior frequencies, forecasts under every
  declaration, and legal deterministic complete games are tested.
- Both full repository checks passed all 17 runnable suites. The gate reports
  CUDA forge unavailable on this Mac and retired Burl not run. The original
  nine tournament participant records remain byte-identical.

## Publication and remaining work

[Complete public evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/a8795eb591fca57df436b69c0e03fa6d1ba4437d/demon/2026-10-14-next-form)
contains all seven trial folders, plans and frozen selections, raw games, native
costs, reports, runners, source provenance and validation logs: 2,774 files,
11,206,757 bytes, CC0-1.0. Both confirmation reports, one raw confirmation
game and the SHA-256 manifest were freshly downloaded and hash-verified.

The historical immutable tournament participant is
`demon-pimc-n100-bid7ff83929-4454f603`. Its
[published record](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/blob/f0f16c864c3a70f88ba65edcf903d5e95625cef2/participants/demon-pimc-n100-bid7ff83929-4454f603.json)
was freshly downloaded and verified against canonical SHA-256
`e015366afb56f6eb346b05899f37b4d4d12743c8080ba60120d1c8b24c9cf3ea`.
The research games retain their research identities; they are not relabeled
as tournament rating observations. The declaration audit excludes this
participant from current straight-game scheduling and ratings.

[Issue #165](https://github.com/jasonyandell/mk5-main/issues/165) tracks the
remaining matched-compute and pinned-opponent comparisons. Other open knobs
include bidding utility, score-aware risk, adaptive sampling, and readings
calibrated to the actual new bidder. This study changes bidding willingness,
partner-pass behavior and declarations jointly; their individual contributions
are not identified by the winning arm.

## Links

[[the-demon]] · [[demon-power-2026-10-14]] · [[silas]] · [[tournament]] ·
[[strategy-fusion-fence]]
