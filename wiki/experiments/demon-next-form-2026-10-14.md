---
title: "The Demon's next form: declaration evidence and PIMC bidding"
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

**STRATEGY-FUSION FENCE.** The [[the-demon]] remains a sanctioned tournament
opponent only. No values or labels from this work enter Walt, Zeb or Silas policies.

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

The confirmations are running; no new participant is promoted on development
results. [Issue #165](https://github.com/jasonyandell/mk5-main/issues/165) tracks
the strength claim and broader opponent comparisons. Results and public evidence
will be recorded here when the fixed trials finish.

## Links

[[the-demon]] · [[demon-power-2026-10-14]] · [[silas]] · [[tournament]] ·
[[strategy-fusion-fence]]
