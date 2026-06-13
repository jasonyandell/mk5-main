---
title: Arena — full-game harness
kind: entity
first_seen: local-2026-06-12
last_updated: local-2026-06-12
status: active
phase: landed; measuring stick for the champion ladder
---

## What it is

The full-game Texas 42 harness (`arena/`): four seats, real auctions, hands
until a team reaches 7 marks. Rung 1 of the [[champion]] ladder (GitHub
issue #20, closed 2026-06-12) — until it existed, "best player" was not a
measurable sentence, because every eval in the stack forced bid=30.

A player is a (BidPolicy, PlayPolicy) pair. Matches are paired-seed with
team rotation (half 1: A as seats {0,2}; half 2: seats {1,3}; identical
deal seeds), so card luck cancels and the auction itself becomes part of
what is measured.

```bash
python -u -m arena.cli --team-a heuristic+lens:ev --team-b bid30+lens:ev \
    --n-games 192 --n-samples 10 --device mps
```

## Design

- The zeb engine (`forge/zeb/game.py`) owns the play phase; the arena owns
  the auction (docs/rules.md §4: one bid each, left of shaker first, point
  bids 30–41 then marks, reshake on pass-out, forced 30 after max
  redeals), marks scoring (§6–§7: `max(1, bid // 42)` marks, set pays the
  defenders), and the game loop. A hand enters play as a fully-formed
  PLAYING state — the `_force_bid_30` construction from
  [[w42-lens-v1-utility-head-to-head]]'s harness, minus the force.
- Lockstep batching: each tick routes every live game's decision to the
  owning side's play policy as one batched call, so the GPU path amortizes
  across games (same shape as `w42/lens_v1/parallel_match.py`).
- The oracle never sees the bid value (`GameStateTensor` carries decl and
  bidder only), so auction-won contracts are in-distribution for the Q
  model; the bid feeds only the utility thresholds, which were already
  parameterized per game ([[gen-fleet]]'s `contract_threshold_bins`).
- Shipped bidders: `heuristic` (static Roberson risk-budget — bid the
  minimum legal raise while it stays within `42 − unique_exposed_points`
  for the best ≥3-tile pip trump; the wave-2.B arithmetic from
  [[w42-bookval-v1-wave2-bid-aware-atlas]], ported to `arena/hand_metrics.py`
  with an equivalence test), `bid30` (opens 30, else passes — the
  historical baseline as a live player), `random`. Play: `lens:<utility>`
  over the forge E[Q] PDF, or `random`.

## First physics (2026-06-12)

Identical oracle play both sides (lens:ev, N=10), bidders differ —
192 games, base seed 1000:

- **heuristic beats bid30 58.9%** (113/192; halves 59.4% / 58.3%),
  mark margin **+0.78/game, 95% CI [+0.28, +1.26]**.
- Mechanism: hand selection. The heuristic made 51.1% of its contracts vs
  the baseline's 43.3%, taking 57.6% of auctions at mean bid 30.9.
- Self-play symmetry sanity (identical specs both sides) sits at ~50%, as
  it must.
- 192 full games ≈ 150 s on M5 Max MPS — cheap enough to be the default
  eval for every rung above it.

The auction was the predicted high-ground ([[champion]]: auction ≫ play
polish) and the very first measurement agrees: a seven-line static bidder
is already worth ~+0.8 marks/game over always-bid-30.

## Limits (v0)

- Shipped bidders declare pip trumps only and never bid past 42; the rules
  layer itself supports mark bids (84, then +42 per raise). Special
  contracts (nello, sevens, plunge) are out of scope.
- No belief carryover between hands within a game; each hand's play starts
  from the uniform world prior ([[champion]] rungs #24–#25 fix this at the
  model level).

## Links

- [[champion]] — the ladder this measures; rung 1
- [[w42-lens-v1-utility-head-to-head]] — the play-only paired-seed
  predecessor and the lens utilities the arena reuses
- [[w42-bookval-v1-wave2-bid-aware-atlas]] — validated the risk-budget
  arithmetic the heuristic bidder runs on
- [[gen-fleet]] — bid-30 corpus shape; the arena is the consumer that
  finally exercises `contract_threshold_bins` with real bids
