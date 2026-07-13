---
title: Arena — full-game harness
kind: entity
first_seen: 2026-06-12
last_updated: 2026-07-13
status: active
phase: 2026-07-13 — landed; the measuring stick for the champion ladder. Runs on the repaired uniform-completion-dp-v1 sampler; measurement baseline reproduced at [[stage-0-closure]]; canonical decision provenance via [[partnership-decision-record-v1]].
---

## What it is

The full-game Texas 42 harness (`arena/`): four seats, real auctions, hands
until a team reaches 7 marks. Rung #20 of the [[champion-ladder]] (GitHub
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

First physics (2026-06-12): a seven-line static risk-budget bidder beats
always-bid-30 by +0.78 marks/game under identical oracle play, and 192 full
games cost ≈ 150 s on M5 Max MPS — cheap enough to be the default eval for
every rung above it. The rung results measured here (#20 first physics, #21
Gus bidder, #22 net bidder, #25 belief-weighting nulls, #27 score-conditioned
play) live on [[champion-ladder]].

## Design

- The zeb engine (`forge/zeb/game.py`) owns the play phase; the arena owns
  the auction ([[rules-of-42]] §Bidding: one bid each, left of shaker first,
  point bids 30–41 then marks, reshake on pass-out, forced 30 after max
  redeals), marks scoring (§Scoring: `max(1, bid // 42)` marks, set pays the
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
- Shipped bidders: `heuristic` (static Roberson risk-budget — the wave-2.B
  arithmetic from [[w42-bookval-v1-wave2-bid-aware-atlas]], ported to
  `arena/hand_metrics.py` with an equivalence test), `bid30`, `random`,
  `gus[:N[,wp]]` (`champion.GusBidder`, rung #21), `net[:wp]`
  (`champion.NetPointsEvaluator`, rung #22), and the value-native
  `margin[:wp]` / `jud` family ([[jud]]). Play: `lens:<utility>` over the
  forge E[Q] PDF, `scorelens[:band]` (rung #27), `judplay`/`judsearch`
  ([[w42-jud-v1]]), or `random`.
- `BidContext` carries the live game score (`marks`, `marks_to_win`), so a
  bidder can condition on it — the channel the marks-to-7 utility rides. The
  static bidders ignore it; the model-backed bidders use it.

## Measurement surfaces

[[partnership-decision-record-v1]] adds a post-match surface: one
replay-verified row per play, with distinct public-state, actor-information,
auction/score-context, and offline-world identities, and exact
bidder/player/artifact/sampler/utility/partner provenance (`--emit-decisions`).
The exporter changes no policy execution. Action likelihood, plan state,
fixed/shuffled cohort, and Q/PDF values remain explicitly unavailable.

`--emit-snapshots` dumps every contracted hand's real deal + per-seat auction
for corpus generation (`forge.cli.generate_eq_from_snapshots`) — the rung-#26
bridge that feeds the belief and `V_realized` training loops
([[w42-champion-selfplay-fixed-point]], [[w42-jud-v0]]).

## Sampler boundary

[[world-sampler-mrv-audit]] falsified the validity guarantee of the legacy
`WorldSamplerMRV` shared by the historical Lens and arena runs; the repaired
`uniform-completion-dp-v1` (with the `4123b2d5` MPS fix) is the only sampler at
HEAD. [[stage-0-closure]] (2026-07-13) closed the question the repair opened:
C0 reproduces on two held-out seed blocks, the symmetry sanity is clean, and
historical exposure is quantified — legacy harm was real, rare, tail-bound,
and not load-bearing for the reproduced conclusions. The repaired sampler
roughly doubles a C0 block's wall time on MPS; it is kernel-launch bound, so
batch width, not device, is the throughput lever.

## Performance

- **Perf pass 1 (2026-07-06, `d678598`):** dispatch/sync reduction on the
  oracle decision path, byte-identical results as the gate. Key finding: the
  arena is **dispatch-bound, not compute-bound** on MPS (~1,500–2,000 kernel
  launches and ~25–45 GPU→CPU syncs per tick). Paired MPS bench: 0.56 → 1.34
  games/s (**2.38×**). Full profile: `docs/arena-perf-2026-07-06.md`.
- **Perf pass 2 (2026-07-06):** fast batching — `run_paired` pools both halves
  of the paired match into one lockstep batch, killing the straggler tail.
  Byte-identity is structurally impossible here (batch composition feeds the
  world-sampling RNG), so the gate was relaxed to distribution-level
  equivalence (made-rate p=0.78, mark-margin p=0.64 over 256 games/mode).
  **1.55–1.63× at 32 games, 1.29–1.41× at 128**; `--fast-batching` is the
  default, `--no-fast-batching` restores the byte-identical sequential path.
- These numbers were measured on the legacy sampler; the repaired sampler's
  cost is measured at [[stage-0-closure]].

## Limits (v0)

- Shipped bidders declare pip trumps only and never bid past 42; the rules
  layer itself supports mark bids (84, then +42 per raise). Special
  contracts (nello, sevens, plunge) are out of scope.
- No belief carryover between hands within a game; each hand's play starts
  from the uniform world prior.
- Both sides of a paired match are [[pimc]] players, so the harness is
  **information-blind**: play-marks cannot reward belief or concealment value
  ([[champion-design-review]] caveat 1).

## Links

- [[champion-ladder]] — the rung record this harness measured; [[champion]] —
  the player it measures for
- [[stage-0-closure]] — the reproduced measurement baseline on the repaired
  sampler; [[world-sampler-mrv-audit]] — the audit that forced the repair
- [[partnership-decision-record-v1]] — the canonical decision/provenance seam
- [[w42-lens-v1-utility-head-to-head]] — the play-only paired-seed
  predecessor and the lens utilities the arena reuses
- [[w42-bookval-v1-wave2-bid-aware-atlas]] — validated the risk-budget
  arithmetic the heuristic bidder runs on
- [[gen-fleet]] — bid-30 corpus shape; the arena is the consumer that finally
  exercises `contract_threshold_bins` with real bids (rung #23 threaded
  `bid_value` through generation the same day — [[champion-ladder]])
