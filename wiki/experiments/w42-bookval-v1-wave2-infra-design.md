---
title: w42 Book Validation v1 — Wave 2 Infra Design
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-05-03
status: active
parent_bead: t42-4zi6
---

## Summary

Wave 2 of the [[w42]] book-validation campaign builds the two infra pieces
that block most of the remaining unconfirmed claims. This page is a design
record, not a result page; it exists so future agents have a stable target
before any code lands.

The two builds:

- **State-Injection Harness** — generate E[Q] from arbitrary mid-game
  snapshots, not only from fresh deals.
- **Bid-Aware E[Q] Driver** — drive the existing forge generator across a
  bid range on the same hands so the W42 corpus actually contains bid
  variation. (The forge layer already supports `--bid-values`; the W42
  consumer side does not yet exploit it.)

Both unlock specific blockers found in [[w42-phase4-final-claim-audit]] and
[[w42-bookval-v1-wave1-mark-utility-transform]].

## Why Now

[[w42-bookval-v1-wave1-mark-utility-transform]] proved that on a fixed-bid
corpus, mark utility is algebraically identical to `p_make` (the multiplier
collapses to 1). The Ch 10 claims about marks scaling differently at 35/36/42
contracts are therefore *unreachable* on existing data.

[[w42-phase4-final-claim-audit]] identifies several blockers that share the
same root cause: no exact late-state snapshots are available. Reentry
preservation, void creation, low-trump trap, 35/36 pounce pressure, 84
throwaway ladders, and laydown audits all require an arbitrary engine state
to start from rather than a fresh deal.

The forge engine already supports both surfaces in principle:

- `forge.eq.generate.cli` already accepts `--bid-values` for per-game bid
  variation.
- `forge.eq.game_tensor.GameStateTensor.apply_actions` produces well-formed
  late states all the time. The blocker is that
  `forge.eq.generate.pipeline.generate_eq_games_gpu` only ever calls
  `GameStateTensor.from_deals` for initialization.

Both are small surface extensions, not architectural rewrites.

## Build A — State-Injection Harness

### What

A new `GameStateTensor.from_snapshot(snapshot)` class method plus a
`forge.eq.generate.pipeline.generate_eq_from_snapshots` driver that calls
the existing inner loop after replacing the initialization step.

### Snapshot schema

```json
{
  "schema_version": "forge.eq.snapshot.v1",
  "decl_id": 2,
  "bid_value": 30,
  "bidder": 0,
  "hands": [[...4 lists of 0..7 domino_ids each...]],
  "played_mask": [false, ...28 booleans...],
  "history": [[player, domino_id, lead_domino_id], ...],
  "trick_plays": [domino_id_or_-1, x4],
  "leader": 0,
  "score": {"offense": 12, "defense": 6}
}
```

Validation:

- sum of remaining-hand counts + length(history) == 28
- played_mask is consistent with history + trick_plays
- history sequence respects follow-suit rules under decl_id
- trick_plays length matches (current_player - leader) % 4

### Implementation surface

Three files:

- `forge/eq/game_tensor.py` — add `@classmethod from_snapshot` constructing a
  full `GameStateTensor` from a list of snapshots.
- `forge/eq/generate/pipeline.py` — add
  `generate_eq_from_snapshots(model, snapshots, ...)` mirroring
  `generate_eq_games_gpu` but taking snapshots instead of `(hands,
  decl_ids)`.
- `forge/eq/generate/cli.py` — add `--snapshot-file` option that loads a
  JSONL file of snapshots and dispatches to the new driver.

### Pre-built snapshot corpora

Five corpora cover the highest-leverage blocked claims. Each is generated
once and committed under `w42/book_validation_v1/wave2/snapshots/`:

- `reentry_preservation/` — bidder mid-hand with one trump reentry remaining
  and a vulnerable off suit. Tests `ch03-reentry-preservation`.
- `void_creation/` — setter side with one suit they could plausibly void.
  Tests `ch05-void-creation` and related Chapter 5 claims.
- `low_trump_trap/` — bidder holds the dominant trump but a low trump can
  trap them. Tests `ch04-low-trump-trap-against-count-dump`.
- `pounce_window_high_bid/` — high-bid (35-42) contract entering trick 2-3
  with bidder offsuit exposed. Tests `ch12-setter-pounce-high-bid-off`.
- `eighty_four_throwaway/` — 84 contract entering final two tricks with
  defender weapon shape. Tests `ch08-throwaway-priority-ladder`.

Each corpus is a JSONL of 200-500 hand-curated or seed-mined snapshots with
provenance.

## Build B — Bid-Aware E[Q] Driver

### What

A W42-side driver that runs the existing forge generator across a bid sweep
on the same seeds, then joins outputs into a comparable per-bid table.

The forge layer already supports `--bid-values`. The work is on the W42 side:
a wrapper that emits the per-bid `.pt` files, joins them into one table
keyed by `(seed, decl_id, bid_value)`, and recomputes the full distribution
feature family at each bid.

### Implementation surface

One file:

- `w42/book_validation_v1/wave2/run_bid_aware_atlas.py` — driver that loops
  bid_values=[30, 32, 35, 36, 39, 42, 84], calls forge.eq.generate per bid
  on a fixed seed range, joins outputs into one parquet/csv keyed by
  (seed, decl_id, bid_value, decision_idx), and writes a per-bid summary
  manifest.

The first sweep should target a 50-seed × 10-decl × 7-bid grid to keep
runtime tractable. That's 3,500 games — same order of magnitude as
[[w42-branch-atlas-scaled-v0]] (which is 1 seed × 10 decls = 10 games).
GPU runtime at H100 ~100 games/sec implies ~35 sec compute plus tokenizer
overhead.

### Coverage targets

- Ch 02 bid-only-enough: paired contracts at bid=30 vs bid=32 on identical
  hands.
- Ch 07 84 contract regime: bid=84 outcomes vs bid=42 marks on hands that
  are eligible for both.
- Ch 10 special-bid mark multiplier: bids 35/36/42 with mark_ev properly
  multiplied (the algebraic-identity finding that triggered this build).
- Ch 12 setter-pounce-high-bid-off: bid >= 35 paired with bid = 30.

## Sequencing

Both builds can proceed in parallel. They have zero overlap in the file
surface they touch.

The state-injection harness has more downstream consumers in Wave 2's agent
fan-out. The bid-aware driver has fewer consumers but tighter wave-2
dependencies (Ch 10 claim work was explicitly bumped here by Wave 1).

A reasonable assignment:

- Foreground orchestrator: drafts both designs (this page), files the
  beads, dispatches agents, reconciles.
- Agent A (analytics-engineer): implements `from_snapshot` + driver +
  one snapshot corpus end-to-end. The first corpus is a working integration
  test for the harness.
- Agent B (analytics-engineer): implements bid-aware driver + first 50-seed
  sweep, joins outputs.
- Agents C-F: consume A's output to generate the remaining four snapshot
  corpora and run their respective claim probes.
- Agents G-J: consume B's output to run Ch 02 / Ch 07 / Ch 10 / Ch 12 paired
  bid tests.

## Validation Plan

Each agent must produce a wiki page following the contract in
`w42/book_validation_v1/AGENTS.md`. Reconciliation by orchestrator after
each pair of agents finishes (not after the whole wave).

State-injection harness validation:

- `from_snapshot` round-trips: snapshot → tensor → snapshot equals input.
- A snapshot extracted from `apply_actions(from_deals(...))` at decision
  K, when fed back through `from_snapshot`, produces the same E[Q] PDF as
  continuing from `apply_actions`.

Bid-aware driver validation:

- At bid=30 on existing seeds, output matches existing branch_atlas_v1 to
  within sampling noise.
- At bid=84, only 84-eligible declarations and seats produce non-trivial
  E[Q] PDFs.

## Deferred Out Of Wave 2

- Real auction policy and partner/opponent simulator (Wave 3).
- Mark-utility model training (Wave 4 — needs Wave 2.B output as input).
- Belief-attention head training on hidden-threat targets (Wave 5).

## Links

[[w42]] | [[w42-book-claim-synthesis-and-ai-directions]] |
[[w42-bookval-v1-wave1-mark-utility-transform]] |
[[w42-bookval-v1-wave1-distribution-lens-reranker]] |
[[w42-phase4-final-claim-audit]] |
[[w42-branch-atlas-scaled-v0]] |
[[w42-powered-branch-atlas-v1]]
