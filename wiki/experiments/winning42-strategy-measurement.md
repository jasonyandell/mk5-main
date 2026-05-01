---
title: Winning 42 Strategy Measurement
kind: experiment
first_seen: local-2026-04-30
last_updated: local-2026-04-30
status: active
---

## Summary

The Winning 42 book is now treated as a strategy-hypothesis source for Gus/Burl,
not as ground truth. The working breakdown lives in:

- `scratch/winning42/strategy_measurement_breakdown.md`

The core claim: each strategy concept should become one or more detector labels,
adversarial buckets, regret metrics, belief-calibration tests, or training examples.

## Current Artifacts

- `scratch/winning42/strategy_measurement_breakdown.md` — project shape, MVP detectors,
  chapter harvest map, first work package, and a lengthy analysis catalog.
- `scratch/winning42/winning42.with_figures.md` — OCR/preview source used for the chapter harvest.
- [[experiments/gus-strategy-tags-probe]] — first promoted empirical probe showing that
  explicit strategy tags improve a tiny Gus-like policy, though not enough to beat `E[Q] N=10`.

## Analysis Catalog Shape

The catalog enumerates surfaces for:

- rules and state accounting
- bidding
- hand shape
- off-risk and protection
- count liability
- trump control and reentry
- lead choice
- follow/slough/discard decisions
- partnership
- setter defense
- belief and inference
- attention and memory
- 84 bidder play
- 84 defender play
- doubles-as-trump and no-trump regimes
- scoring and tournament objectives
- variants and etiquette
- model evaluation and training
- population, style, and partnership ecology
- statistical analysis

The final statistical-analysis layer is explicit: verify book odds by enumeration,
calibrate claims against forge/oracle outcomes, report confidence intervals and paired
tests, and maintain a supported / contradicted / context-limited / underpowered claim ledger.

## First Work Package

Build a `strategy_tags` analyzer over generated games that emits public-state JSON tags
such as `role_regime`, `risk_budget`, `live_count`, `count_liability`, `outstanding_trumps`,
`void_evidence`, `key_tile_owner_belief`, `off_protection`, `walker_candidates`,
`partner_donation_window`, `setter_pounce_window`, and `rule_variant`.

The first report should have three tables:

1. Book claims checked against enumeration/oracle.
2. Gus belief quality by concept bucket.
3. Burl regret/tail-risk by concept bucket.

## Links

[[gus]] · [[burl]] · [[forge]] · [[topics/regret-eval]] · [[experiments/gus-strategy-tags-probe]]
