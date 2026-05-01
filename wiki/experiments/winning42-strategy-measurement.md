---
title: Winning 42 Strategy Measurement
kind: experiment
first_seen: local-2026-04-30
last_updated: local-2026-05-01
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

## Chapter Workstream

Each Winning 42 chapter now has a bead-backed wiki page. The pages treat the book as
a source of measurable hypotheses for [[gus]], [[burl]], and [[forge]], not as an
authority to hard-code. A chapter is complete only when its bead closes with a wiki
update that lists source-backed concepts, detector inputs, metrics/tests, likely data
sources, implementation notes, and readiness for enumeration, oracle rollout, Gus,
or Burl analysis.

- [[winning42-ch01-in-a-nutshell]] — foundational rule/state accounting.
- [[winning42-ch02-bidding]] — bidding as risk budget.
- [[winning42-ch03-bidder-play]] — bidder sequencing after winning the bid.
- [[winning42-ch04-partner-support]] — helping the bidder make the contract.
- [[winning42-ch05-setter-defense]] — setting the bidder.
- [[winning42-ch06-concentration-style]] — attention, inference, and style.
- [[winning42-ch07-taking-every-trick-84]] — bidder-side 84 plan.
- [[winning42-ch08-setting-84]] — defending the 84 bid.
- [[winning42-ch09-doubles-no-trump]] — doubles-as-trump and no-trump regimes.
- [[winning42-ch10-tournament-scoring]] — reward-function drift under scoring systems.
- [[winning42-ch11-table-talk]] — legal inference versus illegal information.
- [[winning42-ch12-advanced-bidding-playing]] — advanced exception handling.
- [[winning42-ch13-optional-variations]] — ruleset gates and contamination guards.
- [[winning42-ch14-history-tournaments]] — tournament and population ecology.
- [[winning42-ch15-celebrities-style]] — player style and partnership patterns.
- [[winning42-ch16-statistical-odds]] — statistical validation and odds checks.

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
