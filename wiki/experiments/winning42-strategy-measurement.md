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
- [[w42-final-empirical-strategy-report]] — first w42 survey synthesis: strategy tags
  help small models, exact substrate claims can be supported, and most tactical claims
  remain context-limited or underpowered until direct detectors and paired tests land.
- [[w42-book-claim-synthesis-and-ai-directions]] — post-phase-4 synthesis: confirmed
  vs unconfirmed claims, distribution-aware E[Q] alternatives, AI/Burl/Gus directions.
- [[w42-book-validation-campaign]] — live multi-wave campaign (epic `t42-4zi6`) to take
  all 64 ledger rows from "evidence on slice" to paired counterfactuals on real auctions
  or injected late states.

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

Each Winning 42 chapter has a bead-backed wiki page. The pages treat the book as
a source of measurable hypotheses for [[gus]], [[burl]], and [[forge]], not as an
authority to hard-code. The first harvest closed all sixteen chapter beads and left
one page per chapter with source-backed concepts, detector inputs, metrics/tests,
likely data sources, implementation notes, and readiness for enumeration, oracle
rollout, Gus, or Burl analysis.

- [[winning42-ch01-in-a-nutshell]] — foundational rule/state accounting: legal trick winner,
  follow obligation, trump membership, count capture, make/set threshold, walkers,
  and lead-control sanity buckets.
- [[winning42-ch02-bidding]] — bidding as risk budget: trump suitability, off-risk,
  four/five-suit danger, double protection, duplicate count accounting, partner-help
  assumptions, and bid-only-enough discipline.
- [[winning42-ch03-bidder-play]] — bidder sequencing: trump pull defaults and exceptions,
  reentry trump preservation, partner donation windows, double-ahead-of-off, setter
  pounce creation, dangerous outstanding trump, count inventory, and laydown checks.
- [[winning42-ch04-partner-support]] — partner help: safe count donation, lead capture
  for support, virtual boss tiles, lead-away damage, disruptive trump leads, and
  bidder off-suit vulnerability.
- [[winning42-ch05-setter-defense]] — setter defense: pounce windows, deliberate void
  creation, count-on-bidder-off, extra-count-to-set, trump-set detection, trump-rich
  setter policy, count protection, and position-sensitive trump-ins.
- [[winning42-ch06-concentration-style]] — attention and style: void inference,
  bid-derived hand inference, failed-bid signal, trump attention traps, low-trump
  disguise, late-hand tile promotion, count donation prevention, and mistake recovery.
- [[winning42-ch07-taking-every-trick-84]] — bidder-side 84: eligibility, laydown
  proof, protected offs, straight-off risk, off-suit exhaustion, trump exhaustion
  before doubles, next-to-last forcing double, final walkers, and 42-vs-84 gating.
- [[winning42-ch08-setting-84]] — defending 84: live last-trick weapons, live doubles,
  same-suit pairs, protectors, forced versus voluntary pair breaks, throwaway priority,
  dynamic abandonment, and set attribution.
- [[winning42-ch09-doubles-no-trump]] — doubles/no-trump regimes: doubles-as-trump
  strength, no-trump count exposure, fallback-follow variants, double protection,
  double-rich hands, and regime-specific legal/strategic buckets.
- [[winning42-ch10-tournament-scoring]] — scoring-objective drift: points versus marks,
  early mark terminal states, defender partial-point erasure, set severity compression,
  special-bid multipliers, and low-bid distortion.
- [[winning42-ch11-table-talk]] — legal inference versus illegal information:
  anti-leakage boundaries, unauthorized partner information, trump-identification
  assistance, attention responsibility, renege detection, and delayed reconstruction.
- [[winning42-ch12-advanced-bidding-playing]] — advanced exception handling: crisis
  trump, protection stripping, pounce triggers, double-not-always-right, 84 asset
  preservation, natural-bid anomalies, and reputation/style priors.
- [[winning42-ch13-optional-variations]] — ruleset gates and contamination guards:
  straight-42 mode, Nel-O, Sevens, Plunge/Splash, forced bidding, 84 raises, first-lead
  legality, small-end disallowance, high-bid scoring, and direct 84 legality.
- [[winning42-ch14-history-tournaments]] — tournament and population ecology:
  tournament pressure, partner synergy, aggressive bidding culture, setter skill gaps,
  belief memory, opponent adaptation, time management, standardization, and random
  partner robustness.
- [[winning42-ch15-celebrities-style]] — player style and partnership: watchfulness,
  overbid restraint, low-game willingness, dot-count discipline, partner-fit residual,
  partner legibility, style classification, and social-pressure robustness.
- [[winning42-ch16-statistical-odds]] — statistical validation: exact hand-count,
  suit/void, double-count, modal-shape, and four-trump configuration checks, plus the
  odds baselines needed to calibrate book thresholds.

## Harvest Findings

The chapter sweep produces five reusable measurement clusters:

- **Ruleset and state invariants.** Chapters 1, 9, 10, 11, and 13 define deterministic
  legality and objective tags: trick winner, follow obligation, trump regime, count
  captured, scoring mode, legal information boundary, and variant gate.
- **Risk budget and contract selection.** Chapters 2, 7, 9, 10, 12, and 16 turn bidding
  into measurable loss accounting: off-risk, protection, duplicate count liability,
  84 eligibility, no-trump/doubles regime risk, score objective, and odds thresholds.
- **Play sequencing and tactical windows.** Chapters 3, 4, 5, 6, 7, 8, and 12 define
  action-local detectors: pull trump, save reentry, donate count, pounce, create void,
  preserve last-trick weapons, break pairs, strip protection, and abandon a failed plan.
- **Belief, memory, and table discipline.** Chapters 6, 11, 14, and 15 make public
  evidence a first-class object: void inference, bid-derived priors, attention failures,
  anti-leakage checks, opponent adaptation, partner legibility, and style-conditioned
  decisions.
- **Statistical and population analysis.** Chapters 14, 15, and 16 define the outer
  calibration layer: enumeration baselines, confidence intervals, score/tournament
  population splits, player-style clusters, partner-fit residuals, and random-partner
  robustness.

The claim ledger is intentionally conservative. Most chapter claims are harvested as
`context-limited` or `underpowered` because the work produced detectors and tests, not
full empirical verdicts. Chapter 16 is the exception: its page records exact enumeration
checks for several odds claims and establishes the shape of the statistical-analysis
layer. Future experiment pages should promote a claim to `supported` or `contradicted`
only after enumeration, oracle rollout, Gus belief/policy evaluation, or Burl trace
review actually runs.

The first w42 close-out keeps that posture. The survey found useful strategy signal
in model inputs, but only deterministic arithmetic/ruleset/scoring predicates moved
to supported evidence. Tactical advice such as pounce timing, safe donation, 84
weapon preservation, and bid-margin discipline remains the next digging surface, not
a settled conclusion.

## First Work Package

Build a `strategy_tags` analyzer over generated games that emits public-state JSON tags
such as `role_regime`, `risk_budget`, `live_count`, `count_liability`, `outstanding_trumps`,
`void_evidence`, `key_tile_owner_belief`, `off_protection`, `walker_candidates`,
`partner_donation_window`, `setter_pounce_window`, and `rule_variant`.

After the chapter sweep, the first package should expand beyond the original MVP tags:

- **Action-local play tags:** `pulls_trump`, `saves_reentry`, `plays_off`, `throws_count`,
  `donates_to_partner`, `donates_to_opponent`, `pounces_count`, `creates_void`,
  `breaks_pair`, `preserves_last_trick_weapon`, `beats_current_winner`, and
  `promoted_tile`.
- **Bid/risk tags:** `candidate_trump_shape`, `off_count_liability`, `off_protected_by_double`,
  `duplicate_count_exposure`, `natural_bid_bucket`, `eighty_four_candidate`,
  `no_trump_risk`, `doubles_trump_risk`, and `score_objective_mode`.
- **Belief/attention tags:** `void_evidence_strength`, `bid_implied_suit_strength`,
  `failed_bid_signal`, `outstanding_trump_danger`, `legal_info_boundary`,
  `partner_legibility`, and `style_prior_bucket`.
- **Analysis buckets:** `ruleset_variant`, `tournament_scoring_mode`, `partner_fit_bucket`,
  `population_style_bucket`, `odds_threshold_bucket`, and `claim_ledger_status`.

The first report should have three tables:

1. Book claims checked against enumeration/oracle.
2. Gus belief quality by concept bucket.
3. Burl regret/tail-risk by concept bucket.

## Links

[[gus]] · [[burl]] · [[forge]] · [[topics/regret-eval]] ·
[[experiments/gus-strategy-tags-probe]] · [[w42-final-empirical-strategy-report]]
