---
title: w42 Phase 2 Statistics Claims Ledger
kind: experiment
first_seen: local-2026-05-02
last_updated: afd4802
status: complete
---

This page's own open question ("should any subset of these 64 rows become the
canonical central claim ledger?") was resolved by
[[w42-phase4-final-claim-audit]], which carried this exact ledger through to
closure (all 64 rows have evidence and/or bounded blockers).

## Question

Can [[w42]] consolidate the book/wiki statistical claims into a source-backed,
machine-readable ledger without promoting tactical advice beyond the evidence?

This phase-2 pass covers hand odds, trump counts, doubles/no-trump odds, void
frequencies, count exposure, stopper ownership, 84 weapon availability, scoring
transforms, and the existing proxy reports for partner support, setter defense,
and bidder sequencing. It now also incorporates
[[w42-gus-corpus-tactical-claim-deep-dive]] for direct Gus v2 corpus evidence on
setter pounce and partner count-donation boundaries.

## Data And Source Inventory

Primary source slices:

| source | use |
|---|---|
| `scratch/winning42/winning42.with_figures.md:735-1406` | Chapter 2 bidding risk, off exposure, double-ahead protection, and bid-only-enough claims. |
| `scratch/winning42/winning42.with_figures.md:3070-3334` | Chapter 7 84 bidder shapes and straight-off stopper odds. |
| `scratch/winning42/winning42.with_figures.md:3560-3820` | Chapter 8 84 defender weapons, same-suit pairs, and throwaway priorities. |
| `scratch/winning42/winning42.with_figures.md:3838-4239` | Chapter 9 doubles-as-trump and no-trump rules/odds. |
| `scratch/winning42/winning42.with_figures.md:4243-4331` | Chapter 10 marks-vs-points scoring claims. |
| `scratch/winning42/winning42.with_figures.md:9500-9600` | Chapter 16 hand-count, void, double-count, four-trump, and partner-double priors. |

Local evidence artifacts:

| artifact family | path |
|---|---|
| odds/ruleset | `w42/odds_ruleset_claim_validation/` |
| bidding risk | `w42/bidding_risk_budget_claim_validation/` |
| doubles/no-trump | `w42/doubles_no_trump_claim_validation/` |
| scoring objective drift | `w42/scoring_objective_drift_claim_validation/` |
| 84 bidder/defender | `w42/eighty_four_claim_validation/` |
| partner support | `w42/partner_support_claim_validation/` |
| setter defense | `w42/setter_defense_claim_validation/` |
| Gus corpus tactical deep dive | `w42/gus_corpus_claim_deep_dive/` |
| bidder sequencing | `w42/bidder_sequencing_claim_validation/` |

Wiki inputs included [[w42]], [[w42-final-empirical-strategy-report]],
[[winning42-ch16-statistical-odds]], [[w42-claim-ledger]], and the relevant w42
claim-validation report pages.

## Methods

The ledger generator reads the existing JSON/CSV validation outputs, preserves
their conservative statuses, and writes a flat CSV plus summary and manifest
JSON files.

Evidence modes:

- exact enumeration over `C(28, 7)` hands or hand/declaration pairs;
- deterministic ruleset and scoring transforms;
- exact assignment/hypergeometric ownership checks;
- existing proxy regret/model-bucket reports;
- direct role-gated corpus contrasts from Gus v2 joint-world records;
- one small in-ledger arithmetic check for the Chapter 9 five-doubles claim:
  with two missing doubles distributed over 21 unknown slots, at least one
  opponent is void in doubles in `161 / 210 = 76.667%` of assignments.

The status vocabulary is exactly the [[w42-claim-ledger]] vocabulary:
`supported`, `contradicted`, `context-limited`, `underpowered`, and
`not-yet-tested`. Exact substrates can be `supported`; tactical advice stays
`underpowered`, `context-limited`, or `not-yet-tested` unless the artifact
directly tests the tactic.

## Claim Table Summary

The generated ledger has 64 rows.

| status | count |
|---|---:|
| supported | 23 |
| contradicted | 2 |
| context-limited | 12 |
| underpowered | 21 |
| not-yet-tested | 6 |

High-signal rows:

| claim area | ledger reading |
|---|---|
| hand and void odds | `supported`: `C(28,7) = 1,184,040`; void frequencies are 42.314%, 46.982%, 10.349%, and 0.355% for 0-3 void suits. |
| double-count odds | `supported`: exact double counts match the source table within rounding, including the one-hand seven-double extreme. |
| four-trump odds | `supported` for 10/27 and 14/27 assignment counts; `underpowered` for the boss-first policy recommendation. |
| trump counts | `supported` for fixed candidate trump-count priors; `underpowered` for treating three-plus trumps as a sufficient bidding rule. |
| count exposure | `supported` for duplicate-exposure arithmetic; `context-limited` for strong-trump/bad-off risk surfaces; `underpowered` for the 12-point bid threshold as a strategy rule. |
| doubles/no-trump | ruleset predicates are `supported`; declaration choice and regime-switch claims remain `underpowered` or `context-limited`. |
| scoring | deterministic marks-vs-points transforms are `supported`; skill-signal and timed-advancement claims need policy/tournament data. |
| 84 stoppers | named missing-double ownership is `context-limited`; applying the two-to-one prior when either of two matching doubles can set is `contradicted` by the 90% exact ownership result. |
| 84 weapons | static one-to-four last-trick weapon inventory is `context-limited`; same-suit pair defense is `underpowered`; abandonment and throwaway priorities are `not-yet-tested`. |
| setter pounce | `supported` on the operationalized Gus v2 pip-declaration slice: pounce-count actions beat same-decision alternatives by about +4.1 Q; reckless count into the bidder is a supported negative-control warning. |
| partner count donation | safe donation under current bidder-team control is `context-limited` with a small paired signal; unsafe count into defense control is `supported` as a negative-control warning. |
| remaining partner/setter/bidder proxies | mostly `underpowered`; the effective-double partner-support proxy is `contradicted` on the available slice. |

## Caveats

- Static enumeration proves arithmetic and prevalence, not optimal bidding or
  play.
- Hidden ownership appears only in report-time odds/evaluation labels; it is not
  a live-agent feature.
- Several proxy reports use existing Gus/w42 eval slices and broad tags. Their
  statuses should not be widened without direct detectors or paired rollouts.
  The Gus corpus tactical deep dive is the first direct-detector update for a
  narrow subset, not a broad promotion of all tactical advice.
- Book preview OCR is used as provenance, not as a source for long quotation.
- No central ledger page was rewritten; this is a phase-2 local artifact ready
  for the parent integrator to link.

## Provenance

| field | value |
|---|---|
| bead | `t42-5m82.1` |
| generator | `w42/statistics_claims_ledger/build_statistics_claims_ledger.py` |
| primary command | `python w42/statistics_claims_ledger/build_statistics_claims_ledger.py` |
| verification command | `python w42/statistics_claims_ledger/build_statistics_claims_ledger.py --check` |
| current commit at generation | `343a9f4c45244889ea9c1eaa8edacde7e2a69920` |
| random seeds | not applicable for this synthesis; upstream proxy reports preserve their own seeds |
| configs | not applicable |
| data inputs | wiki/source pages and w42 artifact directories listed above |
| W&B additions | `https://wandb.ai/jasonyandell-forge42/w42/runs/zm3jdrnj` for the Gus corpus tactical deep dive |

## Artifact Paths

| artifact | path |
|---|---|
| claims table | `w42/statistics_claims_ledger/claims.csv` |
| summary | `w42/statistics_claims_ledger/summary.json` |
| manifest | `w42/statistics_claims_ledger/manifest.json` |
| generator/checker | `w42/statistics_claims_ledger/build_statistics_claims_ledger.py` |

## W&B / HF Note

No new W&B or HuggingFace artifact was created by the ledger synthesis itself.
Ledger rows preserve source W&B links where upstream validation runs logged one,
including the Gus corpus tactical deep dive, doubles/no-trump run,
scoring-objective run, and corrected 84 run. HF remains not applicable.

## Claim-Ledger Impact

This page and `w42/statistics_claims_ledger/` now track the phase-2 statistics
claims ledger as an active local artifact. The Gus corpus tactical update moves
only narrow operationalized rows; it does not rewrite the central
[[w42-claim-ledger]] schema page.

[[w42-phase2-claim-analysis-matrix]] now routes every current ledger row to a
testability class, required fields, leakage boundary, sample/power need, target
wiki page, and next bead. That matrix is planning metadata only; it does not
move claim statuses by itself.

[[w42-tactical-claim-replication]] confirms the direct tactical rows using a
full legal-action export and harness pass, but does not broaden the statuses
beyond the operationalized Gus v2 slice already recorded here.

Deferred parent integration: decide whether any subset of these rows should
become the canonical central machine-readable claim ledger.

## Links

[[w42]] | [[w42-claim-ledger]] | [[w42-final-empirical-strategy-report]] |
[[winning42-ch16-statistical-odds]] | [[w42-odds-ruleset-claim-validation]] |
[[w42-bidding-risk-budget-claim-validation]] |
[[w42-doubles-no-trump-claim-validation]] |
[[w42-scoring-objective-drift-claim-validation]] |
[[w42-84-claim-validation]] |
[[w42-gus-corpus-tactical-claim-deep-dive]]
