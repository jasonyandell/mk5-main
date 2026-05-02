---
title: w42 Phase 2 Statistics Claims Ledger
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Question

[[w42]] phase 2 asks which statistical claims from Winning 42 are already
measurable, which are supported by exact arithmetic or deterministic rules, and
which remain strategy hypotheses needing direct fixtures.

This ledger treats the book as a hypothesis source, not truth. A correct odds
substrate does not validate a tactical recommendation unless the measured
predicate also matches the recommendation.

## Data And Method

The ledger is a synthesis artifact, not a new simulation run. It inventories
explicit and implied statistical claims from the book-facing W42 work and
classifies each claim by measurable predicate, evidence method, result status,
and next fixture.

Primary inputs:

| source | role |
|---|---|
| [[winning42-ch16-statistical-odds]] | hand odds, void frequencies, double-count priors, four-trump assignment claims |
| [[w42-odds-ruleset-claim-validation]] | exact hand/ruleset enumeration evidence |
| [[w42-bidding-risk-budget-claim-validation]] | static count-exposure and bid-risk arithmetic |
| [[w42-partner-support-claim-validation]] | partner donation and stopper proxy slices |
| [[w42-setter-defense-claim-validation]] | setter pounce and count-pressure proxy slices |
| [[w42-doubles-no-trump-claim-validation]] | doubles-as-trump and no-trump rules/static regime slices |
| [[w42-84-claim-validation]] | 84 weapon availability and ownership inventory |
| [[w42-scoring-objective-drift-claim-validation]] | deterministic scoring-objective transforms |
| `scratch/winning42/strategy_measurement_breakdown.md` | additional inventory for belief, count discipline, and style claims |

Classification uses five testability classes:

| class | meaning |
|---|---|
| `exact-enumerable` | finite hand, assignment, ruleset, or terminal-score space can be exhaustively counted |
| `simulation-estimable` | claim depends on play history, hidden ownership, or future action value but can be estimated with generated states or oracle rollouts |
| `policy-dependent` | claim needs auction policy, repeated-player behavior, tournament objective, or population assumptions |
| `ambiguous` | source wording is not yet precise enough to define a predicate |
| `out-of-scope` | not a statistical or legal-public-state claim for this phase |

No ledger row used `ambiguous` or `out-of-scope`; unclear tactical claims were
kept as not-yet-tested or policy-dependent when a plausible predicate exists.

## Artifact Summary

| field | value |
|---|---:|
| total claims | 37 |
| supported | 13 |
| context-limited | 5 |
| underpowered | 9 |
| not-yet-tested | 9 |
| contradicted | 1 |
| exact-enumerable | 19 |
| simulation-estimable | 13 |
| policy-dependent | 5 |

Artifact paths:

| artifact | path |
|---|---|
| claim table | `w42/statistics_claims_ledger/claims.csv` |
| JSON claim table | `w42/statistics_claims_ledger/claims.json` |
| summary | `w42/statistics_claims_ledger/summary.json` |
| manifest | `w42/statistics_claims_ledger/manifest.json` |
| builder | `w42/statistics_claims_ledger/build_statistics_claims_ledger.py` |

## Claim Table Highlights

| family | claim | method | result/status | next fixture |
|---|---|---|---|---|
| hand odds | Seven-card double-six hand count is 1,184,040. | exact enumeration / combinatorial identity | supported: 1,184,040 | generated-corpus deal sanity check |
| void frequencies | Suit coverage / void odds are about 41/48/10/1. | exact enumeration | supported: 42.314 / 46.982 / 10.349 / 0.355 | corpus chi-square check |
| doubles odds | Double-count odds match the book table. | exact enumeration | supported: 0 through 7 double priors match within rounding | reusable odds fixture |
| trump counts | Four-trump double-first assignment counts are 10/27 and 14/27. | exact 27-case assignment enumeration | supported for risk prior | paired forge rollout by count and bid margin |
| trump counts | Count in trump can justify the near-50/50 four-trump risk. | no value rollout yet | not-yet-tested | controlled four-trump threshold generator |
| count exposure | Duplicate exposed-count accounting is common. | exact hand x trump-candidate enumeration | supported as detector arithmetic: 86.354% of candidate evaluations | unique exposed-count bidding fixture |
| count exposure | Strong trump shape can still exceed the static risk budget. | exact static enumeration | supported as detector arithmetic: 2.927% of candidates | auction-aware E[Q] counterfactual |
| bidding | Bid only enough to win. | no auction counterfactual yet | not-yet-tested | fixed-hand bid-margin rollout |
| partner support | Safe partner count donation. | proxy oracle-regret slice | underpowered: directional, paired n=0 | forced donation / guarantee-strength fixture |
| stopper ownership | Effective double / virtual boss tile. | proxy oracle-regret slice | contradicted on coarse proxy: +2.299 paired regret, CI [0.161, 4.364] | exact effective-walker detector |
| setter defense | Pounce count before certainty. | proxy oracle-regret slice | underpowered; ungated count donation is high regret | direct setter-pounce window detector |
| doubles/no-trump | Doubles-as-trump removes doubles from native suits. | deterministic ruleset fixtures | supported | rules regression fixture |
| doubles/no-trump | Four-plus doubles is only a context gate. | exact static hand slice | context-limited | declaration rollout |
| 84 | Named missing matching double is two-to-one to be on opponent team. | exact static ownership enumeration | context-limited: 14/21 = 66.667% | one-double versus either-double stopper fixtures |
| 84 | Defender last-trick weapon pool is usually one to four weapons. | exact static ownership enumeration | context-limited: 56.667% to 95.320% by pool size | final-two-trick 84 defense rollout |
| scoring | Marks change the objective to hand-level make/set. | deterministic terminal transform | supported | scoring regression fixture |
| scoring | Marks can speed tournaments. | terminal arithmetic only | context-limited | policy-population tournament simulation |
| belief/count discipline | Public voids and dot counting should guide inference. | design inventory only | not-yet-tested | void-event and count-status trace fixtures |
| style | Player style affects overbid restraint and low-game willingness. | concept matrix only | not-yet-tested | repeated-player or partner-shuffle fixture |

The full row-level ledger is in `claims.csv` and `claims.json`.

## Supported Substrate

The supported rows are mostly arithmetic, ruleset, scoring-accounting, and static
prevalence claims:

- double-six hand count, void/suit coverage, double-count priors, modal hand;
- four-trump missing-high-trump assignment priors;
- deterministic doubles/no-trump regime membership and follow-suit behavior;
- duplicate count exposure and strong-trump/static-risk detector arithmetic;
- terminal mark-scoring objective transforms.

These are good fixtures for W42 because they give stable denominators and
regression checks. They should not be rewritten as "the book's play is right."

## Caveats

- Exact enumeration supports only the predicate it enumerates.
- Static inventory does not prove make/set rate, regret, or bid quality.
- Proxy oracle-regret slices are useful for prioritization but often lack role,
  forcedness, hidden ownership, bid margin, or direct detector gates.
- Hidden ownership may be used offline for evaluation and attribution; live
  features must stay legal-public-state or learned-belief only.
- The 84 and doubles/no-trump rows are especially prone to confusing weapon
  availability with policy value.
- The single contradicted row is a coarse proxy result for virtual boss tiles,
  not a final contradiction of every effective-stopper idea.

## Recommended Next Fixtures

| priority | fixture | claims unlocked |
|---|---|---|
| P0 | four-trump threshold generator with paired boss-first / low-trump-first E[Q] rollouts | count-in-trump exception; four-trump tactical value |
| P0 | direct setter-pounce window labels: bidder identity, off suit, points needed to set, and legal alternatives | pounce count before certainty; extra count to set; high-bid off pounce |
| P0 | 84 final-two-trick proof/checker with hidden stopper attribution | one-to-four weapons; same-suit pair ahead of double; abandon dead assets |
| P1 | conditional partner-double prior by bidder hand and declaration | partner double-help prior; multi-off bidding risk |
| P1 | no-trump versus doubles-as-trump paired declaration rollout | no-trump-over-doubles choice; low-double sacrifice; no-trump defense preservation |
| P1 | score-aware 42-vs-84 counterfactual under marks and points | score gate; tournament objective drift |
| P2 | void-event belief calibration and count-status trace-faithfulness audit | void owner inference; dot-count discipline; Burl explanation faithfulness |
| P2 | repeated-player / partner-shuffle population fixture | style and partnership ecology claims |

## Provenance

| field | value |
|---|---|
| bead | `t42-5m82.1` |
| branch | `w42/phase2-statistics` |
| synthesis commit at build time | `343a9f4c45244889ea9c1eaa8edacde7e2a69920` |
| commands | `git status --short --branch`; `sed -n '1,220p' wiki/AGENTS.md`; `sed -n '1,220p' wiki/entities/w42.md`; `sed -n '1,220p' wiki/experiments/w42-final-empirical-strategy-report.md`; `sed -n '1,240p' wiki/experiments/winning42-ch16-statistical-odds.md`; `rg --files w42 scratch/winning42 wiki/experiments \| rg 'claim_validation\|claim-validation\|winning42.with_fig\|w42-.*(odds\|doubles\|bidding\|84\|scoring\|partner\|setter\|ruleset)'`; `python w42/statistics_claims_ledger/build_statistics_claims_ledger.py` |
| validation commands | `python -m json.tool w42/statistics_claims_ledger/claims.json >/dev/null`; `python -m json.tool w42/statistics_claims_ledger/summary.json >/dev/null`; `python -m json.tool w42/statistics_claims_ledger/manifest.json >/dev/null`; CSV/summary row-count assertion |
| random seeds | not applicable; no new simulation |
| W&B links | not applicable for this synthesis; source reports retain their W&B provenance where relevant |
| HF links | not applicable |

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[winning42-ch16-statistical-odds]] | [[w42-odds-ruleset-claim-validation]] |
[[w42-bidding-risk-budget-claim-validation]] |
[[w42-partner-support-claim-validation]] |
[[w42-setter-defense-claim-validation]] |
[[w42-doubles-no-trump-claim-validation]] |
[[w42-84-claim-validation]] |
[[w42-scoring-objective-drift-claim-validation]]
