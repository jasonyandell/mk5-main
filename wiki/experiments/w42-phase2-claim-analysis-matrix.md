---
title: w42 Phase 2 Claim Analysis Matrix
kind: experiment
first_seen: local-2026-05-02
last_updated: afd4802
status: complete
---

## Summary

[[w42]] now has a claim-analysis matrix for routing the 64-row
[[w42-phase2-statistics-claims-ledger]] into empirical work packages. The matrix
does not change any claim status. It classifies each book/ledger row by the kind
of evidence needed next, the data fields required, leakage risks, likely
blockers, sample/power needs, target wiki archive page, and next bead.

This is the bridge from "the book has claims" to "the lab knows what each claim
would need before it can move."

## Artifacts

| artifact | path |
|---|---|
| generator | `w42/claim_analysis_matrix/build_claim_analysis_matrix.py` |
| claim matrix | `w42/claim_analysis_matrix/claim_analysis_matrix.csv` |
| family rollup | `w42/claim_analysis_matrix/family_rollup.csv` |
| powered-test queue | `w42/claim_analysis_matrix/ready_powered_tests.csv` |
| summary | `w42/claim_analysis_matrix/summary.json` |
| manifest | `w42/claim_analysis_matrix/manifest.json` |

The matrix is generated from `w42/statistics_claims_ledger/claims.csv` rather
than hand-edited. Future ledger rows should flow through the same generator so
classification drift stays visible.

## Coverage

The generated matrix covers all 64 current ledger rows across 17 families.

| current status | claims |
|---|---:|
| supported | 23 |
| underpowered | 21 |
| context-limited | 12 |
| not-yet-tested | 6 |
| contradicted | 2 |

Readiness routing:

| ready state | claims | reading |
|---|---:|---|
| fixture only | 19 | exact arithmetic, rules, or deterministic scoring substrates |
| needs generation | 9 | mostly auction/bid-margin or count-risk strategy claims |
| needs 84 generation | 10 | dynamic 84 stopper, weapon, preservation, and abandonment claims |
| needs detector implementation | 9 | partner/support and setter claims still stuck at proxy labels |
| needs sequence counterfactuals | 5 | bidder sequencing, reentry, off timing, and laydown claims |
| needs regime generation | 4 | doubles-trump/no-trump strategy claims beyond ruleset fixtures |
| needs simulation design | 3 | scoring skill, tournament speed, and timed advancement claims |
| needs label refinement | 2 | direct corpus labels exist but are too broad for promotion |
| yes after harness | 3 | direct corpus claims ready for powered replication once the reusable harness exists |

## Testability Classes

The matrix uses conservative testability classes:

| class | claims | meaning |
|---|---:|---|
| deterministic static fixture | 4 | exact hand/void/double arithmetic |
| deterministic ruleset fixture | 5 | legal regime predicates for doubles/no-trump |
| deterministic objective fixture | 6 | marks/points scoring transforms |
| static substrate fixture plus auction follow-up | 4 | exact prior is real, policy claim still needs bidding data |
| auction or bid-margin data needed | 9 | fixed-bid artifacts cannot test the claim |
| static regime substrate needs paired regime rollout | 4 | doubles/no-trump odds need declaration or play counterfactuals |
| policy population simulation needed | 3 | tournament/skill/speed claims need match simulation |
| static 84 substrate needs dynamic rollout | 9 | 84 shape/ownership substrate is not play timing evidence |
| static substrate contradiction then dynamic 84 follow-up | 1 | exact ownership contradicts one overgeneralized stopper prior |
| direct action contrast needs stronger labels | 2 | current corpus contrast is directional but label is not sharp enough |
| direct detector or paired contrast needed | 9 | proxy reports need direct detectors |
| direct action contrast ready for replication | 3 | narrow operationalized labels are ready for powered follow-up |
| sequence counterfactual or model probe needed | 5 | model buckets are not enough for bidder line advice |

## Routing Findings

The blankest high-value substrate is bid margin. Current branch-atlas and Gus v2
artifacts carry `bid_value`, but the durable generated slices are still fixed at
30 and lack auction pressure, runner-up bid, overbid amount, and score-gated
counterfactuals. That blocks "bid only enough," natural bid buckets, four-trump
opening policy, and many count-risk claims from becoming real strategy tests.

The crispest dynamic gap is 84. Existing 84 work has strong static shape and
ownership tables, including a contradicted overgeneralization about either
matching double preserving a two-to-one prior. It still needs legal-action states
where a stopper can be preserved, spent, broken, or abandoned, with forced versus
voluntary distinctions and distribution/tail labels.

The most immediately reusable powered corpus is still the Gus v2 joint-world
slice. It already supports narrow setter-pounce and unsafe count-donation
contrasts in [[w42-gus-corpus-tactical-claim-deep-dive]], but the next pass
needs a reusable harness, guaranteed-control labels, later-seat overtake risk,
and bid-margin/high-bid slices before broad Chapter 4 or Chapter 5 promotion.

The hidden-threat branch atlas is a diagnostic amplifier, not a direct live
feature. It can say which unseen domino/holder facts drive shelves, tails, and
lumps in saved worlds. That makes belief impact measurable, but hidden ownership
must remain an offline label unless converted into learned public-state beliefs.

## Next Beads

| bead | routed claims | emphasis |
|---|---:|---|
| `t42-0b4l.2` | shared dependency | reusable claim analyzer harness, now [[w42-phase2-claim-analysis-harness]] |
| `t42-0b4l.3` | 14 partner/setter rows | pounce, donation, forcedness, later-seat risk |
| `t42-0b4l.4` | cross-cutting | hidden-threat belief impact and distribution-shaped EV omissions |
| `t42-0b4l.5` | 13 bidding/count/four-trump rows | auction data, bid margin, risk budget |
| `t42-0b4l.6` | role/sequence rows | seat/position and bidder line counterfactuals |
| `t42-0b4l.7` | 10 84/stopper rows | dynamic stopper and weapon-preservation tests |
| `t42-0b4l.8` | 9 doubles/no-trump rows | paired regime generation after ruleset fixtures |
| `t42-0b4l.9` | model-probe follow-up | claim-tag ablations after direct labels exist |
| `t42-0b4l.10` | synthesis | final reconciliation and next epic |

## Leakage Boundary

The matrix keeps the w42 boundary explicit:

- online-safe inputs include public auction/declaration, public trick history,
  legal actions, actor hand, score, public count/trump depletion, and beliefs
  learned from legal public evidence;
- offline-only labels include E[Q], E[Q] PDFs, `q_per_world`, `world_hands`,
  hidden holder truth, future outcomes, final set attribution, and
  threshold/tail deltas.

Hidden truth can label a report or train a belief target. It cannot become a
live policy feature.

## Commands

```bash
python w42/claim_analysis_matrix/build_claim_analysis_matrix.py

python - <<'PY'
import csv, json
rows = list(csv.DictReader(open("w42/claim_analysis_matrix/claim_analysis_matrix.csv", newline="")))
assert len(rows) == 64
json.load(open("w42/claim_analysis_matrix/summary.json"))
json.load(open("w42/claim_analysis_matrix/manifest.json"))
print("claim analysis matrix validated", len(rows))
PY
```

## Claim-Ledger Impact

No claim ledger status moved. This page is a research-design and routing
artifact. Claims should move only after the downstream beads produce direct
evidence with stated operational definitions, data slice, CIs, examples, W&B
links where applicable, and conservative caveats.

## Links

[[w42]] | [[w42-phase2-statistics-claims-ledger]] |
[[w42-gus-corpus-tactical-claim-deep-dive]] |
[[w42-phase2-seat-position-strategy-map]] |
[[w42-phase2-hidden-domino-threat-attribution]] |
[[w42-phase2-distribution-aware-ev-report]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- `summary.json` routes 9 rows to `t42-0b4l.10` plus a future tournament-simulation bead; the Next Beads table lists `.10` as "synthesis" without a count — 9 simulation-class rows park there pending a tournament-sim bead.
