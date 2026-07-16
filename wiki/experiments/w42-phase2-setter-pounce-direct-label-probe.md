---
title: w42 Phase 2 Setter Pounce Direct Label Probe
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-15
status: superseded
---

**Superseded** by [[w42-gus-corpus-tactical-claim-deep-dive]] (itself
superseded by [[w42-tactical-claim-replication]]) — three supersession hops
from the current frontier.

## Summary

[[w42]] has a first-pass direct-label spec for setter pounce and count-to-set
windows. The original bead did not move any book claim status because it only
inspected the small eval surface and found missing direct fields. The follow-up
[[w42-gus-corpus-tactical-claim-deep-dive]] uses the richer Gus v2 joint-world
corpus and does run direct role/trick/action labels for pounce count,
count-before-certainty, and reckless count into the bidder.

That follow-up changes the current frontier: setter pounce is no longer only a
label spec on the Gus v2 generated-corpus slice. The direct label remains narrow,
but it now has paired E[Q] evidence.

Artifacts:

- `w42/setter_pounce_direct_label_probe/label_spec.json`
- `w42/setter_pounce_direct_label_probe/fixtures.jsonl`
- `w42/setter_pounce_direct_label_probe/required_fields.csv`
- `w42/setter_pounce_direct_label_probe/anti_leakage_checks.csv`
- `w42/setter_pounce_direct_label_probe/tiny_report_slice.csv`
- `w42/setter_pounce_direct_label_probe/summary.json`
- `w42/setter_pounce_direct_label_probe/manifest.json`

The label spec still sharpens the detector definition from
[[w42-setter-defense-claim-validation]]: setter pounce should be measured as a
public-state window first, with E[Q] distributions and hidden-world effects
attached only as offline evaluation labels.

## Follow-Up Evidence

[[w42-gus-corpus-tactical-claim-deep-dive]] processes 28,000 decisions and
75,079 legal actions from the Gus v2 all-declaration corpus. On pip
declarations, the direct pounce labels report:

| contrast | paired decisions | mean delta | 95% CI | threshold-mass delta |
|---|---:|---:|---|---:|
| pounce count vs other action | 359 | +4.102 | `[+3.144, +5.176]` | +0.049 |
| pounce count that sets now vs other action | 134 | +6.069 | `[+4.219, +8.088]` | +0.073 |
| reckless defender count vs non-reckless action | 1,355 | -8.372 | `[-8.975, -7.791]` | -0.119 |

This supports the operationalized pounce-count label and its negative control on
the Gus v2 N=200 generated-corpus slice. It does not prove every Chapter 5
setter-defense recommendation, and it still does not cover real bid-margin or
high-bid overbid contexts.

## Question

Can [[w42]] define direct, public-state-safe labels for setter pounce and
count-to-set windows, and can the existing artifacts already support a small
direct report?

## Method

The bead read the phase-2 decision, the prior setter-defense proxy validation,
the v1 detector map, and the final empirical strategy report. It inspected the
existing Gus eval corpus item surface at
[corpus_eval_20.pt](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus/blob/main/corpus_eval_20.pt) ([[huggingface-assets]]).

The current corpus inspection found 560 rows with these available item fields:
`tokens`, `attention_mask`, `belief_target`, `belief_mask`, `world_assignment`,
`q_per_world`, `e_q`, `action_taken`, `legal_mask`, `decision_idx`, `player`,
`voids`, `strategy_features`, and `strategy_action_features`.

That is enough to explain why the prior report was proxy-only, but not enough to
run the direct pounce detector. Missing named fields include `bidder_seat`,
`bid_amount`, declaration/contract context, pre-action team points, current-trick
led suit and public winning certainty, and E[Q] PDF or sampled-world outcome
distributions by action.

## Direct Label Spec

The live/public-state label set is:

| label | level | purpose |
|---|---|---|
| `is_setter_decision` | decision | acting player is on the defending team |
| `contract_context` | decision | bid, declaration, ruleset, and pre-action score fields are present |
| `count_to_set_window` | decision | candidate count can cross or materially approach the set threshold |
| `bidder_off_window` | decision | public trick state exposes bidder team on an off-suit trick |
| `count_before_certainty` | action | setter can play count before the trick winner is public-certain |
| `setter_pounce_window` | decision | setter role, bidder-off gate, count-to-set gate, and legal count coexist |
| `setter_pounce_action` | action | legal count action inside a setter-pounce window |

The offline-only labels are:

| label | source | purpose |
|---|---|---|
| `threshold_mass_delta` | E[Q] PDFs or sampled-world outcomes | action-level change in make/set threshold mass |
| `disaster_tail_mitigation_delta` | E[Q] PDFs or sampled-world outcomes | action-level lower-tail or CVaR-style branch shift |

These offline labels are legal for reports, training targets, and diagnostics.
They are not live strategy features.

## Fixture Examples

`fixtures.jsonl` contains four deterministic examples:

| fixture | expected role |
|---|---|
| `public_positive_count_before_certainty` | positive public setter-pounce action |
| `negative_bidder_team_actor` | count exists but actor is on bidder team |
| `negative_no_threshold_pressure` | setter count exists but no set-threshold pressure remains |
| `offline_hidden_tail_positive` | public pounce positive with hidden tail labels attached offline only |

The fixtures are intentionally small and synthetic. They test the label boundary,
not engine legality or rollout quality.

## Tiny Report Slice

The tiny slice reuses the prior proxy report only as readiness evidence:

| claim id | prior proxy evidence | direct label status |
|---|---|---|
| `ch05-pounce-count-before-certainty` | paired n=52, proxy count-to-opponent regret delta `+5.111` | direct fields missing |
| `ch05-extra-count-to-set` | paired n=4, ten-count proxy regret delta `+4.353` | direct threshold fields missing |
| `ch12-setter-pounce-high-bid-off` | paired n=0 for broad count-pressure proxy | high-bid off and distribution fields missing |

The positive proxy deltas remain non-verdicts. They mostly show that naive count
dumping is bad without the pounce gate.

## Anti-Leakage Boundary

Live labels may read only public state, current hand, legal actions, public
contract metadata, public voids, current trick state, and pre-action score. They
must not read hidden owner truth, future trick outcomes, terminal hand totals,
forge E[Q], E[Q] PDFs, sampled-world branches, or completed-hand set
attribution.

The key distinction is structural versus outcome labels:

- `setter_pounce_window` and `setter_pounce_action` are structural public labels.
- `threshold_mass_delta` and `disaster_tail_mitigation_delta` are offline
  outcome/distribution labels.
- No broad claim support follows from the structural label alone.

## Required Phase-2 Data

The next run needs a decision table with:

| field family | required examples |
|---|---|
| role and contract | `current_player_seat`, `bidder_seat`, `bid_amount`, `declaration`, `ruleset_id` |
| pre-action score | bidder and defender team points before the candidate action |
| trick state | led suit, trick position, current public winner, current trick count |
| public inference | public voids by seat/suit, public winner certainty |
| action state | legal action set, candidate tile, candidate count points |
| offline distribution | set probability by action, q10/CVaR by action, hidden-domino attribution |

If phase-2 seat/position or distribution-aware outputs appear later, this page
should be revisited and the direct report should replace the readiness slice.

## Next Command Plan

```bash
python w42/setter_pounce_direct_label_probe/build_probe_artifacts.py

# Future, once a phase-2 decision table exists:
python w42/setter_pounce_direct_label_probe/analyze_direct_pounce.py \
  --decisions w42/<phase2_decision_table>.jsonl \
  --output-dir w42/setter_pounce_direct_label_probe
```

The future analyzer should emit direct-label coverage, paired pounce-action
regret, threshold-mass deltas, disaster-tail deltas, and claim-specific bucket
metrics. Training or W&B series should wait until that direct table exists.

## Provenance

| field | value |
|---|---|
| bead | `t42-5m82.5` |
| evidence mode | direct-label spec, fixture package, corpus field inspection, readiness slice |
| generated by | `python w42/setter_pounce_direct_label_probe/build_probe_artifacts.py` |
| source corpora | [corpus_eval_20.pt](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus/blob/main/corpus_eval_20.pt) field inspection |
| W&B links | not applicable |
| HF links | not applicable |
| claim-ledger impact | no central ledger change |

## Links

[[w42]] | [[w42-next-model-decision]] |
[[w42-setter-defense-claim-validation]] |
[[w42-final-empirical-strategy-report]] |
[[w42-strategy-tags-v1-map]]
