---
title: w42 Gus Corpus Tactical Claim Deep Dive
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] now has a first powered tactical claim deep dive over the existing
[[gus]] v2 joint-world corpus. It targets two previously underpowered tactical
families from [[winning42-ch04-partner-support]] and
[[winning42-ch05-setter-defense]]:

- setter pounce / count-before-certainty;
- safe versus unsafe partner count donation.

The run uses existing `gus/data/corpus_v2_train_*_d0-9.pt` files, not new seed
generation. Those files already retain `q_per_world`, `world_hands`,
`bid_value`, per-seat legal masks, and all ten declarations at N=200 sampled
worlds per decision. This makes them sufficient for role-gated direct labels and
paired E[Q] contrasts.

[[w42-tactical-claim-replication]] reruns the same source corpus as bead
`t42-0b4l.3`, exports the full 75,079 legal-action rows for the reusable
[[w42-phase2-claim-analysis-harness]], and adds declaration/seat/trick/control
slice tables. It confirms the paired contrast numbers here while making the
row-level substrate reusable for later tactical beads.

This is not a model-training run and does not use hidden ownership as a live
feature. It is a report run over saved oracle distributions.

## Data Slice

| field | value |
|---|---:|
| input files | 10 |
| seed range | 0-99 |
| declarations | all ten declarations in source files; tactical labels restricted to pip declarations 0-6 |
| sampled worlds per decision | 200 |
| decision rows | 28,000 |
| legal-action rows | 75,079 |
| claim-labeled action rows | 3,903 |
| claim-labeled decision rows | 3,428 |

Artifact directory: `w42/gus_corpus_claim_deep_dive/`.

W&B run:
`https://wandb.ai/jasonyandell-forge42/w42/runs/zm3jdrnj`.

The W&B run logs a progress series over `progress/files_processed`, including
action rows, decision rows, setter-pounce counts, reckless-count counts, and
partner donation counts. It is a dashboard-style report run, not a single final
metric frame.

## Labels

The analyzer reconstructs public trick context from recorded actions, then adds
role-gated action labels:

| label | operational definition |
|---|---|
| `ch05_setter_pounce_count_before_certainty` | defender plays count, offense is currently winning the trick, the defender can beat current winner, and later seats still remain |
| `ch05_setter_pounce_count` | defender plays count that can take a currently offense-won trick |
| `ch05_setter_pounce_count_sets_now` | pounce-count action would put defenders at or beyond the set threshold if it wins the trick |
| `ch05_reckless_count_to_bidder` | defender plays count while offense is winning and the candidate does not win the trick now |
| `ch04_partner_safe_count_donation_current_control` | bidder partner plays count while the bidder team is currently winning |
| `ch04_partner_unsafe_count_to_defense` | bidder partner plays count while defense is winning and the candidate does not beat current winner |

These are public/actor-hand labels. The E[Q] values, `q_per_world` distributions,
and sampled-world outcomes are offline evaluation labels only.

## Findings

The strongest movement is in [[winning42-ch05-setter-defense]]. The direct
pounce labels now have enough support on this slice to replace the earlier
proxy-only reading.

| claim label | actions | decisions | mean regret | threshold mass | best-mean rate |
|---|---:|---:|---:|---:|---:|
| pounce count before certainty | 321 | 317 | 0.999 | 0.893 | 71.3% |
| pounce count | 453 | 442 | 1.075 | 0.920 | 73.1% |
| pounce count sets now | 172 | 170 | 0.637 | 0.951 | 83.7% |
| reckless count to bidder | 1,987 | 1,785 | 7.213 | 0.467 | 29.9% |
| partner safe count donation | 603 | 528 | 1.708 | 0.500 | 61.7% |
| partner unsafe count donation | 860 | 776 | 6.586 | 0.069 | 33.5% |

Paired same-decision contrasts are clearer than unpaired action means:

| contrast | paired decisions | mean delta | 95% CI | threshold-mass delta |
|---|---:|---:|---|---:|
| setter pounce count vs other action | 359 | +4.102 | `[+3.144, +5.176]` | +0.049 |
| pounce count that sets now vs other action | 134 | +6.069 | `[+4.219, +8.088]` | +0.073 |
| reckless defender count vs non-reckless action | 1,355 | -8.372 | `[-8.975, -7.791]` | -0.119 |
| safe partner count vs other action | 400 | +0.683 | `[+0.147, +1.232]` | +0.007 |
| unsafe partner count vs non-unsafe action | 559 | -8.428 | `[-9.268, -7.613]` | -0.132 |

Interpretation:

- The book-shaped setter pounce concept is no longer merely designed or
  proxy-supported on this corpus slice. When the defender can take an
  offense-winning trick with count, the best pounce-count action beats the best
  same-decision alternative by about four Q points. When that count immediately
  reaches the set threshold, the advantage is about six Q points.
- Reckless count donation to the bidder side is strongly negative. This
  validates why the pounce gate matters: "play count aggressively" is wrong
  unless the trick/control condition is present.
- Partner safe donation has a smaller but positive paired signal. Unsafe partner
  count donation is strongly negative. This strengthens the Chapter 4 safety
  boundary but still needs a stronger "guaranteed win" detector before broad
  claim-ledger promotion.

## Claim-Ledger Reading

This page should move local interpretation, not every broad book claim.

| claim area | previous status | current reading |
|---|---|---|
| Ch05 pounce count before certainty | underpowered / direct fields missing | supported on this Gus v2 N=200 pip-declaration corpus slice for the operationalized pounce-count label |
| Ch05 reckless count into bidder | proxy-only negative | supported as a negative control on the same slice |
| Ch04 safe partner count donation | underpowered, directional proxy | context-limited directional support with paired signal; needs guaranteed-trick strength and later-seat risk labels |
| Ch04 unsafe partner count donation | underpowered, directional proxy | supported as a negative control on this slice |

The verdict is narrow. It does not prove all setter-defense advice, all partner
support advice, or human table play. It supports these operationalized
role/trick/action labels against forge E[Q] distributions in the saved Gus v2
generated corpus.

## Caveats

- The source corpus uses generated games and forge E[Q] labels, not human games.
- The tactical labels are restricted to pip declarations 0-6.
- The source files have fixed `bid_value=30`, so bid-margin and overbid claims
  remain untested.
- "Safe partner donation" currently means the bidder team is winning now; it is
  not yet a proof that later seats cannot overtake the trick.
- The analyzer compares legal actions within recorded decision states. It does
  not run sequence counterfactuals after choosing a different action.
- Hidden sampled-world tensors are used only as offline label substrate.

## Artifacts

| artifact | path |
|---|---|
| analyzer | `w42/gus_corpus_claim_deep_dive/analyze_gus_claims.py` |
| summary | `w42/gus_corpus_claim_deep_dive/summary.json` |
| claim metrics | `w42/gus_corpus_claim_deep_dive/claim_metrics.csv` |
| paired contrasts | `w42/gus_corpus_claim_deep_dive/paired_contrasts.csv` |
| declaration slices | `w42/gus_corpus_claim_deep_dive/slice_metrics_by_decl.csv` |
| examples | `w42/gus_corpus_claim_deep_dive/examples.json` |
| paired examples | `w42/gus_corpus_claim_deep_dive/paired_examples.json` |
| action rows | `w42/gus_corpus_claim_deep_dive/claim_action_rows.jsonl` |
| manifest | `w42/gus_corpus_claim_deep_dive/manifest.json` |

## Commands

```bash
python w42/gus_corpus_claim_deep_dive/analyze_gus_claims.py \
  --bootstrap-samples 5000 \
  --wandb-mode online \
  --wandb-group w42-gus-corpus-claim-deep-dive \
  --wandb-name w42-gus-corpus-claim-deep-dive-v0-stable

python -m py_compile w42/gus_corpus_claim_deep_dive/analyze_gus_claims.py
```

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[w42-phase2-setter-pounce-direct-label-probe]] |
[[w42-tactical-claim-replication]] |
[[w42-setter-defense-claim-validation]] |
[[w42-partner-support-claim-validation]] |
[[winning42-ch04-partner-support]] |
[[winning42-ch05-setter-defense]]
