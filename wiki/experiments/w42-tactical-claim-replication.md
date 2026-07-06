---
title: w42 Tactical Claim Replication
kind: experiment
first_seen: local-2026-05-02
last_updated: afd4802
status: superseded
---

**Superseded** by [[w42-phase4-sequence-handshape-tests]], which reads
directly from this page's output artifact (`all_action_rows.jsonl`) to
sharpen the same Ch3/4/5 claims. This page's own "harness migration"
follow-up (moving corpus replay into shared `w42/claim_analysis/` adapters)
appears to have landed by June, per the `w42/claim_analysis/` directory's
mtime — asserted, unverified beyond directory presence.

## Summary

[[w42]] now has a powered tactical replication artifact for `t42-0b4l.3`. It
reruns the existing Gus v2 all-declaration joint-world corpus used by
[[w42-gus-corpus-tactical-claim-deep-dive]], exports full legal-action rows, and
routes those rows through [[w42-phase2-claim-analysis-harness]].

The replication confirms the original narrow result:

- setter pounce count remains strongly positive on this generated corpus slice;
- pounce count that sets now is stronger still;
- reckless defender count into the bidder side remains strongly negative;
- safe partner count donation has a small positive paired signal;
- unsafe partner count donation remains strongly negative.

The result does not broaden the claim ledger beyond the existing conservative
reading. It adds reproducible full-row artifacts, W&B progress series, and slice
tables needed by the next tactical beads.

## Data Slice

| field | value |
|---|---:|
| source files | 10 |
| decisions | 28,000 |
| legal action rows | 75,079 |
| claim-labeled action rows | 3,903 |
| claim-labeled decision rows | 3,428 |
| sampled worlds per decision | 200 |

Artifact directory: `w42/tactical_claim_replication/`.

W&B run:
`https://wandb.ai/jasonyandell-forge42/w42/runs/jv9luhgp`.

The W&B run logs ten progress points over `progress/files_processed`, including
action rows, decision rows, claim-action rows, and tactical label counts.

## Outputs

| artifact | path |
|---|---|
| runner | `w42/tactical_claim_replication/run_tactical_replication.py` |
| summary | `w42/tactical_claim_replication/summary.json` |
| claim metrics | `w42/tactical_claim_replication/claim_metrics.csv` |
| paired contrasts | `w42/tactical_claim_replication/paired_contrasts.csv` |
| slice metrics | `w42/tactical_claim_replication/slice_metrics.csv` |
| paired contrast slices | `w42/tactical_claim_replication/paired_contrasts_by_slice.csv` |
| examples | `w42/tactical_claim_replication/examples.json` |
| manifest | `w42/tactical_claim_replication/manifest.json` |
| full legal-action rows | `w42/tactical_claim_replication/all_action_rows.jsonl` |
| harness outputs | `w42/tactical_claim_replication/harness/` |

The full legal-action row export is intentionally useful for follow-up harness
work, but it is large enough that a later commit should decide whether to keep
it as local generated data, compress it, or regenerate it from the script.

## Claim Metrics

| label | actions | decisions | mean regret | threshold mass | best-mean rate |
|---|---:|---:|---:|---:|---:|
| pounce count before certainty | 321 | 317 | 0.999 | 0.893 | 71.3% |
| pounce count | 453 | 442 | 1.075 | 0.920 | 73.1% |
| pounce count sets now | 172 | 170 | 0.637 | 0.951 | 83.7% |
| reckless count to bidder | 1,987 | 1,785 | 7.213 | 0.467 | 29.9% |
| partner safe count donation | 603 | 528 | 1.708 | 0.500 | 61.7% |
| partner unsafe count donation | 860 | 776 | 6.586 | 0.069 | 33.5% |

## Paired Contrasts

| contrast | paired decisions | mean delta | 95% CI | threshold-mass delta | lower-tail delta |
|---|---:|---:|---|---:|---:|
| pounce count vs other | 359 | +4.102 | `[+3.144, +5.161]` | +0.049 | -0.046 |
| pounce count sets now vs other | 134 | +6.069 | `[+4.276, +8.086]` | +0.073 | -0.064 |
| reckless count vs non-reckless | 1,355 | -8.372 | `[-8.997, -7.780]` | -0.119 | +0.117 |
| safe partner count vs other | 400 | +0.683 | `[+0.151, +1.228]` | +0.007 | -0.010 |
| unsafe partner count vs non-unsafe | 559 | -8.428 | `[-9.285, -7.570]` | -0.132 | +0.107 |

Interpretation:

- the pounce gate matters because positive pounce labels and the reckless
  negative control point in opposite directions;
- immediate set threshold strengthens the pounce signal;
- safe partner count donation remains real but small under the current-control
  label;
- unsafe partner count donation is a clear negative-control boundary.

## Slice Additions

This replication adds:

- 166 per-label slice rows over declaration, seat role, team, trick index, trick
  position, count amount, current-control side, candidate win/beat flags,
  later-seat count, and sets-now label;
- 133 paired-contrast slice rows using the preferred action's slice context.

Example slice findings:

| slice | paired decisions | mean delta | 95% CI |
|---|---:|---:|---|
| pounce count, five-count actions | 201 | +4.364 | `[+3.092, +5.751]` |
| pounce count, ten-count actions | 158 | +3.768 | `[+2.254, +5.309]` |
| pounce count, ones declaration | 41 | +5.484 | `[+2.571, +9.132]` |
| pounce count, twos declaration | 36 | +2.916 | `[-0.218, +6.237]` |

Slices should be read as exploratory diagnostics unless their paired counts are
large and the operationalization is unchanged. They help decide where to dig
next; they do not automatically move ledger statuses.

## Harness Migration

The replication proves the [[w42-phase2-claim-analysis-harness]] can run over
full Gus legal-action rows, not only label-filtered JSONL. The harness output
under `w42/tactical_claim_replication/harness/` records six registered Gus
tactical claim specs and six label metric rows.

The remaining migration is to move raw corpus replay and public trick-context
construction from the old deep-dive script into shared `w42/claim_analysis/`
adapters. The current bead leaves that as a tooling follow-up rather than
rewriting the original analyzer.

## Leakage Boundary

Public role/trick/action labels are report features. E[Q], `q_per_world`,
`world_hands`, sampled outcomes, threshold mass, and lower-tail mass are offline
labels or diagnostics only. No hidden-world truth becomes a live feature.

## Commands

```bash
python w42/tactical_claim_replication/run_tactical_replication.py \
  --output-dir w42/tactical_claim_replication \
  --bootstrap-samples 2000 \
  --min-slice-n 20 \
  --min-paired-slice-n 20 \
  --wandb-mode online \
  --wandb-group w42-tactical-claim-replication \
  --wandb-name t42-0b4l.3-tactical-replication-v0

python -m py_compile w42/tactical_claim_replication/run_tactical_replication.py
```

## Claim-Ledger Impact

No status changes. The existing narrow reading remains:

- operationalized setter pounce and reckless-count negative control are
  supported on the Gus v2 generated corpus slice;
- safe partner count donation remains context-limited until a guaranteed-control
  and later-seat risk detector replaces current-control as the safety proxy;
- unsafe partner count donation is supported as a negative-control warning.

## Links

[[w42]] | [[w42-gus-corpus-tactical-claim-deep-dive]] |
[[w42-phase2-claim-analysis-harness]] |
[[w42-setter-defense-claim-validation]] |
[[w42-partner-support-claim-validation]] |
[[winning42-ch04-partner-support]] |
[[winning42-ch05-setter-defense]]
