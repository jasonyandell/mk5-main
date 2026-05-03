---
title: w42 Phase 2 Claim Analysis Harness
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] now has a small reusable claim-analysis harness at `w42/claim_analysis/`.
It is the first extraction from the one-off
[[w42-gus-corpus-tactical-claim-deep-dive]] loop into shared tooling for later
claim beads.

The harness consumes existing row artifacts, combines configured label fields,
computes per-label metrics, runs paired same-decision label-vs-nonlabel
contrasts where all legal actions are present, bootstraps CIs, extracts
examples, writes CSV/JSON artifacts, and logs W&B progress series.

This is harness/tooling work. It does not move any claim status.

## Code Surface

| file | role |
|---|---|
| `w42/claim_analysis/harness.py` | row normalization, label parsing, bootstrap CIs, label metrics, paired contrasts, artifact writing, W&B progress logging |
| `w42/claim_analysis/run_existing_rows.py` | CLI for Gus claim rows, branch-atlas action tables, phase-2 decision tables, or custom row artifacts |
| `w42/claim_analysis/__init__.py` | package exports |

The first version expects row artifacts that already expose public/action facts
and offline labels. Raw Gus corpus replay still lives in
`w42/gus_corpus_claim_deep_dive/analyze_gus_claims.py`; the next extraction
should move replay/context construction into a shared adapter.

## Artifact Schema

Each harness run writes:

| artifact | meaning |
|---|---|
| `label_metrics.csv` | one row per label with action count, decision count, actual/best/safest rates, mean Q, regret, threshold mass, lower-tail mass, and bootstrap CI |
| `paired_contrasts.csv` | one row per label when same-decision nonlabel alternatives exist |
| `label_counts.csv` | raw action counts by parsed label |
| `claim_specs.csv` | registered claim specs with online fields, offline labels, and leakage policy |
| `examples.json` | high-absolute-delta paired examples |
| `summary.json` | run coverage, label fields, W&B status, leakage statement |
| `manifest.json` | inputs, outputs, repo commit, dirty status |

Supported source kinds:

| source kind | default label fields |
|---|---|
| `gus_claim_rows` | `labels` |
| `branch_atlas_actions` | `distribution_shape_tags`, `strategy_context_tags`, `matched_position_detectors`, `direct_label_readiness` |
| `phase2_decision_actions` | `distribution_shape_tags`, `strategy_context_tags`, `matched_position_detectors`, `direct_label_readiness` |
| `custom_rows` | caller-provided `--label-field` values |

## Smoke Runs

Branch-atlas smoke:

```bash
python w42/claim_analysis/run_existing_rows.py \
  --input w42/branch_atlas_scaled_v0/decision_actions.csv \
  --output-dir w42/claim_analysis_smoke/branch_atlas_scaled_v0_wandb \
  --source-kind branch_atlas_actions \
  --min-label-n 10 \
  --bootstrap-samples 200 \
  --progress-chunk-size 100 \
  --wandb-mode online \
  --wandb-group w42-claim-analysis-harness \
  --wandb-name t42-0b4l.2-branch-atlas-smoke
```

Result:

| metric | value |
|---|---:|
| action rows | 773 |
| decision rows | 280 |
| label metric rows | 36 |
| paired contrast rows | 14 |
| registered claim specs | 6 |
| W&B run | `https://wandb.ai/jasonyandell-forge42/w42/runs/rj3j0jsz` |

The W&B run logs eight progress points over `progress/rows_processed`, with
coverage and label-count series. This satisfies the dashboard rule for harness
work: the run has a trajectory, not only a final scalar.

Gus claim-row smoke:

```bash
python w42/claim_analysis/run_existing_rows.py \
  --input w42/gus_corpus_claim_deep_dive/claim_action_rows.jsonl \
  --output-dir w42/claim_analysis_smoke/gus_claim_rows \
  --source-kind gus_claim_rows \
  --min-label-n 5 \
  --bootstrap-samples 200 \
  --progress-chunk-size 500 \
  --wandb-mode disabled
```

Result:

| metric | value |
|---|---:|
| action rows | 3903 |
| decision rows | 3428 |
| label metric rows | 6 |
| paired contrast rows | 4 |

The Gus smoke verifies label parsing and metric reproduction over the exported
claim rows. Its paired contrasts are not equivalent to the original
[[w42-gus-corpus-tactical-claim-deep-dive]] contrasts because
`claim_action_rows.jsonl` is label-filtered and does not contain every legal
alternative. Future powered tactical replications should run the harness on
full legal-action rows or move the raw corpus replay adapter into
`w42/claim_analysis/`.

## Migration Notes

The one-off Gus deep dive already has the correct scientific loop:

- reconstruct public trick context from recorded actions;
- emit one row per legal action;
- attach public/action-local detector labels;
- keep E[Q], PDFs, `q_per_world`, and hidden sampled worlds as offline labels;
- compare actions within the same decision;
- write metrics, paired contrasts, examples, manifest, and W&B progress.

The reusable harness now owns the row-level half of that loop. The remaining
migration is the adapter layer: `process_game`, current-winner reconstruction,
candidate facts, and threshold computation should move from the deep-dive script
and [[w42-branch-atlas-scaled-v0]] into shared corpus adapters.

## Leakage Policy

The harness can read rows that contain oracle means, E[Q] PDFs, `q_per_world`,
`world_hands`, hidden-holder labels, and future outcomes. Those fields are
offline labels for analysis, training targets, or diagnostics. They are not live
policy inputs.

Claim specs and future adapters should declare:

- online-safe public fields;
- actor-private legal fields;
- offline-only labels;
- forbidden hidden-truth fields;
- the promotion rule that a detector is not evidence by itself.

## Next Use

The next bead, `t42-0b4l.3`, can use this harness for setter-pounce and partner
count-donation replication if it first supplies full legal-action rows with
guaranteed-control, later-seat risk, and bid-threshold labels.

`t42-0b4l.4` can use the same row-level shape for hidden-threat branch labels
from [[w42-branch-atlas-scaled-v0]], but those labels remain offline diagnostic
targets.

## Claim-Ledger Impact

No claim ledger status moved. This page establishes reusable tooling and smoke
coverage for later claim-analysis beads.

## Links

[[w42]] | [[w42-phase2-claim-analysis-matrix]] |
[[w42-gus-corpus-tactical-claim-deep-dive]] |
[[w42-branch-atlas-scaled-v0]] |
[[w42-phase2-distribution-aware-ev-report]]
