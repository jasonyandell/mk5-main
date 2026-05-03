---
title: w42 Claim-Tag Model Probe
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] has a direct `t42-0b4l.9` claim-tag probe over legal-action rows. The
probe trains a cheap row model to rank legal candidate actions inside each
decision, comparing public/action features alone against public/action features
plus derived claim detector tags from [[w42-phase2-seat-position-strategy-map]].

The artifact is `w42/claim_tag_model_probe/`. It is model-feature evidence, not
claim proof and not a Gus replacement.

W&B run: `https://wandb.ai/jasonyandell-forge42/w42/runs/gn7xxk14`

On the held-out seed split, the direct tag model improves the public-feature
row model. The richer run in `metrics.json` reports `1.483` mean selected regret
and `61.70%` oracle-best match for public features, versus `1.319` regret and
`63.64%` match for public plus claim tags. A coordinator rerun with the smaller
`model_metrics.csv` path independently finds the same direction: `1.384` public
regret versus `1.295` tag regret, with match improving from `64.52%` to
`66.04%`.

## Method

The input table is `w42/seat_position_claim_tests/labeled_action_rows.csv`, which
comes from [[w42-tactical-claim-replication]]. The row grain is one legal action
candidate per public decision. The train/eval split is by seed: seeds `0..79`
train, seeds `80..99` eval.

The target is whether the candidate is the top mean-E[Q] action in its decision.
Evaluation chooses the highest-scored legal action per decision and reports
match rate, mean regret, near-tie rate, and tail-regret rate.

Model variants:

| variant | feature surface |
|---|---|
| `public_features_only` | public row context, role/position categorical fields, and candidate action facts |
| `public_plus_claim_tags` | public features plus derived detector tags |
| `drop_pounce_donation` | full tag surface with pounce/donation tags removed |
| `drop_seat_position_closure` | full tag surface with seat/position/closure tags removed |

The bid-risk, 84/endgame, doubles/no-trump, and hidden-threat/distribution
families are recorded in the tag inventory but are not used as live features in
this small training table. Bid-risk and 84 have separate artifacts; hidden truth
and distribution labels remain offline labels only.

## Results

Primary trained-variant table:

| variant | status | eval decisions | mean regret | match | tail regret >=5 | interpretation |
|---|---|---:|---:|---:|---:|---|
| `public_baseline` | trained | 5600 | 1.483 | 61.70% | 10.43% | public numeric/context/action facts only |
| `public_plus_claim_tags` | trained | 5600 | 1.319 | 63.64% | 8.95% | direct detector tags help on this split |
| `drop_pounce_donation` | trained | 5600 | 1.416 | 62.29% | 9.77% | removing pounce/donation gives back most of the gain |
| `drop_seat_position_closure` | trained | 5600 | 1.372 | 63.16% | 9.39% | removing seat/closure hurts less, but still weakens the full tag model |

The largest available family contribution is pounce/donation. Dropping those
tags worsens mean regret by `+0.098` versus the full tag model; dropping
seat-position/closure worsens by `+0.053`. The result is encouraging but modest:
the actual policy action in this corpus remains much stronger than either cheap
row model, at about `0.122` mean regret in the coordinator rerun.

The direct probe also scores slices. The tag model improves the pounce/donation
slice from `1.299` to `1.031` selected regret, the closure slice from `1.312` to
`1.064`, no-trump declarations from `1.627` to `1.478`, and doubles declarations
from `1.356` to `1.260`. The high-hidden-tail slice barely moves, which is the
right conservative behavior because hidden-threat labels are not legal live
features.

## Artifacts

| artifact | role |
|---|---|
| `w42/claim_tag_model_probe/run_claim_tag_model_probe.py` | coordinator rerunnable direct row probe |
| `w42/claim_tag_model_probe/validate_outputs.py` | artifact validator |
| `w42/claim_tag_model_probe/metrics.json` | primary richer run metrics, slices, and OOD 84 score |
| `w42/claim_tag_model_probe/ablation_results.csv` | primary family-drop table |
| `w42/claim_tag_model_probe/model_metrics.csv` | coordinator rerun metrics table |
| `w42/claim_tag_model_probe/tag_inventory.csv` | available and blocked tag families |
| `w42/claim_tag_model_probe/feature_manifest.json` | public features, tag families, and excluded offline label columns |
| `w42/claim_tag_model_probe/prediction_sample.jsonl` | compact prediction sample |
| `w42/claim_tag_model_probe/summary.json` | coordinator run summary |
| `w42/claim_tag_model_probe/manifest.json` | coordinator run manifest |
| W&B run | per-variant summary series at `gn7xxk14` |

## Leakage Boundary

Inputs are public row context, public candidate-action facts, and derived
public/action-local detector tags. Oracle mean/regret values are targets and
metrics only. Hidden truth, sampled worlds, and `q_per_world` are not live
features.

## Claim-Ledger Impact

no central claim-ledger change

This probe tests whether detector tags help a small model. It does not validate
book claims by itself.

## Phase-3 Continuation

[[w42-phase3-joined-claim-row-model-table]] keeps the same cheap row-model idea
but joins the newly available auction, sequence/seat, 84, doubles/no-trump, and
public hidden-proxy families into one ablation table. The phase-3 table replaces
this page's unavailable-family caveat with public-safe joins and explicit
eval-only boundaries.

## Commands

```bash
python -m py_compile w42/claim_tag_model_probe/run_claim_tag_model_probe.py
python w42/claim_tag_model_probe/run_claim_tag_model_probe.py \
  --output-dir w42/claim_tag_model_probe \
  --max-iter 300
python w42/claim_tag_model_probe/validate_outputs.py \
  --artifact-dir w42/claim_tag_model_probe
```

W&B summary logging:

```bash
python - <<'PY'
import csv, json, wandb
from pathlib import Path
base = Path('w42/claim_tag_model_probe')
rows = list(csv.DictReader((base / 'ablation_results.csv').open()))
run = wandb.init(project='w42', entity='jasonyandell-forge42',
                 group='t42-0b4l',
                 name='t42-0b4l.9-claim-tag-row-probe-summary',
                 mode='online')
for idx, row in enumerate(rows):
    log = {'variant_step': idx, 'variant': row['variant'], 'status': row['status']}
    for key in ['eval_selected_mean_regret', 'eval_oracle_match_rate',
                'eval_near_tie_rate_regret_le_0_5',
                'eval_tail_regret_rate_ge_5',
                'delta_regret_vs_full_tags',
                'delta_regret_vs_public_baseline']:
        if row.get(key):
            log[key] = float(row[key])
    wandb.log(log, step=idx)
run.finish()
PY
```
