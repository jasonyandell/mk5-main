---
title: w42 Data Adapter Smoke
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] has a tiny promoted smoke path for the [[forge]] / [[gus]]
public-state, action, and oracle-label adapter shape:

- `w42/data_adapter_smoke.py`
- `w42/data_adapter_smoke/manifest.json`
- `w42/data_adapter_smoke/report.json`
- `w42/data_adapter_smoke/example_row.json`

The main checkout contains `gus/data/corpus_train_100.pt`, so the merged smoke
now validates the adapter against a real Gus joint-world corpus row. The script
still keeps a deterministic fixture fallback for machines without local corpus
bytes, but the recorded report for this run is `source_mode: real-corpus`.

No serious model was trained.

## Data Inputs

The smoke checked the declared w42 source shapes from
[[w42-dataset-manifest]]:

| path or glob | observed locally |
|---|---|
| `gus/data/corpus_train_100.pt` | present; loaded as the smoke source |
| `gus/data/corpus_train_chunk_*-*.pt` | not consumed in this smoke |
| `gus/data/corpus_v2_train_*_d0-9.pt` | not consumed in this smoke |
| `gus/data/corpus_eval_20.pt` | not consumed in this smoke |
| `gus/data/corpus_v2_eval.pt` | not consumed in this smoke |
| `data/eq-games/train` | not consumed in this smoke |
| `data/eq-games/val` | not consumed in this smoke |
| `data/eq-games/test` | not consumed in this smoke |

The smoke intentionally loads only the first available declared corpus path so
it remains a small adapter check rather than a bulk data scan. The generated
manifest records `exists_at_manifest_time: true` for
`gus/data/corpus_train_100.pt`.

## Batch Shape

The deterministic batch uses the existing Gus adapter field names and strategy
tag dimensions:

| field | shape | role |
|---|---:|---|
| `tokens` | `[1, 33, 5]` | public-state token sequence |
| `attention_mask` | `[1, 33]` | token mask |
| `belief_target` | `[1, 28]` | hidden-owner label |
| `belief_mask` | `[1, 28]` | belief supervision mask |
| `world_assignment` | `[1, 28, 3]` | sampled-world hidden-seat assignment |
| `q_per_world` | `[1, 7]` | oracle per-world Q label |
| `e_q` | `[1, 7]` | marginal E[Q] oracle label |
| `action_taken` | `[1]` | oracle/action policy label |
| `legal_mask` | `[1, 7]` | action legality |
| `decision_idx` | `[1]` | decision position |
| `player` | `[1]` | acting player |
| `voids` | `[1, 24]` | public void features |
| `strategy_features` | `[1, 68]` | cheap public-state strategy tags |
| `strategy_action_features` | `[1, 7, 32]` | cheap action-local strategy tags |

Derived labels:

| label | value |
|---|---:|
| `action_taken` | `5` |
| `oracle_best_action` | `5` |

## Example Row

The full example row is in
`w42/data_adapter_smoke/example_row.json`.

Important fields:

```json
{
  "decision_idx": 0,
  "player": 0,
  "action_taken": 5,
  "legal_mask": [true, true, true, true, true, true, true],
  "e_q": [20.909, 15.15, 18.93, 16.764, 14.965, 21.372, 12.356]
}
```

The adapter keeps oracle values as labels only. They are not included in
`tokens`, `strategy_features`, or `strategy_action_features`.

## Reproducibility

Run commit at smoke generation:
`68e6ee3afb659e26239f72264e5c94515ef947f9`.

Exact command:

```bash
python w42/data_adapter_smoke.py --seed 42 --batch-size 1
```

Config:

| key | value |
|---|---|
| bead | `t42-csw6.4` |
| batch size | `1` |
| device | `cpu` |
| source mode | `real-corpus` |
| data input | `gus/data/corpus_train_100.pt` |
| data manifest | `w42/data_adapter_smoke/manifest.json` |
| report JSON | `w42/data_adapter_smoke/report.json` |
| checkpoint | `not applicable` |
| W&B links | `not applicable` |
| HF links | `not applicable` |
| claim ledger impact | `no claim-ledger change` |

Random seeds:

| seed | value |
|---|---:|
| torch | `42` |
| dataset shuffle | `42` |
| train loader | `42` |
| data generation | `not applicable` |
| eval sampling | `not applicable` |

## Commands And Checks

Required reading and local state checks:

```bash
git status --short --branch
git rev-parse HEAD
bd show t42-csw6.4 --json
sed -n '1,240p' wiki/AGENTS.md
sed -n '1,260p' wiki/entities/w42.md
sed -n '1,280p' wiki/experiments/w42-dataset-manifest.md
sed -n '1,260p' wiki/experiments/w42-lab-infrastructure.md
sed -n '1,320p' wiki/experiments/w42-claim-ledger.md
sed -n '1,340p' wiki/experiments/gus-strategy-tags-probe.md
sed -n '1,320p' wiki/entities/gus.md
sed -n '321,760p' wiki/entities/gus.md
sed -n '1,340p' wiki/entities/forge.md
```

Code and data inspection:

```bash
rg --files gus data w42 wiki/experiments | rg 'gus/data|corpus|eq-games|JointWorld|strategy_features|strategy_probe|w42'
rg -n "class JointWorldFullDataset|JointWorldFullDataset|strategy_features|strategy_action_features|action_taken|q_per_world|e_q|legal_mask" gus -g '*.py'
find gus/data data/eq-games w42 -maxdepth 3 -type f
sed -n '1,230p' gus/model/dataset_seq_world.py
sed -n '1,220p' gus/model/strategy_features.py
sed -n '220,380p' gus/model/strategy_features.py
find w42 -maxdepth 3 -type f -print
```

Smoke run and output inspection:

```bash
python w42/data_adapter_smoke.py --seed 42 --batch-size 1
sed -n '1,220p' w42/data_adapter_smoke/report.json
sed -n '1,220p' w42/data_adapter_smoke/example_row.json
sed -n '1,240p' w42/data_adapter_smoke/manifest.json
git status --short --untracked-files=all
```

## Claim Ledger

no claim-ledger change

This smoke validates adapter shape and deterministic real-corpus loading only. It creates no
empirical evidence for a Winning 42 strategy claim and does not move any claim
status.

## Links

[[w42]] | [[w42-dataset-manifest]] | [[w42-lab-infrastructure]] |
[[w42-claim-ledger]] | [[gus-strategy-tags-probe]] | [[forge]] | [[gus]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 2 corrections applied in place and independently re-verified; second pass amended 1.

- `gus/data/corpus_train_100.pt` is gitignored, so the real-corpus source bytes trace only to the local checkout, not to any commit.
