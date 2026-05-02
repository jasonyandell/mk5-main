---
title: w42 Data Adapter Smoke
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] has a tiny promoted-scratch smoke path for the [[forge]] / [[gus]]
public-state, action, and oracle-label adapter shape:

- `scratch/w42/data_adapter_smoke.py`
- `scratch/w42/data_adapter_smoke/manifest.json`
- `scratch/w42/data_adapter_smoke/report.json`
- `scratch/w42/data_adapter_smoke/example_row.json`

The local worktree did not contain the declared Gus or Forge corpora, so this
bead produced a deterministic fixture-backed smoke rather than a real-data
smoke. This is a blocker for proving the adapter on local corpus bytes, but it
does validate the expected batch surface and records the missing inputs in the
manifest.

No serious model was trained.

## Data Inputs

The smoke checked the declared w42 source shapes from
[[w42-dataset-manifest]]:

| path or glob | observed locally |
|---|---|
| `gus/data/corpus_train_100.pt` | absent |
| `gus/data/corpus_train_chunk_*-*.pt` | absent |
| `gus/data/corpus_v2_train_*_d0-9.pt` | absent |
| `gus/data/corpus_eval_20.pt` | absent |
| `gus/data/corpus_v2_eval.pt` | absent |
| `data/eq-games/train` | absent |
| `data/eq-games/val` | absent |
| `data/eq-games/test` | absent |

Because the local corpora are absent, the generated manifest records
`exists_at_manifest_time: false` and uses
`scratch/w42/data_adapter_smoke/example_row.json` as the fixture source.

## Batch Shape

The deterministic batch uses the existing Gus adapter field names and strategy
tag dimensions:

| field | shape | role |
|---|---:|---|
| `tokens` | `[1, 33]` | public-state token sequence |
| `attention_mask` | `[1, 33]` | token mask |
| `belief_target` | `[1, 28]` | hidden-owner label fixture |
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
| `action_taken` | `1` |
| `oracle_best_action` | `1` |

## Example Row

The full example row is in
`scratch/w42/data_adapter_smoke/example_row.json`.

Important fields:

```json
{
  "decision_idx": 0,
  "player": 0,
  "action_taken": 1,
  "legal_mask": [true, true, true, false, true, false, false],
  "e_q": [3.25, 6.5, 5.75, -99.0, 4.0, -99.0, -99.0],
  "q_per_world": [2.5, 7.0, 4.75, -8.0, 3.25, -8.0, -8.0]
}
```

The fixture keeps oracle values as labels only. They are not included in
`tokens`, `strategy_features`, or `strategy_action_features`.

## Reproducibility

Run commit at smoke generation:
`fe8500155f4a3dd003feb6e3a2a92c35db28ea4d`.

Exact command:

```bash
python scratch/w42/data_adapter_smoke.py --seed 42 --batch-size 1
```

Config:

| key | value |
|---|---|
| bead | `t42-csw6.4` |
| batch size | `1` |
| device | `cpu` |
| source mode | `fixture` |
| data manifest | `scratch/w42/data_adapter_smoke/manifest.json` |
| report JSON | `scratch/w42/data_adapter_smoke/report.json` |
| fixture source | `scratch/w42/data_adapter_smoke/example_row.json` |
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
rg --files gus data scratch/w42 wiki/experiments | rg 'gus/data|corpus|eq-games|JointWorld|strategy_features|strategy_probe|w42'
rg -n "class JointWorldFullDataset|JointWorldFullDataset|strategy_features|strategy_action_features|action_taken|q_per_world|e_q|legal_mask" gus -g '*.py'
find gus/data data/eq-games scratch/w42 -maxdepth 3 -type f
sed -n '1,230p' gus/model/dataset_seq_world.py
sed -n '1,220p' gus/model/strategy_features.py
sed -n '220,380p' gus/model/strategy_features.py
find scratch/w42 -maxdepth 3 -type f -print
```

Smoke run and output inspection:

```bash
python scratch/w42/data_adapter_smoke.py --seed 42 --batch-size 1
sed -n '1,220p' scratch/w42/data_adapter_smoke/report.json
sed -n '1,220p' scratch/w42/data_adapter_smoke/example_row.json
sed -n '1,240p' scratch/w42/data_adapter_smoke/manifest.json
git status --short --untracked-files=all
```

## Claim Ledger

no claim-ledger change

This smoke validates adapter shape and fixture determinism only. It creates no
empirical evidence for a Winning 42 strategy claim and does not move any claim
status.

## Links

[[w42]] | [[w42-dataset-manifest]] | [[w42-lab-infrastructure]] |
[[w42-claim-ledger]] | [[gus-strategy-tags-probe]] | [[forge]] | [[gus]]
