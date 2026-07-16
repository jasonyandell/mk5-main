---
title: w42 Strategy Tags v0
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-15
status: complete
---

## Summary

[[w42]] now has a w42-owned Strategy Detector v0 wrapper for the cheap
public-state/action-local tags that [[gus-strategy-tags-probe]] found useful:

- `w42/strategy_tags_v0.py`
- `w42/strategy_tags_v0/manifest.json`
- `w42/strategy_tags_v0/report.json`
- `w42/strategy_tags_v0/tag_schema.json`
- `w42/strategy_tags_v0/example_row.json`
- `w42/strategy_tags_v0/summary.csv`

The wrapper imports the existing Gus detector implementation and validates that
the w42 tag metadata still matches the Gus constants: `strategy_features` has 68
global public-state dimensions, and `strategy_action_features` has 32 dimensions
per action slot across 7 action slots. Gus code, Gus training defaults, Burl
code, and forge oracle semantics were not modified.

This bead validates the detector surface and report shape only. It does not test
a Winning 42 strategy claim, train a model, run W&B, or publish a HF artifact.

## Data Slice

The input is the manifest train split using `gus/data/corpus_train_100.pt`.
These bulk corpus files are not in git; they live on HuggingFace at
[texas-42-joint-world-corpus](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus)
(flat basenames at repo root), with `corpus_v2_*` files on
[texas-42-joint-world-corpus-v2](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus-v2);
see [[huggingface-assets]].
The wrapper intentionally consumes only the first available declared corpus path
for this v0 surface check, so it remains a small deterministic tag-shape run
rather than a bulk corpus scan.

Source mode: `real-corpus`.

Blocker note: none for the main checkout. The script keeps a fixture fallback for
machines without local corpus bytes.

Declared inputs checked, in order:

| path or glob | consumed |
|---|---|
| `gus/data/corpus_train_100.pt` | consumed |
| `gus/data/corpus_train_chunk_*-*.pt` | not consumed in this smoke |
| `gus/data/corpus_v2_train_*_d0-9.pt` | not consumed in this smoke |
| `data/eq-games/train` | not consumed in this smoke |
| `data/eq-games/val` | not consumed in this smoke |
| `data/eq-games/test` | not consumed in this smoke |
| `gus/data/corpus_eval_20.pt` | not consumed in this smoke |
| `gus/data/corpus_v2_eval.pt` | not consumed in this smoke |

## Tag Dimensions

The full tag-name schema is recorded in
`w42/strategy_tags_v0/tag_schema.json`.

Global public-state groups:

| group | width |
|---|---:|
| declaration | 10 |
| phase | 3 |
| hand_shape | 10 |
| legal_action_summary | 6 |
| public_count | 5 |
| void_summary | 4 |
| visible_pip_coverage | 7 |
| current_trick_pressure | 9 |
| own_pip_coverage | 7 |
| unseen_count_by_pip | 7 |
| total | 68 |

Action-local groups:

| group | width |
|---|---:|
| slot | 2 |
| identity | 8 |
| trick_relation | 6 |
| count_pressure | 2 |
| suit_pressure | 4 |
| pip_pressure | 3 |
| double_protection | 3 |
| hand_shape | 2 |
| donation_window | 2 |
| total | 32 |

These are public-state and action-local features. Oracle labels such as `e_q`,
`q_per_world`, and `action_taken` remain labels or diagnostics, not detector
inputs.

## Sample Output

The deterministic real-corpus sample has one row:

| field | value |
|---|---|
| `decision_idx` | `0` |
| `player` | `0` |
| `action_taken` | `5` |
| `oracle_best_action` | `5` |
| `legal_mask` | `[true, true, true, true, true, true, true]` |
| `strategy_features` shape | `[1, 68]` |
| `strategy_action_features` shape | `[1, 7, 32]` |

Group summaries are in `w42/strategy_tags_v0/summary.csv`. The numeric
values validate real-corpus loading, shape, naming, and determinism only; they
are not semantic evidence about play quality.

## Reproducibility

Run commit at artifact generation:
`e3bfefa7f3f01b32c39b74545badbb6d1b27a642` (per `w42/strategy_tags_v0/report.json`).

Exact detector command:

```bash
python w42/strategy_tags_v0.py --seed 42 --limit 1
```

Config:

| key | value |
|---|---|
| bead | `t42-csw6.7` |
| device | `cpu` |
| source mode | `real-corpus` |
| data input | `gus/data/corpus_train_100.pt` |
| output directory | `w42/strategy_tags_v0/` |
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
| world sampling | `42` |
| data generation | `not applicable` |
| eval sampling | `not applicable` |

## Commands And Checks

Required reading and grounding:

```bash
git status --short --branch
bd show t42-csw6.7 --json
sed -n '1,220p' wiki/AGENTS.md
sed -n '1,260p' wiki/entities/w42.md
sed -n '1,260p' wiki/experiments/w42-data-adapter-smoke.md
sed -n '1,320p' wiki/experiments/w42-dataset-manifest.md
sed -n '1,320p' wiki/experiments/w42-claim-ledger.md
sed -n '1,420p' wiki/experiments/gus-strategy-tags-probe.md
sed -n '1,280p' gus/model/strategy_features.py
sed -n '280,620p' gus/model/strategy_features.py
sed -n '1,360p' gus/model/dataset_seq_world.py
sed -n '360,760p' gus/model/dataset_seq_world.py
```

Implementation and artifact checks:

```bash
python w42/strategy_tags_v0.py --seed 42 --limit 1
sed -n '1,220p' w42/strategy_tags_v0/report.json
sed -n '1,80p' w42/strategy_tags_v0/summary.csv
sed -n '1,140p' w42/strategy_tags_v0/example_row.json
python -m py_compile w42/strategy_tags_v0.py
git status --short --untracked-files=all
```

## Claim Ledger

no claim-ledger change

This detector surface validates that w42 can name, group, and emit the cheap
public-state/action-local tags from the Gus probe. It creates no empirical
evidence for a Winning 42 strategy claim and does not move any claim status.

## Links

[[w42]] | [[w42-dataset-manifest]] | [[w42-data-adapter-smoke]] |
[[w42-claim-ledger]] | [[gus-strategy-tags-probe]] | [[gus]]
