---
title: Eval Seed Holdout
kind: decision
first_seen: 2026-04-10
last_updated: 2026-04-10
status: active
---

## Decision

Seeds 900000–909999 are permanently reserved for evaluation. They must never appear in any training dataset. All [[expected-q-value]] delta measurements use this range.

> Declared here so it's never ambiguous.

— `lem/OVERVIEW.md` at [[sources/b99c64d]]

## Training / validation split

| Range | Role |
|---|---|
| 0–899999, `seed % 1000 < 950` | Training data |
| 0–899999, `seed % 1000 >= 950` | Validation within training data |
| 900000–909999 | Pure held-out eval — never trained on, never validated on |

## Enforcement

The [[narration]] batch generator (`lem/narrate/batch.py`) rejects any seed in 900000–909999 unless `--allow-eval-seeds` is explicitly passed. This flag exists solely to generate `lem/data/narrations_eval.jsonl`; normal generation runs must not use it.

## Why declared here

Making the boundary explicit in `lem/OVERVIEW.md` prevents ambiguity as the dataset grows across training iterations. A seed that appears in eval must never migrate to any training split, even accidentally.

## Related pages

[[lem]] · [[narration]] · [[experiments/stage-0-v1-training]] · [[sources/b99c64d]]
