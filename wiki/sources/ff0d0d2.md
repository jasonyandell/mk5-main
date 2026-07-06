---
title: "Source digest: ff0d0d2 — use combined narration dataset (7409 examples from seeds 0-499)"
kind: source
first_seen: ff0d0d2
last_updated: ff0d0d2
status: active
---

## Commit

- **SHA:** ff0d0d2033c486b4ad7e4c27fa63ff15b988b2ba
- **Date:** 2026-04-11
- **Author:** Jason Yandell

> ops: use combined narration dataset (7409 examples from seeds 0-499)

## Files modified

| Path | Change |
|---|---|
| `lem/gemma_star/iterate.sh` | 1 insertion, 1 deletion — `DATA` variable switched from `narrations_train.jsonl` to `narrations_train_combined.jsonl` |

## Key details

Expands the narration pool from 3148 examples (seeds 0–199) to 7409 examples (seeds 0–499) — a 2.35x increase in data diversity. The combined dataset is a merge of the original 3148-example file with narrations generated from seeds 200–499.

Significance: iter 5 (the first iteration run against this larger pool) is also the first 42% pass rate in [[experiments/star-10-iterations]], suggesting data diversity contributed to the improvement.

## Related pages

[[experiments/star-10-iterations]] · [[narration]] · [[star-harness]] · [[sources/efad16e]]
