---
title: "Source: 41fdb3c"
kind: source
commit: 41fdb3c
date: 2026-04-21
author: Jason Yandell
first_seen: 2026-04-24
last_updated: 2026-04-24
---

## Commit message

> docs(gus): PRACTICALITIES §18 — qMAE scaling plateau + known fix path
>
> qMAE only improved 7% with 3.3× more data (3k→10k), while regret
> dropped 59% and V-MAE 20%. Flags the structural issue: Q_head trains
> on one random world per forward pass, no cross-world consistency
> reward. Two candidate fixes (multi-world variance reg, joint
> co-training) documented with cost estimate. LAMIR-1 doesn't depend
> on Q_head so this isn't blocking the north-star track.

## Files changed

| File | Change |
|---|---|
| `gus/PRACTICALITIES.md` | §18 added |

## What this commit establishes

§18: qMAE plateau is structural (one world per forward pass), not a data or capacity
issue. Fix requires multi-world variance regularization or joint co-training. Not
blocking LAMIR-1.

## Links

[[experiments/gus-v3-consistency-full-run]] · [[topics/dense-q-supervision]] · [[gus]]
