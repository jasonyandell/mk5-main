---
title: GEN_FLEET (Vast.ai distributed corpus generation)
kind: entity
first_seen: 2026-04-21
last_updated: 2026-07-15
status: superseded
---

## What it is

GEN_FLEET is a plan for scaling [[gus]]'s [[joint-world-tensor]] corpus generation across a
Vast.ai fleet with HuggingFace as durable storage. Pattern mirrors `forge/zeb/vast/`. At
this frontier it is a design doc, not yet launched. (commit message @ a8bc35a)

## Architecture

- **Workers**: stateless processes on interruptible Vast.ai 3090s/4090s. Each covers a
  static seed range. Pushes chunks directly to HF dataset
  `jasonyandell/gus-42-worlds` (repo never published / no longer exists — see
  [[huggingface-assets]] § Referenced but absent).
- **Consumer**: local M5 Max pulls chunks from HF and trains. No learner loop, no
  coordination service — embarrassingly parallel at the seed-range level.
- No training on the fleet side — generation only.

## Cost envelope

~$15 for an 8-worker 7.5-hour run covering 10k diverse-seed games. Reduces local wall time
from ~100 hours to hours. (commit message @ a8bc35a)

## Prerequisite

`--n-decl-per-seed` flag in the [[forge]] generator. The oracle was trained on 10
declarations per seed; Gus currently generates 1 per seed → 100× fewer distinct states per
unit compute. Adding all 10 declarations per seed is the largest single state-diversity
lever available before scaling games. (commit message @ a8bc35a)

## Pre-launch fix list (captured 2026-04-21, commit a14200f)

Working premise: every worker-hour spent generating data under a broken assumption is an
hour paid for twice. All five fixes must land before the diverse-seed fleet runs.

| Fix | Status | Detail |
|---|---|---|
| Lazy dataset loading | **Done** (f138069) | `JointWorldFullIterable` unblocks 100k+ corpus |
| Oracle bid=30 bias | **Not shipped** — high priority | `select_actions.py` hardcodes P(Q ≥ 18) / P(Q ≥ -17) bin offsets for bid=30; `decl_id` encodes trump only, not bid amount. Higher bids need stricter thresholds. Fix: plumb `bid_value` through gen pipeline so threshold shifts at the correct cliff. User priority: preserve "29 is never 30" — the margin-above-threshold behavior is acceptable as-is. |
| Schema v2 | **Not shipped** | Add `bid_value: int` + per-seat oracle action softmax to `DecisionRecordGPU`. Required for LAMIR π_opp targets. Ship alongside bid-threshold fix so corpus re-gen only happens once. |
| Length-cache sidecar | **Not shipped** | Store chunk `__len__` as sidecar `.len` file on HF alongside the chunk so consumers skip the one-time scan after download. |
| Resume-safe HF upload | **Not shipped** | Validate that `huggingface-cli upload` handles a preempted mid-upload cleanly; verify skip-if-exists on restart. |

## Phasing

0. **Generator harden**: `--n-decl-per-seed` flag, chunk naming convention, bid-threshold
   fix, schema v2.
1. **Fleet-of-1**: one worker, small seed range, validate end-to-end (HF push → local pull
   → train → eval).
2. **Scale fleet**: 4-16 workers via `vast_monitor.sh`.
3. **Local consumer**: periodic HF pull + retrain trigger.

## Outcome: never launched

The fleet never ran past phase 0. There is no `jasonyandell/gus-42-worlds` HF dataset (zero
repo-wide hits for the name) and no fleet-provenance commit beyond the pre-launch fix list
above — one commit total (`b1de970`, bulk migration), 2.5 months with zero revisit as of
2026-07-06. The scaling goal this plan targeted was reached by a different path: the 10k-game
corpus that actually shipped ([[consistency-regularizer]]'s v3-10k run) was generated locally,
consumed through [[lazy-iterable-dataset]]'s streaming loader rather than a distributed fleet.
Superseded by local generation; kept as the design record for the Vast.ai pattern should
corpus generation ever need to scale past single-machine wall time again.
