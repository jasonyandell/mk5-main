---
title: LAMIR-1 Pilot — Argmax Opp π_me + V_head Leaf
kind: experiment
first_seen: 581bf1f
last_updated: 581bf1f
status: active
---

## Summary

First end-to-end LAMIR-1 rollout on 560 held-out decisions. 1-ply look-ahead uses
rotation-equivariant π_me as π_opp (no separate π_opp head required) and V_head as the
leaf evaluator. The rollout actively hurts vs direct π_me baseline. (commit message @ 581bf1f)

## Setup

- **Adapter**: v3_consistency_10k (baseline regret 0.551, bot-match 76.07%)
- **Rollout**: for each legal action, simulate remaining trick plays using π_me in
  rotation-equivariant mode as the opponent policy, then score the leaf state with V_head
  averaged over M corpus worlds
- **Script**: `gus/eval/lamir1.py` (536 lines, new)

## Results

| Mode | Regret | Bot-match |
|---|---|---|
| Direct π_me (baseline) | 0.551 | 76.07% |
| LAMIR-1 (argmax opp + V_head leaf) | 2.384 | 62.86% |

Rollout is 4.3× worse on regret. (commit message @ 581bf1f)

## Failure anatomy

Damage concentrates at trick_pos 0-2 (leader and early followers). Trick_pos 3 positions
use direct π_me as fallback and are unchanged (0.551 regret, 76% match).

Root cause: argmax opponent simulation via π_me creates adversarial leaf states that do
not reflect oracle-equilibrium play. V_head correctly evaluates those worse leaf states —
so the action scoring diverges from E[Q] ordering. The look-ahead degrades rather than
improves action selection. (commit message @ 581bf1f)

## What "it runs" establishes

- LAMIR-1 infrastructure end-to-end functional: world sampling, π_me rotation, V_head leaf
- Per-trick-pos breakdown available as a diagnostic tool
- The failure mode is clearly diagnosed and points to two separable issues:
  opponent simulation quality and leaf evaluator distribution shift

Kept active: this is the first-attempt qualitative record (rollout hurts, damage
concentrates at trick_pos 0-2) and it is correctly self-scoped to its own commit. The 2.384
regret number above is this pilot's own reading and is superseded by the post-bug-fix
canonical value (2.094) in [[experiments/gus-lamir1-mode-comparison]] and [[lamir1-ceiling]] —
go there for current numbers.

## Links

[[gus]] · [[topics/lamir1]] · [[topics/pimc]] · [[experiments/gus-lamir1-mode-comparison]] · [[experiments/gus-v3-consistency-full-run]] · [[topics/lamir1-ceiling]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- `gus/eval/lamir1.py` has since grown to 1124 lines (later modes and bug fixes); the "536 lines, new" figure is accurate for the pilot commit this page describes.
