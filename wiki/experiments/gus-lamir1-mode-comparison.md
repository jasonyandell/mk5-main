---
title: LAMIR-1 Mode Comparison (direct / v-bootstrap / q-bootstrap / qleaf)
kind: experiment
first_seen: 7d2af99
last_updated: fb03970
status: superseded
---

## Summary

Four LAMIR-1 inference modes developed and compared across the G8 commit cluster.
All rollout modes underperform direct π_me baseline. Two bugs (Bug 5 and Fix 2) found
and fixed during comparison. Root finding: V_head has distribution shift at
immediately-post-play states; this is the bottleneck, not opponent simulation quality.
(commit messages @ 7d2af99, e4e6862, 566bc4d, 8544fbe, fb03970)

## The four modes

| Mode | Leaf evaluator | Opp model |
|---|---|---|
| `direct` | — (π_me argmax, no look-ahead) | — |
| `v-bootstrap` | V_head (world-blind) | None (depth-1, no opp step) |
| `lamir1` | V_head (world-blind) | π_me rotation-equivariant |
| `q-bootstrap` | Q_head per world (world-conditioned) | None (depth-1) |
| `lamir1-qleaf` | Q_head from trick-winner's POV | π_me rotation-equivariant |

## Results by mode (560 held-out decisions, world-cap=200)

| Mode | Regret | Bot-match | Notes |
|---|---|---|---|
| **direct (baseline)** | **0.551** | **76.07%** | v3-10k |
| v-bootstrap | 2.777 (pre-fix; see below) | 59.46% | Worse than full rollout |
| lamir1 (argmax opp + V_head) | 2.384 | 62.86% | See [[experiments/gus-lamir1-pilot]] |
| q-bootstrap | — | — | Added in 8544fbe; full results in later ingest |
| lamir1-qleaf | — | — | Added in fb03970; full results in later ingest |

**Number correction (2026-07-06 audit):** the 2.777 v-bootstrap regret above is the pre-bug-fix
number from this commit range. The canonical post-fix value, reported in `b42669a`
(PRACTICALITIES §20, see [[lamir1-ceiling]]'s complete ladder), is **1.645** — a 69%
overstatement if 2.777 is read as current. Treat this page as the historical record of the
mode-comparison work itself (bugs found, root-cause reasoning); go to [[lamir1-ceiling]] for
the canonical numbers.

## Per-trick-pos breakdown (v-bootstrap, 7d2af99)

| trick_pos | Regret | Bot-match |
|---|---|---|
| 0 (leads) | 7.41 | 22% |
| 3 (fallback, = direct) | 0.43 | 82% |

trick_pos 3 is unchanged from direct (no rollout applied there). trick_pos 0 (the lead
position — most information-poor, highest spread) has catastrophic regret under both
v-bootstrap and lamir1. (commit message @ 7d2af99)

## Key finding: V_head distribution shift

v-bootstrap at depth-1 — no opponent simulation at all — is as broken as the full rollout
at trick_pos 0. This proves **opponent simulation is not the cause** of the failures.
V_head itself has distribution shift at the immediately-post-play state (a state one play
further into the trick than V_head was trained on). The distribution shift is the primary
bottleneck. (commit message @ 7d2af99)

## Bugs found during mode comparison

### Bug 5 — per-world game_hands (e4e6862)

When opponent tokens were built for LAMIR-1 rollout, the slot→domino lookup used the real
deal hands. But the slot was chosen by π_me against a world hand that may have a different
domino at that slot, creating an incoherent (tokens, world_assign) pair. Fix: precompute
per-world 4-player game_hands (`_world_game_hands`) and pass them through all
`_build_tokens_voids` calls inside `lamir1_decision`. (commit message @ e4e6862)

### Fix 2 — sign-flip V_head by team parity (566bc4d)

V_head is trained in the leaf current_player's team frame. When the leaf player is on the
opponent team relative to the original decision player P, the value must be negated before
argmax. v-bootstrap: sign computed once (leaf_cp = (P+1)%4). lamir1: sign computed per
action_slot from states[0].current_player. (commit message @ 566bc4d)

## Modes added at end of G8 cluster

- **q-bootstrap** (8544fbe): depth-1 world-conditioned Q_head, no opp step. Tests whether
  world-conditioned Q_head beats world-blind V_head at look-ahead.
- **lamir1-qleaf** (fb03970): full opp simulation + Q_head leaf from trick-winner's POV.
  Rotates worlds to winner's frame, takes max over legal Q values, sign-flips if winner is
  on opp team. Results in later ingest.

## Links

[[gus]] · [[topics/lamir1]] · [[topics/pimc]] · [[experiments/gus-lamir1-pilot]] · [[topics/regret-eval]] · [[topics/lamir1-ceiling]]
