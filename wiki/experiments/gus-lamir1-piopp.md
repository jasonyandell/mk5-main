---
title: LAMIR-1 with π_opp Rollout — Ceiling Finding (§20)
kind: experiment
first_seen: 1a1a324
last_updated: b42669a
status: superseded
---

## Summary

Full LAMIR-1 attempt with trained π_opp rollout + Q_head leaf evaluator. All 8 look-ahead
variants fail to beat direct π_me baseline (0.551 regret). Root cause identified: Q_head
is OOD at depleted post-rollout leaf states. PRACTICALITIES §20 documents the ceiling and
four pivot options. (commit messages @ 1a1a324, 2c380a6, 8106f01, b42669a)

## Setup

- **Adapter**: v3_consistency_10k (baseline 0.551 regret, 76.07% bot-match)
- **π_opp**: PiOppHead at 68.6% oracle accuracy (see [[experiments/gus-pi-opp-training]])
- **Mode**: lamir1-piopp — same rollout as lamir1-qleaf but opp steps use trained PiOppHead
  instead of rotated π_me; leaf scored by Q_head from trick-winner's POV

## Complete mode ladder (560 held-out decisions, §20)

| Mode | Regret | Bot-match | Notes |
|---|---|---|---|
| **direct π_me (baseline)** | **0.551** | **76.07%** | v3-10k |
| q-bootstrap | 0.679 | — | Best look-ahead mode |
| v-bootstrap | 1.645 | 66.96% | V_head distribution shift |
| lamir1 (argmax opp) | 2.094 | 60.36% | — |
| lamir1-qleaf | 2.006 | 64.29% | Full rollout, Q_head leaf |
| lamir1-piopp | 2.268 | 62.10% | π_opp rollout, Q_head leaf |

Direct π_me beats all 8 look-ahead variants. q-bootstrap (0.679) is the best look-ahead
mode but still 23% worse than direct. Full rollouts uniformly worse than depth-1.
(commit message @ b42669a)

**Number correction (2026-07-06 audit):** this page cites the right commit (`b42669a`) but had
been carrying the wrong (pre-fix) numbers from it — v-bootstrap was previously listed as 2.777
and lamir1 as 2.384, both superseded values from earlier in the same bug-fix cluster. The
table above now matches `b42669a:PRACTICALITIES.md` §20 and [[lamir1-ceiling]]'s canonical
ladder exactly.

## Bug 6 — stale world_assign at Q_head leaf (2c380a6)

After 1-3 opp rollout plays, the world_assignment tensor passed to the leaf Q_head
still reflected the pre-rollout hand layout — dominoes played out of world hands were
still marked as present. Fix: during the rollout loop, record each opp's played domino
ID per world; after the loop, zero those rows in `world_assign_leaf` before the Q_head
call. Matches the training convention in `dataset_seq_world._world_to_assignment`.
Applied to both `lamir1_qleaf_decision` and `lamir1_piopp_decision`. (commit message @ 2c380a6)

## Root cause: Q_head OOD at depleted leaf states

Q_head was trained on initial-deal world assignments (full hands). Post-rollout leaf
states have partially depleted hands (1-3 dominoes played). This is out-of-distribution
for Q_head — it was never trained on partially-depleted world layouts. Bug 6 correctly
identifies and fixes the data invariant violation, but does not cure the underlying
distribution shift: the training data simply never contained depleted states. (commit
message @ b42669a)

Deeper issue per §20: LAMIR paper's T×T multi-valued-states value function is more
expressive than Gus's scalar V/Q heads. Scalar distillation noise compounds across
rollout steps and overwhelms any leaf evaluator signal.

## §20 pivot options

Four paths documented in PRACTICALITIES §20:

1. **Retrain Q_head with partial-depletion augmentation** — generate training data that
   includes partially-played-out hands at various trick depths
2. **End-to-end LAMIR training** — train the full pipeline jointly rather than distilling
   each head separately
3. **Abandon look-ahead; invest in data** — more corpus for direct π_me, which still
   dominates all look-ahead variants
4. **Focus on detect-and-route** — oracle fallback on flagged decisions (already shown to
   reach 0.49 regret; see [[experiments/gus-router-pilot]])

## π_opp as real side product

Despite the LAMIR-1 ceiling, the 68.6%-accurate π_opp head is a genuine deliverable —
a supervised opponent model conditioned on seat identity and observable state. (commit
message @ 8106f01)

## Links

[[gus]] · [[topics/lamir1]] · [[topics/lamir1-ceiling]] · [[topics/regret-eval]] · [[experiments/gus-pi-opp-training]] · [[experiments/gus-lamir1-mode-comparison]] · [[experiments/gus-router-pilot]]
