---
title: LAMIR-1 with π_opp Rollout — Ceiling Finding (§20)
kind: experiment
first_seen: 1a1a324
last_updated: b42669a
status: superseded
---

## Summary

Full LAMIR-1 attempt with trained π_opp rollout + Q_head leaf evaluator. All 8 look-ahead
variants fail to beat direct π_me baseline (0.551 regret). Root cause identified: distilled
scalar V/Q noise flips argmax at decision boundaries — the failure mode Kubíček & Lisý warn
about. PRACTICALITIES §20 + MORNING4_STATUS document the ceiling and four pivot options. (commit messages @ 1a1a324, 2c380a6, 8106f01, b42669a)

## Setup

- **Adapter**: v3_consistency_10k (baseline 0.551 regret, 76.07% bot-match)
- **π_opp**: PiOppHead at 68.6% oracle accuracy (see [[experiments/gus-pi-opp-training]])
- **Mode**: lamir1-piopp — same rollout as lamir1-qleaf but opp steps use trained PiOppHead
  instead of rotated π_me; leaf scored by Q_head from trick-winner's POV

## Complete mode ladder (560 held-out decisions, §20)

| Mode | Regret | Bot-match | Notes |
|---|---|---|---|
| **direct π_me (baseline)** | **0.551** | **76.07%** | v3-10k |
| q-bootstrap | 0.679 | 72.50% | Best look-ahead mode |
| v-bootstrap | 1.645 | 66.96% | Depth-1 V_head, world-blind |
| lamir1-qleaf | 2.006 | 64.29% | Full rollout (rotated π_me), Q_head leaf |
| lamir1 | 2.094 | 60.36% | Full rollout (rotated π_me), V_head leaf |
| lamir1-piopp | 2.268 | 62.10% | π_opp rollout, Q_head leaf |
| lamir1-piopp + Fix 6 | 2.350 | 62.50% | Assignment-update "fix" made it worse |

Direct π_me beats all 8 look-ahead variants. q-bootstrap (0.679) is the best look-ahead
mode but still 23% worse than direct. Full rollouts uniformly worse than depth-1.
(commit message @ b42669a)

**Number correction (2026-07-06 audit):** this page cites the right commit (`b42669a`) but had
been carrying the wrong (pre-fix) numbers from it — v-bootstrap was previously listed as 2.777
and lamir1 as 2.384, both superseded values from earlier in the same bug-fix cluster. The
table above now matches `b42669a:PRACTICALITIES.md` §20 and [[lamir1-ceiling]]'s canonical
ladder exactly.

## Bug 6 / Fix 6 — the stale-world_assign hypothesis that made things worse (2c380a6)

Hypothesis: after 1-3 opp rollout plays, the world_assignment tensor passed to the leaf
Q_head still reflected the pre-rollout hand layout, giving Q_head stale beliefs. Fix 6
recorded each opp's played domino ID per world and zeroed those rows in
`world_assign_leaf` before the Q_head call, applied to both `lamir1_qleaf_decision` and
`lamir1_piopp_decision`. (commit message @ 2c380a6)

Result: regret ticked UP across all modes (lamir1-piopp 2.268 → 2.350). The training
convention in `dataset_seq_world` actually preserves the original world_assignment as
the played_mask advances — the "fix" broke an invariant the Q_head relied on. The bug
was in the hypothesis, not the model. (PRACTICALITIES §20 @ b42669a)

## Root cause: distilled scalar V/Q noise flips argmax

Per §20, Kubíček & Lisý explicitly warn that a value function trained by distillation
(like Gus's V_head) cannot be used for look-ahead reasoning. The scalar noise of the
distilled V/Q heads is enough to flip argmax at decision boundaries, while π_me trained
on argmax directly preserves ordering. V_head is also architecturally world-blind
(std=0.000 across 200 world samples for the same decision). (commit message @ b42669a)

Deeper issue per §20: LAMIR paper's T×T multi-valued-states value function is more
expressive than Gus's scalar V/Q heads. Scalar distillation noise compounds across
rollout steps and overwhelms any leaf evaluator signal.

## §20 pivot options

Four paths documented in MORNING4_STATUS (commit b42669a):

1. **Accept depth-1 ceiling; ship q-bootstrap as an alternative inference mode** — +25%
   regret vs direct but world-conditioned; could be a second opinion in a router when
   π_me entropy is high (see [[experiments/gus-router-pilot]])
2. **Train a look-ahead-compatible V-head** — train V on expected Q under sampled opp
   play rather than marginal oracle E[Q]
3. **Implement the LAMIR paper faithfully** — multi-valued states + CFR+ solver; large,
   research-grade effort
4. **Bridge-AI / BMCS recipe** — PPO self-play to reshape V/Q heads; the π_opp head and
   world-conditioned Q_head are the raw materials. Gus doesn't have to be LAMIR.

## π_opp as real side product

Despite the LAMIR-1 ceiling, the 68.6%-accurate π_opp head is a genuine deliverable —
a supervised opponent model conditioned on seat identity and observable state. (commit
message @ 8106f01)

## Links

[[gus]] · [[topics/lamir1]] · [[topics/lamir1-ceiling]] · [[topics/regret-eval]] · [[experiments/gus-pi-opp-training]] · [[experiments/gus-lamir1-mode-comparison]] · [[experiments/gus-router-pilot]]
