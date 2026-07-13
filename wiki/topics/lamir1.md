---
title: LAMIR-1 (continual-resolving look-ahead)
kind: topic
first_seen: 2026-04-20
last_updated: 2026-04-22
status: superseded
---

## Overview

LAMIR (Look-Ahead Monte Carlo with Information Re-weighting) is a 2025 technique for doing look-ahead during imperfect-information game play by sampling hidden-state worlds, evaluating each via a neural Q head, and aggregating by the current belief posterior. "LAMIR-1" is the depth-1 variant: a single look-ahead step (31e10ef).

**Superseded.** Every rollout mode built here was measured against direct π_me and lost — see [[lamir1-ceiling]] for the full ladder and root cause. The project took pivot option 4 (self-play/value-native, no CFR+), which became [[w42-jud-v1]]/[[champion]], not a further LAMIR refinement. This page is the frozen historical record of the depth-1 attempt, not an open frontier.

## Why it matters for Gus

Standard neural play runs a forward pass on the visible state and reads off the policy head. This ignores hidden information. [[lamir1]] enables look-ahead over hidden states at inference **without any oracle calls** (31e10ef):

1. Sample a set of hidden-state worlds from the **belief head** (which has learned `P(domino ∈ seat | visible state)`).
2. For each world, fuse the visible state + the sampled world and evaluate via the **Q head** — producing `Q(state, world)`.
3. Re-weight the per-world Q values by their belief probability.
4. Take the action with highest re-weighted expected Q.

This is continual-resolving: it can be run at every decision point during a game, not just at the root, because the belief and Q heads are always available (31e10ef).

## The LAMIR-critical piece

The Q head on fused `(state, world)` input is what makes LAMIR-1 possible. During training, the [[forge]] oracle generates `(world_hands, q_per_world)` tensors per decision (joint-world tensors, [[joint-world-tensor]]). The Q head learns to reproduce `q_per_world` from `(state, world)` input. At inference, no oracle is called — the Q head replaces it (31e10ef).

Tire-kick validation on seed 900000 (31e10ef):
- Top-10 (domino, seat) correlations converge by M~100-500 samples, settle at r≈0.2-0.4 for the strongest belief-Q relationships.
- Q-std is position-intrinsic and stable across sample sizes (~23 at decision 0 regardless of M).
- SEM<0.5 adaptive sampling converges in ~10s/game on MPS.

## Rollout modes

Four modes implemented in `gus/eval/lamir1.py`, selectable via `--mode` (7d2af99, 8544fbe, fb03970):

**direct** — baseline. No look-ahead; runs π_me argmax at the root. Used as the comparison baseline (regret 0.551, bot-match 76.07% on v3-10k).

**v-bootstrap** — depth-1 look-ahead with V_head leaf evaluator, no opponent simulation. For each candidate action, scores the resulting state by averaging V_head over M corpus worlds.
- Result: regret 2.777, bot-match 59.46% as first measured; corrected to **regret 1.645** after the eval fix — see [[gus-lamir1-mode-comparison]]. Still loses to direct.
- Per-trick-pos breakdown (pre-fix run): pos 0 (leads) 7.41 regret / 22% bot-match; pos 3 (last follower) 0.43 regret / 82% — same as direct.
- V_head at depth-1 is as broken as full rollout at trick_pos 0. Opp simulation is not the cause — V_head has distribution shift at the immediately-post-play state (7d2af99).

**lamir1** (original mode) — 1-ply rollout: simulate remaining trick plays using rotation-equivariant π_me as π_opp, then score the leaf with V_head averaged over M worlds.
- Result: regret 2.384, bot-match 62.86%
- Damage concentrated at trick_pos 0-2. Trick_pos 3 falls back to direct and is unchanged.
- Argmax π_me as π_opp creates adversarial leaf states that don't reflect oracle-equilibrium play; V_head correctly evaluates those worse leaf states — causing action scores to diverge from E[Q] ordering (581bf1f).

**q-bootstrap** — depth-1 look-ahead using world-conditioned Q_head instead of V_head. For each candidate action, rotates world to next actor's POV, runs Q_head, takes max over next actor's legal slots per world, sign-flips if opponent team, averages over M worlds. Tests whether world-conditioned Q_head beats world-blind V_head at depth-1 (8544fbe).

**lamir1-qleaf** — full 1-ply rollout (same opp simulation as lamir1) but with Q_head as the leaf evaluator instead of V_head. At the end-of-trick leaf, evaluates from the trick-winner's POV (world-conditioned), sign-flips if winner is on opponent team relative to original decision player (fb03970).

**lamir1-piopp** — full 1-ply rollout identical to lamir1-qleaf but opponent steps use the trained [[pi-opp-head]] (`PiOppHead`, seat-relative embedding) instead of rotated π_me. Requires `--pi-opp-adapter PATH`; optional `--pi-opp-sample` uses τ=1.0 sampling. Result: regret 2.268, bot-match 62.10% — slightly worse than lamir1-qleaf (2.006), confirming the bottleneck is the leaf evaluator rather than opp simulation quality (1a1a324, b42669a). See [[lamir1-ceiling]].

## Bugs fixed during G8

**Bug 5: per-world game_hands for opp simulation** (e4e6862) — when building opponent tokens for rollout, slot→domino lookup was using the real deal hands. But the slot was chosen by π_me against a world hand that may have a different domino at that slot, creating an incoherent `(tokens, world_assign)` pair. Fix: precompute per-world 4-player `game_hands` (`_world_game_hands`) and pass them to all `_build_tokens_voids` calls inside `lamir1_decision`.

**Fix 2: sign-flip V_head by leaf-player team parity** (566bc4d) — V_head is trained in the leaf `current_player`'s team frame. When the leaf player is on the opponent team relative to the original decision-player P, the value must be negated before argmax to convert to P's frame. For v-bootstrap: `leaf_cp = (P+1)%4`, sign computed once before the action loop. For lamir1: `leaf_cp` read from `states[0].current_player` after full rollout, sign computed per action_slot.

Both bugs are silent — they produce plausible-looking numbers without crashing. Auditing the rendered game_hands and checking team parity before scoring are the invariants to test for any future rollout implementation.

**Bug 6: stale world_assignment at Q_head leaf** (2c380a6) — after 1-3 opp rollout plays, the `world_assignment` tensor passed to the leaf Q_head still reflected the pre-rollout hand layout. Dominoes played out of world hands during the rollout loop remained marked as present in opp hands, giving Q_head stale world state. Fix: record the domino ID played by each opp in each world during the rollout loop; zero those rows in `world_assign_leaf` before the Q_head call. Applied to both `lamir1_qleaf_decision` and `lamir1_piopp_decision`.

**Fix 6 outcome**: The zeroing made every rollout mode slightly worse (e.g., lamir1-piopp + Fix 6 = 2.350 vs 2.268 without). The training convention in `dataset_seq_world` preserves the original `world_assignment` as the `played_mask` advances — our "fix" broke an invariant the Q_head relied on. The bug was in the hypothesis: the model was trained with full original assignment, so depleting it at inference is OOD (2c380a6, b42669a).

## Links

[[gus]] [[joint-world-tensor]] [[expected-q-value]] [[student-distillation]] [[forge]] [[pimc]] [[qmae-plateau]] [[pi-opp-head]] [[lamir1-ceiling]] [[w42-jud-v1]] [[champion]]
