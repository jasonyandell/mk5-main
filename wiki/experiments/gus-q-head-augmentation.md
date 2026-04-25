---
title: Gus Q_head Partial-Depletion Augmentation (Path A)
kind: experiment
first_seen: a9fa0c6
last_updated: 5f390fb
status: active
---

## Summary

Path (a) from §20: fine-tune Q_head with partial-depletion augmentation so it becomes
robust to OOD depleted leaf states from LAMIR-1 rollouts. Augmented Q_head reaches
2.216 regret on lamir1-qleaf — far worse than direct baseline (0.551). Path (a) closed.
(commit messages @ a9fa0c6, 5f390fb)

## Setup

- **Base**: v3_consistency_10k (frozen encoder, belief, v_head, pi_me)
- **Trained**: q_head + world_encoder only
- **Augmentation**: with probability `aug_prob`, zero k random assigned domino rows from
  `world_assign` before the forward pass — simulating partially-depleted post-rollout states
- **Target**: `q_per_world` unchanged — Q should be invariant to whether played dominoes
  remain in the assignment tensor
- **Script**: `gus/train/train_q_aug.py`

## Result

| Mode | Regret | Notes |
|---|---|---|
| Direct π_me (baseline) | 0.551 | Unchanged |
| lamir1-qleaf with aug Q_head | 2.216 | Path (a) result |

2.216 regret: worse than the pre-bug-fix rollouts, not better. Augmentation failed to
teach the Q_head OOD robustness at post-rollout leaf states. (commit message @ 5f390fb)

## Path (a) postmortem

Augmentation zeros random domino rows from `world_assign`, but this is not the same
distribution shift that occurs during actual LAMIR-1 rollout. In the rollout, specific
dominoes are removed in a causally consistent order (played by opponents in world-specific
sequences). Random zeroing teaches the Q_head to be robust to arbitrary missing entries,
not to the structured depletion pattern of a real rollout. The training signal is
mismatched. (commit message @ 5f390fb)

## Conclusion

Path (a) is closed. Augmenting the existing Q_head with random depletion is insufficient —
the gap between training distribution and rollout distribution cannot be bridged by this
technique. The only viable path for LAMIR-1 look-ahead is end-to-end joint training with
actual rollout-generated leaf states, which is a substantially larger project.

## Links

[[gus]] · [[experiments/gus-lamir1-piopp]] · [[experiments/gus-lamir1-mode-comparison]] · [[topics/lamir1]] · [[topics/regret-eval]]
