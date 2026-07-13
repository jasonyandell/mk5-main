---
title: Gus Q_head Partial-Depletion Augmentation (Path A)
kind: experiment
first_seen: 2026-04-22
last_updated: 2026-04-22
status: complete
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
- **Script**: `gus/train/train_q_head_augmented.py` (15 epochs, lr=5e-5, 160,007 trainable params; adapter saved as `gus/adapters/q_head_aug.pt`)

## Result

| Mode | Regret | Notes |
|---|---|---|
| Direct π_me (baseline) | 0.551 | Unchanged |
| lamir1-qleaf + Bug6 (no aug) | 2.156 | Direct comparison baseline |
| lamir1-qleaf + Bug6 + aug Q_head | 2.216 | Path (a) result |

2.216 vs 2.156: slightly worse, and the 0.06 regret difference is noise, not signal
either way. Eval q_mae improved marginally (8.277 → 8.169), but that didn't translate
to better rollout decisions. (gus/MORNING4_STATUS.md @ 5f390fb)

## Path (a) postmortem

OOD augmentation fixes a measurement artifact (q_mae on depleted inputs) but not the
fundamental ordering problem: scalar noise from distillation is large relative to the
action-value gap at decision boundaries. Augmenting the input distribution alone is
insufficient — the leaf evaluator would need to be trained end-to-end in the rollout
context (path b) to fix this. (gus/MORNING4_STATUS.md @ 5f390fb)

## Conclusion

Path (a) is closed. Random-depletion augmentation is insufficient — it fixes the OOD
measurement artifact (q_mae on depleted inputs) but not the ordering problem at decision
boundaries, where scalar distillation noise swamps the action-value gap. The remaining
path for LAMIR-1 look-ahead is training the leaf evaluator end-to-end in the rollout
context (path b), a substantially larger project. (gus/MORNING4_STATUS.md @ 5f390fb)

## Links

[[gus]] · [[experiments/gus-lamir1-piopp]] · [[experiments/gus-lamir1-mode-comparison]] · [[topics/lamir1]] · [[topics/regret-eval]] · [[topics/q-head-augmentation]]
