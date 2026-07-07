---
title: Gus Belief Calibration Diagnostic (receipt 15)
kind: experiment
first_seen: 137a8e7
last_updated: 137a8e7
status: active
---

## Summary

Frozen-trunk fine-tune of the belief head using a distribution target (empirical
marginal over M sampled worlds) instead of the standard one-hot truth. Belief
calibration improves measurably; nothing downstream benefits. Not promoted — ablation
result kept in `scratch/`. (commit message @ 137a8e7, PRACTICALITIES receipt 15)

## Setup

- **Base adapter**: `v2_voids_3000g_big`
- **Intervention**: frozen trunk; only belief_head weights updated
- **Target**: empirical marginal distribution over M sampled worlds (not one-hot true answer)
- **Rationale**: one-hot targets may over-sharpen belief; distributional target softens
  predictions toward the marginal, potentially improving calibration for downstream PIMC

## Results

### Belief head (isolated)

| Metric | Before | After |
|---|---|---|
| truth top-1 | 38.4% | 38.3% (unchanged) |
| KL vs world-marginal | 0.078 | 0.062 |
| (uniform-belief baseline KL) | 0.081 | — |

Calibration is real — distribution training closes ~47% of the gap between the
uniform-prior baseline (0.081) and hypothetical perfect belief (0). Truth-trained
belief was optimizing for the mode, not the shape. (gus/PRACTICALITIES.md receipt 15)

### Downstream (where it matters)

| Metric | Before | After |
|---|---|---|
| pimc-belief K=50 | 66.4% | 65.4% (−1pp, regressed) |
| blunder detector ROC-AUC | 0.792 | 0.793 (flat) |
| blunder detector PR-AUC | 0.175 | 0.133 (regressed) |

## Why calibration didn't propagate (receipt 15)

Q_head was co-trained alongside mode-sharp belief samples. At inference, distributional
(softened) belief samples create a distribution shift that Q_head was never prepared for.
The world_encoder + Q_head pipeline expects the sharper belief samples it saw during
training; softer samples produce noisier Q estimates that hurt downstream decisions.
(commit message @ 137a8e7)

## Takeaway

Stacking calibration onto a frozen ecosystem doesn't work. To benefit from distributional
belief targets, the full {belief_head, world_encoder, Q_head} cluster must be co-trained
together with distribution-belief targets and regular Q loss. Sequential fine-tuning
creates irreconcilable distribution shift. (commit message @ 137a8e7)

## Links

[[gus]] · [[topics/regret-eval]] · [[topics/dense-q-supervision]] · [[joint-world-tensor]]
