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

Calibration is real — the receipt's gloss is that distribution training "closes
about 47% of the gap between uniform-prior and a hypothetical perfect belief".
Truth-trained belief was optimizing for the mode, not the shape. The raw KL drop
0.078 → 0.062 is ~20%; the 47% figure only works against a nonzero KL floor
(~0.041, unstated in the receipt — plausibly the finite-M sampling floor of the
empirical marginal), not against 0. (gus/PRACTICALITIES.md receipt 15)

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

The co-training follow-up was run in §21 ([[experiments/gus-belief-co-train]], cf8ff79):
co-training worsened q-bootstrap regret slightly (0.685 → 0.718), falsifying the
"co-training propagates calibration" hypothesis on that setup — but sampling worlds
from belief at inference (q-bootstrap-belief, 0.655) became the closest look-ahead
to the direct baseline.

## Links

[[gus]] · [[topics/regret-eval]] · [[topics/dense-q-supervision]] · [[joint-world-tensor]] · [[experiments/gus-belief-co-train]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 2 corrections applied in place and independently re-verified; second pass amended 1.

- Training script `scratch/train_belief_distribution.py` (named in the receipt) was never committed and scratch/ is gitignored — the artifact is unrecoverable from the repo; "kept in scratch/" quotes the commit but the file is gone.
- Base adapter `v2_voids_3000g_big` is not named in receipt 15 itself (it appears as "best single" in the §14 router table); plausible as the standing promoted adapter, not directly verified.
