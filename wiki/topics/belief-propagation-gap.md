---
title: Belief Propagation Gap (calibration doesn't transfer to Q)
kind: topic
first_seen: 137a8e7
last_updated: 137a8e7
status: active
---

## Overview

The belief propagation gap is a diagnostic finding from [[gus]] G6: the belief head can be measurably improved in calibration by fine-tuning, but this improvement does not propagate to downstream heads (Q, [[pimc]]) or to the [[blunder-detector]]. Better beliefs don't produce better plays (137a8e7).

## The experiment

Frozen-trunk fine-tune of the belief head with a distribution target (empirical marginal over M sampled worlds, not one-hot truth). The belief head was trained to output a softer, better-calibrated probability distribution.

Results (PRACTICALITIES receipt 15, 137a8e7):

| Metric | Before | After | Delta |
|---|---|---|---|
| Belief truth top-1 | 38.4% | 38.3% | flat |
| KL vs world-marginal | 0.078 | 0.062 | −21% (47% closer to perfect) |
| PIMC-belief K=50 | 66.4% | 65.4% | −1pp (regressed) |
| Blunder detector AUC | — | flat | — |
| Blunder detector PR-AUC | — | — | regressed |

The calibration is real — KL improved substantially. The downstream effect is zero or negative.

## Why calibration doesn't propagate

The Q_head was co-trained with mode-sharp (uncalibrated) belief samples. At inference, softening the belief output creates a distribution shift that the Q_head was not prepared for. The Q_head learned to interpret belief outputs of a specific sharpness; feeding it softer distributions degrades its performance (137a8e7).

Possible mechanisms (not fully diagnosed at this frontier):
- **Distribution shift**: Q_head internalized the mode-sharp belief distribution during co-training.
- **Encoder bottleneck**: the shared encoder may not surface belief information in a form the Q_head actually uses.
- **Task interference**: belief and Q gradients may be competing through the shared encoder.

## Takeaway

Fine-tuning individual heads in isolation does not work when the heads were co-trained. Improving belief calibration requires co-training `{belief, world_encoder, Q_head}` together with a distribution-belief target alongside the regular Q loss — so that downstream heads learn the new calibration from the start. Stacking calibration on a frozen ecosystem creates distribution mismatch (137a8e7).

This result motivates later co-train experiments. The ablation was not promoted; kept in `scratch/` (137a8e7).

## Links

[[gus]] [[student-distillation]] [[dense-q-supervision]] [[pimc]] [[blunder-detector]] [[joint-world-tensor]]
