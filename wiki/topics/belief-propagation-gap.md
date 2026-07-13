---
title: Belief Propagation Gap (calibration doesn't transfer to Q)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-04-22
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

Substantially answered the next day by [[belief-co-train]] (`cf8ff79`): the mechanism is
distribution-shift/task-interference, not a coupling deficiency. Joint co-training of belief +
world_encoder + Q_head with a distribution target did make calibration propagate as designed
(KL dropped 20%, 0.0840 → 0.0672) — but downstream q-bootstrap regret still got slightly
*worse* (0.685 → 0.718), because Q_head had learned the old belief head's output shape as
implicit context and joint retraining moved it off that sweet spot. So the takeaway below
holds, with one caveat co-train sharpened: co-training fixes the *propagation* problem but
does not, by itself, guarantee a net win — the Q_head's dependence on a specific upstream
shape is itself the residual issue.

## Takeaway

Fine-tuning individual heads in isolation does not work when the heads were co-trained. Improving belief calibration requires co-training `{belief, world_encoder, Q_head}` together with a distribution-belief target alongside the regular Q loss — so that downstream heads learn the new calibration from the start. Stacking calibration on a frozen ecosystem creates distribution mismatch (137a8e7). [[belief-co-train]] confirmed the propagation mechanism but found the co-trained result was net-neutral-to-negative on regret, not a clean win.

This result motivates later co-train experiments. The ablation was not promoted; kept in `scratch/` (137a8e7).

## Links

[[gus]] [[student-distillation]] [[dense-q-supervision]] [[pimc]] [[blunder-detector]] [[joint-world-tensor]] [[belief-co-train]]
