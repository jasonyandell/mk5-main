---
title: Belief Co-Train Experiment
kind: topic
first_seen: 2026-04-22
last_updated: 2026-04-22
status: active
---

## Motivation

After [[belief-bayes-ceiling]] showed top-1 accuracy is at the Bayesian limit, the remaining lever is posterior shape (calibration). §15 showed distribution-target training closed 47% of the KL gap on a frozen ecosystem, but downstream play didn't improve because the consuming heads (world_encoder, Q_head) weren't co-trained. The proposed fix: train belief + world_encoder + Q_head jointly with a distribution target so that calibration propagates to look-ahead value estimates (cf8ff79, PRACTICALITIES §21).

## Setup

- **Trainer**: `gus/train/train_belief_q_joint.py`
- **Frozen**: trunk (encoder), π_me, V_head
- **Trainable**: belief head + world_encoder + Q_head
- **Loss**: α=1 belief KL + β=1 Q_head qMAE
- **Epochs**: 15, 1000-game v2 corpus
- **Result adapter**: `gus/adapters/v3_belief_q_joint.pt`

## Results

**Belief KL dropped 20%**: 0.0840 → 0.0672 (epoch 11). Top-1 stayed at ~38% (consistent with [[belief-bayes-ceiling]] — top-1 can't improve). The calibration target worked as designed.

**But downstream regret got slightly worse:**

| adapter | q-bootstrap (corpus worlds) | q-bootstrap-belief (belief-sampled) |
|---|---:|---:|
| original v3_consistency_10000g | 0.685 | **0.655** |
| joint co-trained | 0.718 | 0.679 |

The hypothesis that "co-training heads lets calibration propagate to look-ahead" is falsified on this setup (cf8ff79).

**Diagnosis**: Q_head was trained with the original belief head's output distribution as implicit context. Joint retraining shifts the state_emb → Q_head mapping. The calibration improvement doesn't compensate — Q_head was moved off its sweet spot. In a distillation pipeline, "better upstream head" is not strictly additive when downstream heads were trained against the old shape.

## Unexpected win — q-bootstrap-belief mode

As a required A/B harness for the co-train experiment, a new inference mode `q-bootstrap-belief` was added to `gus/eval/lamir1.py`: worlds are sampled from the belief head at inference rather than read from the oracle's saved `world_hands` corpus.

On the **original** adapter (no co-train), belief-sampled worlds give **regret 0.655** vs corpus worlds 0.685 — a 4.4% relative improvement, and the closest any look-ahead variant has come to the 0.551 direct baseline (gap 0.104 Q-pts, ~19%) (cf8ff79).

Likely mechanism: oracle adaptive sampling (SEM<0.5) can overconcentrate on a few high-posterior worlds at near-consensus decisions. Belief-head softmax sampling is smoother and more aligned with the world distribution Q_head saw during training. Sampling from the learned belief head appears to be a better world source for look-ahead than the oracle corpus (PRACTICALITIES §21).

This mode reuses `gus/model/sample_worlds.py` — a file written off-plan during an earlier overnight run that turned out to be exactly what this follow-up needed.

## Artifact

`gus/eval/lamir1.py --mode q-bootstrap-belief` — usable technique for any future PIMC/BMCS/LAMIR work. K=200 belief-sampled worlds already matches or beats corpus at M~3000, suggesting the distribution is better rather than the diversity.

## Links

[[belief-bayes-ceiling]] [[belief-propagation-gap]] [[lamir1-ceiling]] [[consistency-regularizer]] [[student-distillation]] [[gus]] [[expected-q-value]]
