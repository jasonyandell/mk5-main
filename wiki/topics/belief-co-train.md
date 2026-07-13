---
title: Belief Co-Train Experiment
kind: topic
first_seen: 2026-04-22
last_updated: 2026-07-13
status: complete
---

## Motivation

After [[belief-bayes-ceiling]] showed top-1 accuracy is at the Bayesian limit, the
remaining lever is posterior shape (calibration). [[gus-belief-calibration-diagnostic]]
(§15) showed distribution-target training improves calibration on a frozen ecosystem,
but downstream play didn't improve because the consuming heads (world_encoder, Q_head)
weren't co-trained. The proposed fix: train belief + world_encoder + Q_head jointly with
a distribution target so that calibration propagates to look-ahead value estimates
(cf8ff79, PRACTICALITIES §21).

## What happened

Full setup and numbers on [[gus-belief-co-train]]. Two findings:

1. **The co-train hypothesis is falsified on this setup.** Belief KL improved (−20%) as
   designed, but downstream q-bootstrap regret got slightly worse (0.685 → 0.718).
   Q_head had learned the original belief head's output distribution as implicit
   context; joint retraining moved it off that sweet spot. In a distillation pipeline,
   "better upstream head" is not strictly additive when downstream heads were trained
   against the old shape.
2. **Unexpected win — q-bootstrap-belief.** Sampling worlds from the belief head at
   inference (rather than reading the oracle corpus) gives regret 0.655 on the original
   adapter — beating corpus worlds (0.685) and coming closest of any look-ahead to the
   0.551 direct baseline. Likely mechanism: oracle adaptive sampling overconcentrates on
   a few high-posterior worlds; belief-head softmax sampling is smoother and matches the
   distribution Q_head saw in training.

## Why the second finding outlived the line

q-bootstrap-belief — belief-sampled worlds beat corpus worlds — is the standing evidence
[[champion]] cites for wiring the Gus belief posterior into oracle world sampling
(`--mode q-bootstrap-belief` in `gus/eval/lamir1.py`, reusing
`gus/model/sample_worlds.py`). The belief-conditioned marginalization idea carried into
[[belief-conditioned-self-play]] and [[jud]].

## Links

[[belief-bayes-ceiling]] [[belief-propagation-gap]] [[gus-belief-co-train]] [[gus-belief-calibration-diagnostic]] [[lamir1-ceiling]] [[gus]] [[gus-line]] [[champion]]
