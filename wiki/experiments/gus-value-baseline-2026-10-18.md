---
title: "Gus: a success-value baseline on the frozen representation"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-19
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Finding and scope

A 65-parameter linear readout of [[gus-tiny-gpt]]'s original frozen pretrained
representation predicts terminal hand success better than fitted constant,
role-by-turn and matched random-feature controls. The registered criterion
passes. No actor changes and no new games are required for this diagnostic.
This supports testing a value baseline in PPO; it does not establish lower
policy-gradient variance, an exact success probability or stronger play.

The source is **training** trajectories from [[gus-exposure-budget-2026-10-18]]:
1,024 physical deals, eight draws each, seven learner decisions per draw.
All 128 changing learner snapshots occur in both partitions. The resulting
pooled predictor is not an on-policy critic. These roots are held out from
value fitting, not fresh playing-strength confirmation.

## Registered method

Within every eight-root collection round, two physical roots are hash-reserved
before fitting: 768 fit roots and 256 diagnostic test roots. All draws and
prefixes of a root stay together. Only the 28,714 pre-public-settlement fit
states enter the regression. No exposure confirmation outcomes are used.

The original 185,652-parameter splat parent remains frozen. Its final normalized
64-dimensional actor-position representation is read from the unchanged own-seven
and public-history tokens. A random encoder of the same architecture, seed
91044, supplies the representation control. Causal capture is checked by changing
all padding after each prefix; hidden hands and outcomes never enter the encoder.

Each head fits mean squared binary success error plus a fixed 0.001 ridge penalty
on standardized nonintercept weights. Mean and standard deviation use fit rows
only; the scale floor is 1e-8. The intercept is unpenalized. The float64 solve
does no hyperparameter selection. Predictions are clipped to [0,1] for Brier
error and calibration; raw MSE and clipping remain in the report.

The simple controls are the fit-wide success mean and fit-only means by learner
turn × bidder-team membership. The primary slice is unsettled discretionary
decisions. The criterion requires at least 0.01 lower Brier error than both
priors, with paired 95% upper delta below zero against both priors and random
features. Uncertainty resamples physical roots, 2,000 times, seed 91045.

## Held-out predictions

The primary slice contains 6,579 decisions on the reserved roots. Lower Brier
error is better; it is a forecast error, not win rate.

| Predictor | Brier error |
|---|---:|
| Pretrained representation | **0.206561** |
| Random representation | 0.221284 |
| Constant prior | 0.225641 |
| Role × turn prior | 0.228811 |

| Pretrained minus control | Paired mean delta | 95% interval |
|---|---:|---:|
| Constant | −0.019079 | [−0.033473, −0.004324] |
| Role × turn | −0.022249 | [−0.034488, −0.009611] |
| Random | −0.014722 | [−0.027724, −0.001799] |

The pretrained head also has lower point-estimate error in the registered
opening, middle and late slices. The late-slice interval against random spans
zero. Clipping affects 2.17% of primary-slice predictions. The full report owns
slice counts, calibration bins, raw scores and all intervals.

## Verification and recovery

Extraction independently reconstructs clean inputs and terminal labels from all
8,192 source hands. An independent augmented least-squares calculation reproduces
normalization, priors, predictions and scores; coefficient maximum differences
are 5.38e-14 for pretrained features and 2.90e-12 for random features.

An actual SIGTERM pauses after three completed chunks. Resume preserves their
bytes and completes all 128 chunks. Re-extracting rounds 128, 64 and 1 under the
pinned CPU runtime and microbatch 112 gives identical numerical arrays. Completed
fit resume preserves the fit, prediction and report bytes. Four focused tests
cover root grouping, independent fitting, causal capture and atomic chunks.
This is not a cross-runtime or arbitrary-batch identity claim.

## Subsequent comparison

[[gus-success-probability-planning]] selects a matched PPO/value versus GRPO and
rehearsal comparison. `lem/observer/value.py` implements a separate linear
baseline on the frozen reference representation; the actor's inputs and size
stay unchanged. Complete terminal returns minus old values provide advantages,
the undiscounted GAE λ=1 case. The current batch fits the critic only after its
actor update. The diagnostic regression is not used as a warm start.

PPO implementation and recovery are engineering work distinct from this completed
prediction study. The subsequent registered [[gus-ppo-value-2026-10-19]] comparison establishes
no full-hand gain. Its freshly fitted eight-root online critic forecasts worse
than a simple prior before fitting the next batch. This differs from the pooled
standardized ridge fit here; the representation result does not validate that
online fitting recipe. Other seats remain fixed.

## Implementation verification

The subsequent PPO implementation passes all 154 Lem tests and the changed-suite
wiki, license and conflict-marker gates. CPU tests interrupt both actor and critic
optimizers with actual SIGTERM and simulated crashes; full recovery reproduces
weights, optimizer states, RNG and selection. Tests also check future-padding
causality, unchanged captured advantages, explicit recipes and a policy update
that differs from matched rehearsal.

A real MPS smoke uses eight previously used exposure training roots, the original
parent and fixed G0/Zeb seats. Two rounds complete before and after interruption.
The **strict actor bitwise comparison fails**: the largest final weight delta is
2.08663e-6, and the largest actor-optimizer tensor delta is 3.63216e-8. Critic
weights/optimizer and RNG states match exactly; all training trajectory bytes
agree. Completed resume preserves artifact bytes. The original failed check is
retained beside a descriptive follow-up audit; no relaxed passing criterion
replaces it. MPS training is not promised bitwise. Lazy frozen evaluation keeps
its separate pinned CPU execution and strict byte-recheck contract.

## Public evidence

- [Preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/933d19abcaa7e7726ff97a1c12d147417a31db37/lem/gus-value-baseline-2026-10-18/registration)
  precedes extraction and fitting; registration SHA256
  `a3da07f1cfdd437cf16468c2d96cae78e142c2969d47dd41828fa3b0b50fc5ce`.
- [Completed evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/b799cd1be36cfcbe48ee2506156d177c725070b1/lem/gus-value-baseline-2026-10-18):
  15 files, 19,710,208 bytes, 290 archive members; every fresh anonymous file
  download and archive member matches its SHA256. Includes frozen parent,
  random twin, feature chunks, heads, audit, recovery and restore map.
- Execution source: `dd78db6a3092450ae216e32c8f7c65f66e2e9fdf`.
  Local evidence: `scratch/lem/gus-value-baseline-v1`.
- Full source trajectories remain in the exposure study's immutable archive,
  revision `b1e119fa560dbe714a4d2310a9d3367a1cfa9ec7`.
- [Separate implementation evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/2921ba5b99e7d8b6421d6441658a515314b2869c/lem/gus-value-baseline-2026-10-18/ppo-implementation):
  9 files, 15,865,868 bytes, 56 archive members, all fresh anonymous downloads
  and members SHA256-verified. Includes tests, source hashes, gate exit receipt,
  failed MPS strict check, recovery checkpoints and its descriptive audit.
  These are engineering specimens, not a promoted player.
