---
title: "Gus: PPO with a learned success-value baseline"
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Does PPO with a learned success-value baseline improve [[gus-tiny-gpt]]'s
fixed-field full-hand play over matched GRPO and supervised rehearsal?
[[gus-value-baseline-2026-10-18]] supplies the training-only motivation: the
existing frozen representation predicts terminal success better than fitted
priors and random features. Prediction does not establish useful policy credit
or stronger play. [[gus-success-probability-planning]] owns the method choice.

## Registered comparison

Three arms start from the original 185,652-parameter splat actor: PPO/value,
GRPO and supervised rehearsal. Each receives the same 256 fresh physical roots,
eight draws per root, 32 rounds and 64 actor updates. Same original clean tokens,
bid30/pip contract, fixed G0 partner, pinned Zeb opponents and original frozen
KL reference. The recipe retains lr3e-5, clip .2, KL .02, rehearsal weight 1/
batch 32, eight roots per round, full seven-turn credit and evaluation every
eight rounds. Rehearsal collects trajectories but has no reward or KL gradient.

PPO uses complete binary returns minus old state values, normalized over the
batch. GRPO standardizes within each root's sampled group. This compares the
estimator packages; it does not isolate learned baselines from normalization.
The 65-parameter linear critic reads frozen reference features, starts with
zero weights/bias .5, and uses its own Adam at lr .001 for 20 steps after each
actor batch. The diagnostic regression is not a warm start. Old values and
advantages remain fixed throughout the two actor updates. No new actor input,
hidden information, external search or opponent is introduced.

Complete terminal returns use gamma 1 and lambda 1; there is no bootstrapped
target. PPO clipping and value regression are standard components; the frozen
linear critic and retained KL/rehearsal are explicit experimental choices.
Sources: [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html),
[GAE](https://arxiv.org/abs/1506.02438), [GRPO](https://arxiv.org/abs/2402.03300).

## Selection, confirmation and limits

The 128-root development panel selects greedy then sampled success within the
original tile-NLL/splat-CE bounds +.02 each. Initialization is eligible; ties
keep the earliest checkpoint. The development criterion requires selected PPO
to beat both selected controls by at least 2/128, with retention.

After all selections freeze, 512 untouched roots evaluate parent and the three
selected labels, greedy and one coupled sampled rollout. Exact identical
inference functions may share lazy records under the same pinned recipe.
Promotion requires PPO greedy gains of at least 2pp over parent and 1pp over
both controls, all paired 95% lower bounds above zero, with retention.
Intervals use 20,000 physical-root bootstrap draws, seed 91942. Sampled results
are secondary; confirmation cannot select another checkpoint. One seed and
recipe do not rank the algorithms generally.

Each arm has at most 16 cooperative 45-second segments with 90-second outer
caps, two live arms maximum. Complete-round recovery retains actor, critic,
optimizers and RNG. Prior CPU recovery is exact; the prior MPS smoke failed
strict actor equality, so GPU training has no bitwise promise. Frozen lazy
evaluation retains its separate CPU byte-recheck contract. Identity, numeric,
replay or segment-budget failures halt with state preserved.

Before-fit value error on new on-policy roots is reported separately from
after-fit training error. Unsettled discretionary slices, original-task losses,
output movement and native captures remain distinct diagnostics. Fixed G0/Zeb/
fenced-Demon faceoffs, at most 60 seconds each, are integration checks only.

## Registration and capability refresh

[Public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/5168b85571a6ee2db217824ab7610a475fc8358e/lem/gus-ppo-value-2026-10-19/registration)
precedes training. Registration SHA256:
`c0304217687d67b39975925dc36a81ad401eb2e7495c6841be786514a17c899a`.
All 12 registration files passed fresh anonymous SHA256 verification.
Source is `6a42931b`; local recovery root is `scratch/lem/gus-ppo-value-v1`.
All 896 reserved deals exclude 602,325 prior physical deals and become future
training exclusions themselves. Root validation checks seating, uniqueness and
rehearsal separation.

The refresh in [[gus-book-silas-tools]] found no new replies or merged capability
that changes this comparison. Silas construction remains branch-only; the
Gus-to-Lem audit adapter remains separate work. Book concepts are questions for
interpretation, not reward labels. The `gus-training` skill now repeats this
capability check at experiment boundaries and substantive blockers. The ongoing
investigation is [#223](https://github.com/jasonyandell/mk5-main/issues/223).
