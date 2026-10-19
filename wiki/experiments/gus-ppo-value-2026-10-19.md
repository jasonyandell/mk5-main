---
title: "Gus: PPO with a learned success-value baseline"
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: complete
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

## Completed result

The registered development and promotion criteria both fail. All selected
models retain the original prediction losses; no new incumbent is promoted.
Selections are PPO round 16, GRPO round 8 and rehearsal round 32.

| Selected player | Development greedy /128 | Fresh greedy /512 | Fresh sampled /512 |
|---|---:|---:|---:|
| Original parent | 47 | 218 | 203 |
| PPO/value | 47 | 221 | 202 |
| GRPO | 48 | 222 | 202 |
| Rehearsal | 47 | 221 | 203 |

PPO versus parent gains **0.586 percentage points**, paired 95% interval
**[−0.977, +2.148]**: nine previously lost hands become wins, six wins become
losses. PPO versus GRPO is −0.195pp [−1.367, +0.977]; versus rehearsal is
0.000pp [−1.367, +1.367]. The sampled secondary result also establishes no gain.
These are paired physical-root bootstrap intervals, not general algorithm ranks.

All arms finish their 32 rounds within the registered segment budgets. PPO and
GRPO each take eight segments, rehearsal seven. Host instructions retired are
13.725, 13.727 and 13.731 trillion respectively; CPU times 797.2, 796.2 and
750.6 seconds. Segment wall totals are 450.9, 450.7 and 417.0 seconds, descriptive
under shared load. Host instructions do not include MPS GPU work.
The critic performs 640 separate Adam steps, each on 448 saved decision rows.
Its elapsed time and host instructions were not isolated from the surrounding
actor/collection work, so the arm totals cannot attribute a critic overhead.

## What the critic learned, and did not

Before fitting each fresh batch, evaluate its old forecasts on unsettled
positions with more than one legal action. The 6,612 PPO decisions cover 250
physical training roots; six roots contain no decision in this slice.

| Forecast diagnostic | Error; lower is better |
|---|---:|
| Old critic, clipped to [0,1], before fitting | Brier 0.277446 |
| Success frequency from earlier batches only | Brier 0.239171 |
| Old critic, raw, before fitting | MSE 0.296104 |
| Same batch after fitting | Training MSE 0.105568 |

The clipped critic's excess error over the simple prior is **+0.038275**,
paired root-bootstrap 95% **[+0.014938, +0.061371]**. These sequential forecasts
come from changing on-policy training states; the interval is descriptive for
that stream, not fresh strength confirmation or measured gradient variance.
The head fits its current examples but this recipe does not produce better
forecasts on the next batches than the simple prior.

This differs from [[gus-value-baseline-2026-10-18]]'s pooled, standardized ridge
fit over hundreds of roots. That result showed usable information in the frozen
representation. It did not establish that 20 Adam steps on each eight-root
batch would maintain a useful on-policy predictor. The present result does not
show that the representation lacks success information or that PPO cannot work.

PPO assigns nonzero credit to all 6,612 unsettled discretionary decisions;
GRPO does so for 2,545/6,613. Nonzero credit and lower after-fit error do not
establish better policy credit. The next evidence-supported question concerns
critic forecast generalization and stability at fixed inputs and seats. No next
recipe is selected by these confirmation scores; register it separately after
the discussion window and capability refresh.

## One actual update

The preserved opening prefix decodes to: bid30, you bid, fours trump; your hand
11, 20, 21, 22, 44, 60, 63; you lead. Gus samples 44 and that trajectory succeeds.
The old critic predicts .5, so the raw return-minus-value advantage is +.5;
batch normalization produces +1.249. Across the two whole-batch actor updates,
44's probability moves from 58.90% to 62.15%. The update also includes other
trajectories, KL and rehearsal: this is a literal teaching example, not causal
credit assigned to that single opening. Tokens and all legal before/after
probabilities are in `ppo-value-worked-update.json` in the archive.

## Audit, specimens and integration

Independent replay validates all 6,144 training hands, 4,096 confirmation hands,
43,008 reference-policy decision probabilities, all 896 regenerated roots,
formation outputs and the three selection histories. Sixteen actual lazy CPU
units replay byte-identically in reverse order. Fixed-seat outputs match until
learner histories diverge. All 32 critic-feature batches match bitwise on the
same device. No runtime source, outcome, selection or numeric threshold changed.

Two corrections remain visible. The original prefix audit incorrectly reused
a CPU/MPS log-probability tolerance for raw hidden features. Its failed check
and diagnostic are preserved. A separate same-device feature check, with
log-probability tolerances unchanged, was published before confirmation in the
[evaluation protocol](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/91b0e37afa8e92616b8196b863a207799363b8fb/lem/gus-ppo-value-2026-10-19/protocol).
Later the frozen diagnostic script computed its statistics but failed saving a
NumPy float32 field. The separate correction casts only that output field to
Python float; original arithmetic, script and failure remain preserved. This
[serialization correction](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/3d85629309b0482cfec864205b4e8b23cedd3621/lem/gus-ppo-value-2026-10-19/diagnostic-correction)
was published after confirmation and before citing the corrected output.

The three selected research specimens have exact tensor and both-head CPU
logit export parity, plus fresh anonymous download hashes:
[PPO](https://huggingface.co/jasonyandell/gus/tree/58c384b78d46143a385eddd0b26229a87d8703a2),
[GRPO](https://huggingface.co/jasonyandell/gus/tree/2e45b64802f0948966752b334f458b46d120be40),
[rehearsal](https://huggingface.co/jasonyandell/gus/tree/f82678c89005749194a1168768e575fb059b3988).
The main Hub incumbent remains unchanged. All 96 per-round formation checkpoints,
full recovery and three final-round native captures of 11,116 frozen rows each
are preserved for `clean-neuron`; captures are not causal unit explanations.

Nine duplicate faceoffs complete under their 60-second caps, maximum 5.66s:
18 games, 176 hands, 4,928 plays independently replayed. PPO splits G0 and Zeb,
loses both Demon games; GRPO splits G0 and loses both against Zeb/Demon;
rehearsal splits all three. Each opponent has one paired seed, so this is
integration evidence only. The immutable records are
[canonical tournament data](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/0f0b27ad4ff098590a6a09fa3aa9c26af84a09b1).

## Completed public evidence

[Evidence archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1aea81dd8341dad5eb26de0a218a5125c328ee4c/lem/gus-ppo-value-2026-10-19): 19 files,
580,012,369 bytes and 4,949 archive members. Every file
passed fresh anonymous SHA256 download verification and every archive member
matched its manifest. Full inputs, trajectories, formation, recovery, corrections,
native traces, worked updates and restore instructions are included. Training
source remains `6a42931b`; local recovery root remains `scratch/lem/gus-ppo-value-v1`.

The completion refresh found no new Lem/Book replies and Silas PR #166 remains
open; available capabilities do not change this readout. Unanswered requests
remain pending. [[gus-book-silas-tools]] routes those boundaries. Jason's
30-minute steering window follows the result readout; the next experiment is
not started as part of this publication.
