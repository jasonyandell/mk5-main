---
title: "Gus: endgame initialization for full-hand outcome training"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and scope

Does the exact endgame curriculum in [[gus-fixed-field-curriculum-2026-10-18]]
provide a better initialization for subsequent full-hand GRPO than the original
splat parent or its matched endgame rehearsal control? That curriculum improved
fresh finite-model endgame regret without confirming stronger full-hand play.
This comparison tests the proposed transfer through another outcome-training
stage; it does not assume transfer has already happened.

[[gus-tiny-gpt]] keeps its 185,652 parameters, original own-seven/public-history
inputs, fixed bid30/pip task, G0 partner and pinned Zeb opponents. The changed
factor is the learner's starting weights. Original parent weights remain the
common frozen KL reference and common retention baseline in every arm. The
same supervised rehearsal, optimizer recipe, physical roots, action-stream
namespace and update counts are shared. The pinned Demon remains a separate
integration competitor and never supplies labels.

## Engineering

`lem.observer.outcomes` requires explicit `initial`, `reference`, `replay` and
their SHA256 identities. The starting policy receives fresh AdamW state;
`reference` supplies frozen legal distributions and the original-task retention
baseline. An initial policy that violates those declared bounds fails before
writing a usable training checkpoint. A changed reference invalidates exact
resume. Old studies require their published runtime rather than an implicit
reference default.

Real SIGTERM and hard-crash tests cover a distinct initial/reference pair.
The reference remains unchanged and has no gradients; initial snapshots preserve
the supplied warm-start weights. A warm start into the new phase is different
from `--resume`, which retains AdamW, RNG, source/plan identities and progress
within that phase. Completed resumes still preserve existing bytes.

## Registered comparison

Source `1dff0a63` is merged with 146 Python tests and repository gates GREEN.
The [public preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/885cede1d7ac6ee021adcb28772236c273271823/lem/gus-staged-outcomes-2026-10-18)
contains all plans and exclusions; all 11 files (12,423,655 bytes) passed fresh
anonymous hashes. Registration SHA256:
`13c07ae7ad556c9199ee80550a817df896546ffbdd52c83527d85737e8a5df8e`.
All 896 reserved physical roots exclude 599,687 prior deals and become future
exclusions themselves.

| Arm | Initial policy | Phase objective |
|---|---|---|
|parent|Original splat parent|Full-hand GRPO + KL + rehearsal|
|endgame|Predeclared seed42 exact-success selection|Same full-hand GRPO + KL + rehearsal|
|rehearsal_start|Predeclared seed42 endgame rehearsal selection|Same full-hand GRPO + KL + rehearsal|
|rehearsal_only|Same start as endgame|Supervised rehearsal only|

Every arm uses 256 fresh physical training roots with eight draws per root,
32 rounds/64 AdamW steps, learning rate 3e-5, clip .2, KL .02, rehearsal weight 1
and batch 32. The seven-turn reward horizon is unchanged. Cooperative segments
are 45s with 90s outer caps; at most 16 per arm, with two live arms. This is
an initialization comparison plus a matched supervised-continuation control,
not a changed opponent population or new architecture.

The shared 128-root development panel selects greedy then sampled success within
the common original-loss bounds +.02 each, including initialization and retaining
the earliest tie. The development method criterion requires the endgame arm to
gain at least 1/128 over its start and 2/128 over each other selected arm. All four
selections freeze before reading the separate 512-root confirmation. Confirmation
measures all four selected functions and all three original starts, with greedy
and one coupled sampled rollout per root. Identical inference states may share
records under the same pinned runtime/field/seed recipe; labels are counted
separately from unique trajectories.

Promotion requires the selected endgame arm to gain at least 2pp over the original
parent and 1pp over its own start and each other selected arm, with positive paired
95% lower bounds and common-reference retention. This prevents an already-stronger
start from being mistaken for learning during the new phase. The established objective is in
[[gus-full-hand-outcomes-2026-10-18]] and the lazy evidence contract is in
[[gus-success-probability-planning]].

The continuation is tracked by [issue #223](https://github.com/jasonyandell/mk5-main/issues/223).

## Result

Neither the registered development criterion nor full-hand promotion criterion
passed. Every arm selected round 8, retained separately from the trained round-32
specimen. All selections passed the common original-task loss bounds. The main
Hub incumbent is unchanged; confirmation did not trigger reselection.

| Frozen function | Greedy wins /512 | Sampled wins /512 |
|---|---:|---:|
|Original splat parent|204|186|
|Exact-endgame start|210|191|
|Endgame rehearsal start|201|192|
|Selected parent GRPO|201|183|
|Selected endgame GRPO|202|198|
|Selected rehearsal-start GRPO|202|192|
|Selected rehearsal-only continuation|204|199|

For endgame GRPO, greedy success changed **−0.39 percentage points** versus
the original parent (paired 95% interval **[−2.15, +1.37]**) and **−1.56pp**
versus its own start (**[−3.13, −0.20]**). Sampled success improved **+2.34pp**
versus the original parent (**[+0.59, +4.10]**), but was **−0.20pp** versus
rehearsal alone (**[−1.17, +0.78]**). Thus the sampled improvement does not isolate
an effect of reward learning. These are conditional bid30 hand outcomes under
fixed G0/Zeb, not unrestricted-auction strength or exact population probabilities.
Intervals use 20,000 paired physical-deal bootstrap draws, seed 91542.

## Training diagnostics and next lever

Only 67/256 endgame-arm roots produced mixed outcomes across eight draws;
129 were all-loss and 60 all-win. All-loss groups nevertheless averaged 5.26
distinct learner paths **before public settlement**. On unsettled discretionary
decisions, mean legal entropy was .893 nats and mean largest-action probability
was .612. These training-only observations do not support assuming that the
policy collapsed to one path. All-loss groups are not proven unwinnable.

The prior 512 **training** endgames diagnose forgetting: original-parent regret
was 3.193pp, exact-endgame start 1.598pp, selected endgame GRPO 1.694pp and final
endgame GRPO 1.674pp. Most of the learned endgame improvement survives. This is
retrospective retention evidence on used exercises, not a new held-out gain.

The next registered [[gus-exposure-budget-2026-10-18]] tests four times the fresh
on-policy exposure with the same network and recipe, comparing short and long
prefixes against matched rehearsal. The current 64-update result does not prove
an exposure bottleneck; it motivates this bounded test before another mechanism.

A literal training example starts with own hand `22 32 50 53 55 61 64`, bid30,
sixes trump and the learner leading. Eight sampled trajectories produce five
wins and three losses, giving advantages +.7746 and −1.291. The winning trajectory's
sampled `22` rises from .08984 to .10840 after the two inner updates. Those updates
include the whole batch, KL and rehearsal; this is not causal credit for `22`
alone. `endgame-worked-update.json` preserves tokens and all legal probabilities.
The group baseline follows [outcome-supervised GRPO](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS2);
verified game rules supply the reward here.

## Replay, recovery and cost

Independent replay verified all 8,192 training hands and 7,168 unique confirmation
trajectories across seven distinct inference states. The audit reconstructed
57,344 common-reference decision distributions and each formation checkpoint's
saved after-update probabilities. Maximum CPU/MPS log-probability errors were
3.34e-5 and 3.72e-5, below the preregistered 5e-5 tolerance; this is numerical
cross-device agreement. Separately, 28 actual lazy CPU units recomputed
byte-identically in reverse order. Paired frozen-seat behavior matched on 6,144
training trajectories until the learner first changed history.

Four **final** native captures each cover the same 11,116 frozen rows; selection
confirmation evaluates the separate round-8 checkpoints. Formation, optimizer,
RNG and full literal inputs/outputs are retained for [[lem-the-teacher]] and
`clean-neuron`. Native captures are specimens, not causal unit explanations.

| Training arm | Retired host instructions | CPU seconds | Wall seconds |
|---|---:|---:|---:|
|parent|13.830 trillion|822.7|461.1|
|endgame|13.857 trillion|824.9|457.6|
|rehearsal_start|13.861 trillion|853.7|475.5|
|rehearsal_only|13.847 trillion|857.1|484.1|

Every arm used eight cooperative segments and saved 32 formation checkpoints.
Host instructions do not measure MPS work; CPU/wall time describe this run and
are not player thinking budgets. The twelve fixed-competitor integration faceoffs
completed 24 games, 258 hands and 7,224 independently replayed plays; maximum
faceoff wall time was 4.74s under the 60s cap. Each model/opponent comparison has
only one duplicate pair and cannot support a reliable strength ranking.

## Published evidence

[Complete evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/8f2728f25bdccac90cfd4f6a6e83ec7644210f82/lem/gus-staged-outcomes-2026-10-18) contains 17 release files (773,431,481 bytes) and 8,299 archive members. Every file and member
passed fresh anonymous SHA256 verification. Direct reports include confirmation,
training signal, endgame retention, replay audit, costs and the restore map.

- [parent selected model](https://huggingface.co/jasonyandell/gus/tree/d79c81e5c54c63009fa282c0066d380d59509e77) — exact tensor and both-head CPU logit export parity.
- [endgame selected model](https://huggingface.co/jasonyandell/gus/tree/962049e99e20b9ee6913ca8e5cf70e281c694454) — exact tensor and both-head CPU logit export parity.
- [rehearsal_start selected model](https://huggingface.co/jasonyandell/gus/tree/1d00e4e03f1c08ac5ef3645859f2319cd3b2ee8c) — exact tensor and both-head CPU logit export parity.
- [rehearsal_only selected model](https://huggingface.co/jasonyandell/gus/tree/48eb75c61846151e42b2c3a871251c91bb9ce08c) — exact tensor and both-head CPU logit export parity.

Canonical tournament synchronization remains deferred until the recorded service
window (2026-10-18 23:24:27 UTC); all audited records are already in the public
research archive. The existing idempotent batch includes the preceding exact
curriculum. No games need to be rerun. The broader investigation remains open
in [#223](https://github.com/jasonyandell/mk5-main/issues/223).
