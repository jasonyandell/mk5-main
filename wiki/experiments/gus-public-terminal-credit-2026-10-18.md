---
title: "Gus: public-terminal policy credit"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and finding

This is the second full-hand outcome study for [[gus-tiny-gpt]], following
[[gus-full-hand-outcomes-2026-10-18]]. It tests one change: stop giving actions
reward credit after the public score guarantees make/set. The ordinary GRPO,
settlement-masked GRPO and rehearsal-only arms all selected their initialization.
No improvement or new incumbent was established.

## Textbook reason and exact scope

The training-only audit of the preceding study found diverse trajectories even
in uniform-reward groups: all-loss groups averaged six distinct learner paths
out of eight. About32% of learner decisions occurred after public settlement.
Repeating an identical path was therefore not the sole source of weak signal.
The new comparison removes known irrelevant reward-score terms before adding
rollouts, a learned critic, model capacity or a curriculum.

For bid30, bidder points>=30 or defender points>12 fixes the binary outcome,
because all42 points are nonnegative and allocated once. The mask is evaluated
**before** each action; the move that earns the settling trick remains credited.
Conditioned on that public prefix and the other group rewards, the trajectory's
normalized advantage A is fixed. For a subsequent legal action,

`E[A grad log pi(a|I)] = A sum_a pi(a|I) grad log pi(a|I) = A grad 1 = 0`.

This is an application of policy-gradient causality and the expected score
identity, described in [Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you).
It removes zero-mean noise from the on-policy score estimate. It does not prove
that a finite clipped GRPO update improves play, or fix all-win/all-loss groups.

`lem.observer.outcomes` now requires an explicit `reward_horizon`: `full` or
`unsettled`. The latter masks only the reward surrogate. It keeps the original
seven-turn denominator, full-state categorical KL, rehearsal and every full
28-play trajectory. No active-length renormalization, reward shaping, input
feature, external search or opponent change is introduced. Independent scenario
rules verify partial banked points and the mask, alongside full outcome/input
replay. The model still sees only its original own seven and clean public history.

## Registered matched comparison

Source `7747f5c2d3fd5ca2c64c6a79ffa51482d55a581d`; issue
[#223](https://github.com/jasonyandell/mk5-main/issues/223). Three185652-parameter
arms start from the same splat-pretrained parent. Each has128 distinct training
deals, eight trajectories per deal,32 rounds and64 AdamW steps. Development has
64 disjoint physical deals;256 further deals were reserved before training.
The registered prior-data exclusions and all13 potential prior tournament deals per opponent are retained.

Fixed controls: G0 partner, pinned Zeb opponents, bid30/pip, root/action/replay
streams, learning rate3e-5, clip0.2, parent KL0.02, rehearsal weight1, rollout microbatch112 and rehearsal batch32.
Rehearsal alone receives the same supervised draws and optimizer count, without
RL or KL. Each arm includes initialization in selection; maximize development
greedy success, then sampled success, then earliest round, within+.02 retention
bounds on both original losses. All selections were frozen before confirmation.
The method criterion required >=2pp over parent and >=1pp over both controls,
strictly positive paired95% lower bounds and retention.

## Results

All three arms selected round0: the identical pretrained function. Fresh
confirmation therefore measures that retained function, not the final trained
weights. Every selected arm and parent won103/256 greedy hands (40.23%) and
104/256 sampled hands (40.63%); every paired difference was zero. The registered
promotion criterion failed. These hand probabilities are conditional on this
fixed field; the empirical rates are not exact population probabilities.

| Specimen | Final dev greedy | Final dev sampled | Tile NLL | Splat CE |
|---|---:|---:|---:|---:|
| Parent |22/64|20/64|1.329962|2.349329|
| Full credit |19/64|22/64|1.316000|2.327235|
| Unsettled credit |19/64|21/64|1.315962|2.327140|
| Rehearsal |19/64|21/64|1.313882|2.325617|

Both original prediction losses improved in every final specimen, without a
better selected player. Both GRPO arms had85/128 uniform-reward groups: only43
groups supplied relative win/loss signal. The unsettled arm removed202 of1478
nonzero-advantage discretionary credit terms (13.67%); its remaining1276 terms
are17.80% of all7168 learner decisions. Thus the roughly one-third fraction of
post-settlement decisions overstates the fraction of effective reward credit
removed: forced moves and zero-advantage groups already contribute no reward
gradient. The cleanup did not address the dominant zero-variance-group issue.

Training cost was7.13e12 retired instructions for full credit,7.11e12 for
unsettled credit and7.10e12 for rehearsal. Descriptive summed segment wall times
were251s,223s and229s, including setup, development and tracking. Each segment
used45s cooperative training with a90s outer cap; complete rounds were resumed.
These clocks are not strength budgets or a controlled speed ranking.

A saved mixed group had four wins and four losses, producing advantages+1/-1.
The first winning trajectory's5-0 probability changed0.76015 to0.75440 across
two updates. That is the combined minibatch, KL and rehearsal result, not
proof that every move in a winning trajectory is individually good or must
increase. The archive preserves its literal prefix and both legal probabilities.
The follow-up gradient diagnosis and independent-deal batch comparison are
recorded in [[gus-independent-root-batches-2026-10-18]].
Confirmation remains excluded from future training and recipe selection.

## Reproducibility and interpretation

All3072 training hands and2048 confirmation trajectories passed independent
reconstruction of clean inputs, legal actions, intermediate banked points,
terminal masks and outcomes. Sixteen immutable confirmation units, recomputed
in reverse request order, matched their saved canonical bytes. The recipe pins
source and native hashes, Python/NumPy/Torch versions, platform/machine, Torch
build, CPU float32/single-example inference and evaluator bytes. Seeds alone
are insufficient for a cross-runtime promise.

The twelve targeted tests cover exhaustive public-score boundaries, an exact
toy score-gradient cancellation, independent partial-score disagreement,
terminal-mask tampering, masked-gradient behavior, single-writer locking and
actual SIGTERM/mid-update crash recovery for both horizons. CPU recovery is
bitwise; MPS and cross-device training are not promised bitwise identical.

Three final models each retain native representations on the original11116
frozen rows, all32 formation checkpoints, initialization and full recovery.
This supplies `clean-neuron` specimens, not causal unit explanations. The
already-clean hypothesis is not graded by this reward-credit comparison.

## Fixed competitors and public evidence

One predeclared duplicate pair per model/opponent uses common bid30/pip and
two copies per partnership. The training Zeb and native tournament Zeb0.4.1
remain separately pinned. The Demon is a fenced competitor; its estimates are
never learning targets.

| Final specimen | G0 | Native Zeb | Demon |
|---|---:|---:|---:|
| Pretrained |2/2|1/2|0/2|
| Full credit |2/2|1/2|0/2|
| Unsettled credit |2/2|0/2|0/2|
| Rehearsal |2/2|1/2|0/2|

All24 games,254 hands and7112 plays independently replayed. The slowest faceoff
was5.69s, below the one-minute cap. These are integration observations, not a
strength ranking. Canonical tournament synchronization subsequently completed
with the v3/v4 records in [one verified batch](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/3e2c74cc5f8900435fbbd4696ce6af1eb05be1ac).
[[gus-update-scale-2026-10-18]] owns that receipt. The original archive preserves
the earlier quota-delayed state; no games were rerun for publication.

Final trained research releases: [full credit](https://huggingface.co/jasonyandell/gus/tree/ed4cdf45d41c35bb166536af3ee1c3230fa97c3f),
[unsettled credit](https://huggingface.co/jasonyandell/gus/tree/6110f707d3997dff2e06a7ae96267ed41b0a9abe),
and [rehearsal](https://huggingface.co/jasonyandell/gus/tree/d35bf30399df6d20515f9156c4ec630910a24aff).
All export files passed fresh anonymous hash checks; both-head logits and safe
weight tensors match their training sources. The main Hub incumbent is unchanged.

[Public CC0 evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/299bb73395a7a91c6f1e413ce8720b42603a02cb/lem/gus-public-terminal-credit-2026-10-18)
contains the registered plans, exclusions, recovery, formation, literal
inputs/outputs, lazy confirmation records, native captures and tournament
records. A deterministic bundle plus member manifest avoids thousands of Hub
resolver requests. All13 uploaded files (568636255 bytes), including the bundle, passed fresh
anonymous SHA256 checks; all2816 bundle members matched their individual
manifest hashes. File verification and the sixteen actual replay recomputations
are separate checks.

Live curves: [full](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/db35c136),
[unsettled](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/3788f0a3).
The `gus-training` skill records the explicit horizon, normalization invariant,
runtime-pinned lazy evidence and bundle verification workflow.
