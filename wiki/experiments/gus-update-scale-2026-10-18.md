---
title: "Gus: outcome update scale"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Can a modest learning-rate increase improve verified full-hand success at the
same data and optimizer-step budgets? This fourth full-hand study follows
[[gus-independent-root-batches-2026-10-18]] in the fixed-player series for
[[gus-tiny-gpt]]. It keeps the architecture, information contract and behavioral
field fixed. External search and future-splat auxiliary training remain deferred.

## Why update scale

A training-only diagnostic reconstructed the preceding study's saved before/after
legal distributions and formation changes. No confirmation outcomes enter it.
All 31 informative larger-batch rounds improved their clipped reward surrogate;
the rehearsal control improved it in 16/32 rounds. The reward update is doing
something measurable, but its scale is small.

Across the larger-batch arm, median old-to-new categorical KL was 5.137e-5 nats
across all learner turns and 9.382e-5 on unsettled discretionary turns. Maximum
all-turn KL was .000793. The median active total-variation distance was .00434;
median per-batch active greedy-choice change was zero. These are movements
across two optimizer steps on the same collected states, not overall change
from initialization. All-turn metrics include forced moves; the active slice
avoids that dilution. About 69% of nonzero-advantage active actions moved in the
advantage's indicated direction. Aggregation through shared parameters means
not every rewarded move must increase.

Old/current policy KL and frozen-parent KL answer different questions. The old
policy generated this round's samples and supplies the probability-ratio
denominator. The frozen parent anchors the whole post-training trajectory.
Small old/new KL does not prove insufficient learning, and a lower surrogate
loss does not prove better return. It motivates a controlled update-scale test.
[PPO](https://arxiv.org/abs/1707.06347) and
[Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) describe
optimization of the clipped surrogate while controlling policy change; neither
source prescribes this study's particular learning rates. GRPO retains its
within-deal normalized outcome advantages and fixed-reference regularization.

## Registered comparison

Training source `14298336`; registration SHA256
`262d0653b3ce249674289b763e4341e0d3c924b675961bcaeac1d59581fdfa54`.
The diagnostic is hashed into the registration before training.

| Arm | Objective | Learning rate | Training roots | Sampled hands | AdamW steps |
|---|---|---:|---:|---:|---:|
| Base |GRPO + rehearsal|3e-5|256|2,048|64|
| Step |GRPO + rehearsal|1e-4|256|2,048|64|
| Rehearsal |Original supervised tasks|1e-4|256|2,048|64|

All start from the same 185,652-parameter joint splat parent with fresh AdamW
state. The same 256 physical starting deals, action streams and supervised RNG
are used across arms, in 32 rounds of eight deals and eight learner trajectories
per deal. Each round takes two updates. Learning rate is the changed factor;
the higher-rate rehearsal control separates stronger supervised updates from
outcome learning. This is a matched data/update-count comparison.

The fixed recipe retains clip .2, frozen-parent KL .02, rehearsal weight 1,
rollout microbatch 112, rehearsal batch 32 and full seven-turn reward credit.
The G0 partner, pinned Zeb opponents and common bid30/pip remain unchanged.
Other seats respond to the histories the learner creates. The clean network
input remains own original seven and public contract/actor/tile history; legal
masks constrain output. Terminal outcomes and hidden deals are audit sidecars,
not added features. The Demon remains a separate fenced competitor.

A further 128 physical deals provide development and 512 provide confirmation,
all reserved before learning after excluding 591,033 prior ordered physical
deals and schedules. Selection includes initialization, maximizes greedy then
sampled development success, keeps the earliest tie and requires each original
prediction loss to stay within +.02 of its parent value. All three selections
are frozen before reading confirmation. Promotion requires step-arm greedy
gains of at least 2pp over parent and 1pp over base and rehearsal, with strictly
positive paired 95% lower bounds and retention. Conditional hand results do not
establish broad tournament strength.

Independent arms run in separate processes through shared-machine CPU slots.
Each segment uses a 45s cooperative budget and 90s outer cap, with at most
16 segments per arm. Numerical, replay or probability-identity failures stop
the affected run with its state preserved. Pauses retain AdamW, RNG, complete
rounds, formation, literal inputs/outputs and W&B identity. Completion now writes
consistent `status.json` and `report.json`; actual SIGTERM/crash tests cover
both horizons and rehearsal, including byte-preserving completed resumes.

## Results

All three arms selected initialization. The base arm's final development score
tied its parent, so the registered earliest-tie rule retained round zero. Fresh
confirmation therefore measures one shared inference function under four labels:
**209/512 greedy wins (40.82%) and 181/512 sampled wins (35.35%)**, identical
to the parent. Every paired selected-policy difference is zero. No promotion.
This is not three independently trained policies tying on confirmation.

Final trained checkpoints remain separate research specimens:

| Specimen | Dev greedy /128 | Dev sampled /128 | Tile NLL | Splat CE |
|---|---:|---:|---:|---:|
| Parent |52|51|1.32996|2.34933|
| Base final |52|51|1.31567|2.32648|
| Step final |51|50|1.31401|2.31783|
| Rehearsal final |49|50|1.30845|2.31628|

The larger learning rate did increase policy movement. Median per-round old/new
all-turn KL rose from 4.83e-5 to 6.44e-4 nats; active total variation from .00452
to .01529; active greedy-choice change from zero to 1.77%. Every informative
GRPO round improved its sampled clipped surrogate (29/29 base; 30/30 step),
versus 18/30 for rehearsal. Greater movement did not yield a selected playing
gain. Neither small updates nor rehearsal dominance alone explains the null.
All-win/all-loss groups remained common: 186/256 base and 180/256 step groups
had zero reward variance. Each completed nine resumable segments. Summed host
cost was 13.60–13.65 trillion retired instructions per arm; summed segment wall
time 498–504s. Concurrent elapsed time and GPU work are different quantities.

## Replay and specimens

All 6,144 training hands and 1,024 unique confirmation trajectories passed
independent replay. Four confirmation labels share one exact inference state;
the lazy store computed that state's records once. Sixteen confirmation units
recomputed in reverse order matched their canonical bytes. All 2,048 common
base/step trajectories retained frozen-seat equality until the learner first
changed history; the initial 64 trajectories matched completely. Each final
specimen retains native captures on 11,116 original frozen rows. Formation,
optimizer/RNG state and literal model inputs/outputs are preserved. These are
interpretability specimens for `clean-neuron`, not unit explanations.

One duplicate pair per specimen/opponent completed 24 games against fixed G0,
native Zeb and fenced Demon: 268 hands, 7,504 independently replayed plays,
maximum faceoff 6.98s. All specimens split G0 and lost both Zeb games. Base and
rehearsal split Demon; step and parent lost both. These tiny integration samples
do not rank playing strength. Training Zeb and native tournament Zeb have
different pinned identities; Demon supplies no learning labels.

Final trained exports: [base](https://huggingface.co/jasonyandell/gus/tree/5efdca2faf620d3a5cfadc1913e619960b2a5d7d),
[step](https://huggingface.co/jasonyandell/gus/tree/692627b14c4699e8e63f99c6d7695c3c0fa159e3),
[rehearsal](https://huggingface.co/jasonyandell/gus/tree/7258b9a4417f1b3c7551daa34ca3041d3904dcb8).
All files passed fresh anonymous SHA256 verification and both output heads
matched their source checkpoint on CPU. The Hub main incumbent is unchanged.

## Next lead: exact feedback under the fixed game field

A separate feasibility probe selects 16 eligible two-choice endgames from
training trajectories only. It enumerates 329 legal-consistent remaining worlds
and 658 complete continuations using the actual fixed G0 partner and pinned Zeb
opponents. All continuations independently replay. Enumeration took 1.12s on
this machine (descriptive timing). Five questions have unequal action values;
one best-action set differs from the earlier lowest/highest/uniform verifier.
This small diagnostic does not establish field mismatch as the cause of failed
transfer. It trains no model and adds no search to Gus.

Each root action is the same across all worlds; the learner's later move is
forced. Counts are exact rationals. For one missed question the values are 1/3
and 4/9, whereas the old three-style verifier tied them at 13/27. The probe's
uniform legal root belief ignores pre-root behavior likelihood. This remains an
explicit finite approximation, not a true-posterior certificate. The old
verifier retains its identity and results; this would be a new named curriculum
using the same G0/Zeb continuation policies as the full-hand stage.

The proposed next comparison uses those cheap complete rewards for direct
expected-success policy gradients, with the existing frozen-parent KL and
supervised rehearsal. Enumerating two available rewards removes action-sampling
noise; it does not replace the terminal objective with teacher agreement. Keep
fresh physical-deal splits, initial/selected/final specimens and full-hand
transfer checks. General playing improvement still requires fresh actual games.
[[gus-success-probability-planning]] owns the objective and lazy-evidence contract.
[[gus-fixed-field-curriculum-2026-10-18]] implements the ensuing curriculum and
tracks its bounded comparison separately.

## Public evidence

[Public CC0 evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/c1913bc459c2ae0adb1c01cbbe14801397f8c1f1/lem/gus-update-scale-2026-10-18)
contains registration, both optimizer diagnostics, all recovery/formation,
confirmation, literal trajectories, native captures, games and the separate
exact-field probe. All 18 uploaded files (569,629,469 bytes) passed fresh
anonymous SHA256 checks; every one of 1,822 deterministic-bundle members matched
its individual hash. The restore recipe pins source `14298336`, all input
assets and the separate inference-identity helper. File hashes and actual
trajectory replay are independently checked claims.

Canonical tournament synchronization subsequently completed in
[one combined v2–v4 batch](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/3e2c74cc5f8900435fbbd4696ce6af1eb05be1ac):
45 missing immutable records added, all 49 shared records freshly downloaded
anonymously and hash-verified. Existing records were compared byte-for-byte;
no games were rerun. The archive's earlier deferred receipt records its original
publication state; the completion supplement records the successful retry.

Curves: [base](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/ada1930f),
[step](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/c5b1767f),
[rehearsal](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/30e32c15).
The `gus-training` skill carries update diagnosis and exact-field scope.
Generated data and weights are CC0; nanoGPT implementation retains MIT.
