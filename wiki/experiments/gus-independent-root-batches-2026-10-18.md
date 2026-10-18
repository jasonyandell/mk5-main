---
title: "Gus: independent deals per outcome update"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Can doubling independent starting deals per update improve verified full-hand
outcome learning at the same 64 AdamW steps? This is the third full-hand study
for [[gus-tiny-gpt]], following [[gus-public-terminal-credit-2026-10-18]].
It changes sample coverage and rollout cost, not model capacity or opponents.

## Why this lever

The preceding study's saved first updates were reconstructed on CPU from each
pre-round checkpoint, decision batch and rehearsal RNG. Reward, parent KL,
tile and splat gradients were separated. Their component losses and weighted
sum matched the original MPS records within declared numeric tolerances.
The maximum relative norm discrepancy was 3.515e-5; this is numeric parity,
not bitwise cross-device training.

Shared-backbone rehearsal/reward gradient norms had median ratios 1.088 and
1.092 in the two previous RL arms. Median cosine was about .007; none of the
26 nonzero-reward batches in either arm had a total raw gradient opposed to
the reward gradient. Six of 32 collected batches had exactly zero current reward gradient.
These observations did not support changing the rehearsal coefficient on the
assumption that it overwhelmed reward. Raw gradients do not attribute AdamW's
step, which also depends on moments, clipping and decay.

The selected lever adds independent Monte Carlo samples to each update. It
retains eight rollouts per starting deal and full seven-turn reward credit.
More deals can reduce empty or noisy batches; they do not make an individual
all-win/all-loss GRPO group informative. The justification is the sampled
[policy-gradient estimator](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
and [GRPO's within-question comparison](https://arxiv.org/abs/2402.03300),
not a claim that either source prescribes this experiment's batch size.

## Registered comparison

Training source `1302a603abe3516041c2dd35abeb9c2776d00057`;
registration SHA256 `188f755d63420782901fe9456f93cde4bbd95f803f3c43ee8f09bce82e02e8b7`.
The 185,652-parameter splat parent, clean token schema, G0 partner, pinned Zeb
opponents, fixed bid 30/pip, learning rate 3e-5, clip .2, parent KL .02 and
rehearsal weight 1 remain fixed. The network receives no search results or
additional inputs. The pinned Demon is a separate evaluation competitor.

| Arm | Independent deals/update | Training deals | Sampled hands | AdamW steps |
|---|---:|---:|---:|---:|
| Small GRPO |4|128|1,024|64|
| Large GRPO |8|256|2,048|64|
| Rehearsal control |8|256|2,048|64|

Each arm runs 32 rounds, two inner updates, rollout microbatch 112 and rehearsal
batch 32. The small arm uses the first four of each large eight-deal batch.
Per-deal action streams and supervised draws are coupled; fixed other seats
respond to the learner's changed history. The rehearsal control collects
trajectories for diagnostics but trains only the original supervised tasks.
This is a matched optimizer-step comparison, not matched rollout cost.

Development uses 128 disjoint physical deals. Before training, 512 further
deals were reserved after excluding 590,098 prior physical deals and scheduled
hands. Selection includes initialization, maximizes development greedy success,
then sampled success, then earliest round, within +.02 of each original loss.
All three selections are frozen before any confirmation. Promotion requires
large-arm greedy gains of at least 2pp over parent and 1pp over both controls,
with positive paired 95% lower bounds and retained original losses.

## Lazy evidence identity

`lem.weights.state_identity`, introduced at `f87d6edb`, hashes exact tensor
names, dtypes, shapes and bytes with architecture, task, tokenizer, output head
and evaluation settings. Full recovery checkpoints with different histories can
share one evidence store only when this inference identity and the complete
execution recipe match. Similar weights or scores are insufficient.

The evaluator separately pins source/native bytes, helper/evaluator hashes,
platform, Torch build, CPU float32, one-example execution, field and random
streams. Label references retain each original checkpoint digest. The
`Extension` store uses immutable atomic units and content-addressed snapshots;
interruption or more requested samples computes only missing units. Byte identity
is scoped to that pinned runtime, not promised across devices or versions.

An actual-game smoke supplied four distinct serialized parent checkpoints with
one exact inference identity. Sixteen requests created four unique trajectories;
four reverse-order recomputations matched saved bytes. Eight targeted tests
cover identity changes, metadata snapshotting, nonmutation and checkpoint-history
reuse. Unique trajectories and model labels remain separate counts.

## Signal and cost

Small GRPO had 89/128 uniform-reward groups; large had 175/256. More independent
deals reduced entirely uniform batches from 6/32 to 1/32, while the fraction of
individual uninformative groups remained about 69%. Nonzero-advantage
discretionary decisions totaled 1,411 and 2,877. This supports the intended
signal-availability mechanism, not a playing-strength conclusion.

Training consumed 9.80e12 retired host instructions for small, 14.19e12 for
large and 14.20e12 for rehearsal. Summed segment wall times were 300s, 436s and
442s, including setup, development and tracking; these are descriptive, not a
speed ranking or complete GPU-work count. The larger arm doubled sampled hands
but shared fixed overhead. Segments used a 45s cooperative deadline and 90s
outer cap, with a maximum of 16 segments per arm; all completed within eight.
Old/current sampled-action log probabilities agreed within 8.47e-6.

## One literal learning example

Large round 1 contains a group with seven wins and one loss. Normalized
advantages are approximately +.378 for each win and -2.646 for the loss.
One winning prefix decodes to:

`start; bid30; partner bids; trump4; your hand 31 41 43 51 52 61 64;`
`play begins; partner 44; right 40; you`.

The model may choose 41, 43 or 64. Across the two combined optimizer updates,
the chosen 41 probability moved from .5993 to .5883, while 43 moved from .2805
to .2957. A trajectory's positive advantage does not promise that every one of
its moves increases: the update aggregates all trajectories, reference KL and
rehearsal through shared parameters. Literal token IDs, legal probabilities,
full hands and before/after arrays are retained. No teacher or hidden hand is
added to that prefix.

## Results and public evidence

All selected checkpoints retained the original prediction losses. Small selected
round 24, large round 8 and rehearsal round 24; their development greedy scores
were 46/128, 47/128 and 47/128, versus parent 45/128. Selection was frozen before
confirmation. The registered promotion criterion failed.

| Selected specimen | Fresh greedy wins | Fresh sampled wins | Tile NLL | Splat CE |
|---|---:|---:|---:|---:|
| Parent |201/512|186/512|1.329962|2.349329|
| Small GRPO |203/512|184/512|1.318359|2.330187|
| Large GRPO |202/512|183/512|1.323647|2.339055|
| Rehearsal |198/512|184/512|1.316132|2.328439|

Large minus parent greedy success was +0.195pp, paired 95% interval
[-0.977, +1.367]pp: five wins gained and four lost. Large minus small was
-0.195pp [-1.758, +1.172]; large minus rehearsal was +0.781pp
[-0.586, +2.148]. Large sampled success was -0.586pp versus parent
[-1.563, +0.391]. Intervals use 20,000 paired physical-deal bootstrap draws.
These are conditional hand results with the fixed field, not exact population
probabilities or general tournament strength. No incumbent was replaced.

At final round 32, development greedy wins were 44/128 small, 45/128 large and
46/128 rehearsal. Both original prediction losses improved in every final
specimen. More informative batches did not establish improved playing strength
at this 64-step budget. This single seed does not establish a capacity ceiling
or show that independent samples are generally unhelpful. The next diagnosis
should measure optimization progress on saved training evidence before changing
an update budget or estimator; confirmation is never reused for selection.

All 5,120 training hands and 4,096 unique confirmation trajectories passed
independent reconstruction of inputs, legal masks, partial scores and outcomes.
All 1,024 common small/large trajectories had identical frozen-seat behavior
until the learner first changed history; all 32 initial common trajectories
were identical. Sixteen confirmation units recomputed in reverse order matched
their canonical bytes. This comparison has four labels and four distinct
inference states; the separate identity smoke demonstrates shared-state reuse.
Three final specimens each retain native captures on 11,116 original frozen
rows, with checked checkpoint and representation digests. This is material for
`clean-neuron`, not an explanation of any particular unit.

## Fixed competitors and releases

One duplicate pair per final specimen/opponent used a predeclared common
bid30/pip schedule. All 24 games, 272 hands and 7,616 plays independently
replayed. The slowest faceoff took 6.24s, within the 60s cap. Every specimen,
including parent, split G0 1/2 and lost to native Zeb 0/2. Rehearsal split Demon
1/2; parent and both RL arms lost 0/2. These tiny samples check integration and
do not rank playing strength. Training Zeb and native tournament Zeb retain
separate pinned identities; Demon supplies no targets.

Final trained research releases: [small](https://huggingface.co/jasonyandell/gus/tree/aa3b34199026e498160c0ff1a789c53cdf11ab68),
[large](https://huggingface.co/jasonyandell/gus/tree/25ea7485e36194ff362f9fe3b758ffdb1a90f623),
and [rehearsal](https://huggingface.co/jasonyandell/gus/tree/e98243612a2ae1376aea1167be5dcf58c0f2b4c0).
Every export file passed fresh anonymous hash checks; safe tensors and both-head
CPU logits match their sources. Final trained exports are distinct from the
selected checkpoints above; the archive preserves both with initialization,
every formation round, optimizer/RNG recovery and literal inputs/outputs.

Canonical tournament synchronization is deferred until the shared repository's
existing commit quota permits the already prepared immutable batch. All records
remain in the public research archive; no games are rerun to publish them.
Issue [#223](https://github.com/jasonyandell/mk5-main/issues/223) tracks this and
continued outcome-learning diagnosis.

[Public CC0 evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/14308a0aefd185cd9dbd29fa84d12dab83e8c8db/lem/gus-independent-root-batches-2026-10-18) contains the registered plans, exclusions,
training-gradient audit, recovery and formation, selected confirmation,
immutable replay records, native captures and game schedules. All 16
uploaded files (577,601,738 bytes) passed fresh anonymous SHA256 checks;
all 4,891 deterministic-bundle members matched their individual hashes.
The directly readable registration, confirmation summary and replay verification
state the scope. Restore instructions pin the training source and separate
inference-identity helper. File-byte checks and actual replay checks are distinct.

Curves: [small](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/c7a4cc55),
[large](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/78d23dde),
[rehearsal](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/7d235fe6).
The `gus-training` skill carries the diagnosis, exact-state reuse and recovery
workflow. Generated science and weights are CC0; nanoGPT code retains MIT.
