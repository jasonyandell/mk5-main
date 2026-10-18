---
title: "Gus: full-hand outcome GRPO with fixed players"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and scope

Can grouped, independently verified complete-hand outcomes improve
[[gus-tiny-gpt]] beyond matched splat rehearsal alone? This is the full-hand
stage after [[gus-probability-reward-v3-2026-10-18]], not another endgame verifier
experiment. Jason authorized autonomous splat-to-outcome iterations while keeping
other players fixed; [[gus-success-probability-planning]] records that choice.

The same 185,652-parameter joint model begins each arm from
[[gus-splat-pretraining-2026-10-18]]. One learner plays with the fixed G0 partner
against the pinned Zeb field player, `k0,n160,f16`, posterior beliefs, uniform
reading, margin tiebreak and planner seed 0. G0/field checkpoint hashes, complete
policy specification, source and native runtime hashes are in the plans.
Bid30/pip and the clean own-original-seven/public-history input remain unchanged.
Other hands, legality, rewards and teacher histograms remain environment/output
sidecars. No external search, opponent evolution or architecture growth occurs.

## Why this textbook pattern fits

Complete continuation values are no longer cheap to enumerate. Outcome-supervised
[GRPO](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS2) supplies a sampled policy
gradient without fitting a critic: eight legal learner trajectories share the
same complete environment root and fixed other-seat policies. A deterministic
referee verifies each terminal make/set reward. That verifies the observed
result, not an exact population success probability.

For trajectory `i`, `A_i = (R_i - mean(R_group)) / std_population(R_group)`;
all seven learner turns receive that advantage. Constant-reward groups receive
zero advantage and remain in the data. The loss combines the clipped action
probability-ratio surrogate, exact categorical `KL(pi || frozen_parent)` over
legal support, and original tile/splat cross-entropy rehearsal. The full-support
KL replaces the paper's sampled estimator because Gus has at most seven legal
outputs. Old sampling probabilities and the separately frozen parent have
different roles. Rehearsal addresses the measured forgetting in
[[gus-retention-iterations-2026-10-18]]; it is not the outcome reward.

`lem/observer/outcomes.py` accumulates all trajectory microbatches before each
AdamW step. Every hand has seven decisions, so mean-over-decisions equals
mean-over-hands then turns. The control uses the same optimizer/rehearsal draws
without the RL or KL terms; it still collects trajectories for diagnostics.
This is conditional imperfect-information play: the learner's later decisions
read its public history, never the environment's hidden deal.

## Registered bounded iterations

Each arm uses 64 distinct training deals × eight rollouts = 512 hands, 16 rounds
and 32 optimizer updates. Both arms share training/development/confirmation deals
and random streams. AdamW uses matrix decay 0.1, betas 0.9/0.95, clip norm 1;
ratio clip 0.2, KL coefficient 0.02, rehearsal weight 1 and batch 32. The network's
capacity stays fixed. Each segment has a 45-second cooperative budget and 90-second
outer cap, at most eight segments per arm; each arm finished in two segments.
These clocks govern safe stopping, never player thinking or outcome scoring.

- **First comparison, 1e-5:** both arms retained initialization; all scheduled
  development counts stayed at 12/32 greedy and 13/32 sampled wins.
- **Single registered follow-up, 3e-5:** same deals, streams, update count and
  fixed seats; only the learning rate changes numerically. GRPO selected round 8;
  rehearsal selected round 12. Both reached 13/32 greedy and 13/32 sampled wins.
  The follow-up was registered before any reserved confirmation outcomes were read.

Selection includes initialization, enforces increases of at most 0.02 in each
retention loss, then maximizes development greedy wins, sampled wins and earliest
round. The control prevents attributing gains from additional imitation to RL.
Full training sources: `da020e79` first comparison; `e358d7fa` follow-up. The latter
adds an output lock and preserves completed artifacts on resume; training
arithmetic is unchanged. Plans pin exact files independently of these labels.

## Fresh confirmation and limits

One 128-deal panel was reserved before either comparison. All three policies
played each complete deal with identical fixed G0/Zeb assignments and action
seeds. Greedy and one sampled rollout per deal are separate deployment readouts.

| Selected policy | Greedy wins | Sampled wins | Retention tile NLL | Retention splat CE |
|---|---:|---:|---:|---:|
| Original splat parent |56/128|56/128|1.329962|2.349329|
| GRPO, round 8 |58/128|55/128|1.324889|2.340213|
| Rehearsal, round 12 |55/128|56/128|1.320387|2.335071|

GRPO gains two greedy wins and loses none versus the parent: +1.5625pp,
paired-deal bootstrap 95% interval [0,3.90625]pp. Against rehearsal it gains three
and loses none: +2.34375pp, [0,5.46875]pp. These are very few discordant hands;
zero observed reversals is not a guarantee of no regressions. Both intervals
include zero. The registered promotion rule requires at least 2pp improvement
over parent and control with intervals strictly above zero and retention.
**No promotion.** Sampled play does not improve. Confirmation is not reused to
select another checkpoint. Retention is measured on the declared balanced
rehearsal validation bundle, not the entire original 500k corpus.

Across the follow-up's 64 GRPO root groups, 40 have uniform rewards and 24 supply
relative outcome signal. That is a concrete next measurement lead: improve the
amount/coverage of useful on-policy experience under the same field before
asserting a capacity shortage. A train-only curriculum or a larger sample budget
needs its own matched comparison; it is not a completed improvement here.
[Issue #223](https://github.com/jasonyandell/mk5-main/issues/223) carries that next
autonomous iteration with the current confirmation now excluded from training.

## A real update and the interpretation handoff

The first follow-up group produced rewards `[0,0,1,0,0,0,0,1]`: winning advantages
are `+1.73205`, losing advantages `-0.57735`. One actual winning trajectory opens
with the clean prefix:

`start; bid30; you bid; trump3; own hand 10 30 33 54 55 63 64; play begins; you`.

Its sampled tile is 3–3. Across the two joint optimizer updates, that tile's legal
probability rises 56.37%→63.70%. This is the recorded combined RL/rehearsal update,
not proof that this opening alone caused victory. Full numeric tokens, legal
probabilities, reference distributions and every subsequent play are preserved.

All four final trained models have captures on the unchanged
[[gus-frozen-benchmark-2026-10-18]]: 11,116 rows, logits, splat distributions,
residuals, MLP activations and attention. Every training round has a full formation
checkpoint. These are specimens for `clean-neuron`, not causal neuron claims.
Final trained weights differ from development-selected weights; both are retained.

## Lazy replay and recovery

Deal seeds and rollout-action seeds are distinct. The plan pins generator,
policies, weights, source, native binary and software versions; seeds alone are
insufficient for a byte claim. The confirmation uses the existing immutable
`benchmark.extension.Extension`: one physical-deal/policy/sample unit, atomic
writes, source-bound recipe, hashes and immutable snapshots. Interrupted runs
compute missing units. The base frozen panel stays unchanged.

All 2,048 main/follow-up training hands were independently reaudited. Twelve
confirmation units were recomputed in reverse order on the same CPU runtime,
matching their canonical serialized bytes. The audit exposed tuple/list
inequality after JSON round-trip: the extension now compares canonical records,
which preserves the existing bytes while still rejecting changed evidence.
A regression test covers unchanged tuples and genuinely changed scores.

SIGTERM and a simulated crash after an AdamW update both recover to bitwise-equal
CPU model, optimizer, sampler, RNG and metrics. Complete-round checkpointing
includes scheduled development work. Completed resumes are byte-preserving,
concurrent writers are rejected, and a timeout never becomes a loss. Numerical
training is not promised bitwise identical across MPS, devices or versions.

## Fixed-competitor smoke and releases

The [[tournament]] uses two copies per partnership, distinct from the one-learner
training field. Pretrained and the two final3e-5 specimens each played one
predeclared duplicate pair against G0, native Zeb 0.4.1 and the pinned 100-world
Demon. The common bid30/pip bidder and all 13 potential deals per opponent are
shared across models. Demon draws retain the candidate-independent paired seed
protocol. Its internal estimates never become rewards.

| Final specimen | G0 | Native Zeb | Demon |
|---|---:|---:|---:|
| Pretrained |1/2|0/2|1/2|
| GRPO |1/2|0/2|1/2|
| Rehearsal |2/2|0/2|1/2|

All 18 games, 196 hands and 5,488 plays were independently replayed. The slowest
faceoff was 5.26s, under the one-minute cap. These are integration observations;
one pair cannot establish a strength ranking or a difference between methods.

Final trained research specimens, not promoted incumbents:
[GRPO](https://huggingface.co/jasonyandell/gus/tree/87a2737abd34d432974b7da10ea31727d648e6f8)
and [rehearsal](https://huggingface.co/jasonyandell/gus/tree/c25277a923f28aa07c17cde097d1547cc42ddfa2).
Each export passed fresh anonymous hash verification and both-head tensor/logit
parity. [Canonical tournament records](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/43a545a675d16c8cde199995ee98041f1d296751)
are separately published and download-verified. Full recovery, selected checkpoints, native captures, plans, trajectories,
confirmation, costs and canonical tournament receipts: [public CC0 archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/2fbbc3e01e73b444bb2f37b9cbbdcc194da95683/lem/gus-full-hand-outcomes-2026-10-18).
All 1,370 archive files (656,016,015 bytes) passed fresh anonymous SHA256
verification. A resolver rate limit interrupted the first download; respecting
Retry-After and fetching only the 207 missing files completed the same revision.
The manifest and replay-verification receipt distinguish complete file verification
from the twelve actual recomputations. The main Hub incumbent stays
unchanged. Live training curves: [GRPO](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/3afbfc65),
[control](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/3bec63a5).
