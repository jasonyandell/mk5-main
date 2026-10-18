---
title: "Gus: four times the outcome-training exposure"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and finding

Does four times the fresh on-policy outcome-training exposure improve
[[gus-tiny-gpt]] beyond its own short prefix and matched supervised rehearsal?
The preceding [[gus-staged-outcomes-2026-10-18]] supplies the training-only
motivation: sparse mixed-outcome groups despite diverse paths, with most prior
endgame skill retained. The completed comparison establishes no fresh full-hand
gain from the larger budget. Both registered criteria fail; the supported incumbent remains unchanged.
This is one seed and recipe, not a capacity ceiling or a verdict against RLVR.

## Registered comparison

Two continuous runs start from the original 185,652-parameter splat parent:
full-hand GRPO and supervised rehearsal alone. Each receives 1,024 fresh physical
training roots, eight draws per root, 128 rounds and 256 AdamW updates. Its own
round-32 prefix supplies the short-budget control: 256 roots and 64 updates.
The long run keeps AdamW and RNG continuously; a new optimizer is not introduced
at that boundary. This varies fresh data and update count together as an exposure
budget, without identifying their separate effects.

Architecture, clean own-seven/public-history inputs, bid30/pip task, fixed G0
partner, pinned Zeb opponents, original KL reference, seed 42 and learning recipe
stay unchanged: lr3e-5, clip .2, KL .02, rehearsal weight 1/batch 32, eight roots per
round, two inner updates, full seven-turn credit and evaluation every eight rounds.
Both arms collect trajectories for matched diagnostics. Rehearsal has no reward
or KL gradient. Demon is a fenced integration competitor, never a label source.

The 128-root development panel selects greedy then sampled success within the
common original-loss bounds +.02 each, including initialization and earliest ties.
At round 32, the saved formation checkpoint's best record freezes the short
selection. At round 128, the long selection freezes. Long may legitimately retain
short. Only then are five labels evaluated on 512 untouched physical roots:
original parent, short/long GRPO and short/long rehearsal. Exact inference-state
identity can share lazy records; labels do not multiply independent evidence.

The development criterion requires long GRPO to gain at least 2/128 over both
short GRPO and long rehearsal. Promotion requires greedy gains at least 2pp over
parent and 1pp over each of short GRPO and long rehearsal, every paired 95% lower
bound above zero, with original-task retention. Confirmation uses 20,000 paired
physical-deal bootstrap draws, seed 91642. Sampled outcomes are secondary; no
checkpoint is reselected after confirmation.

Each arm has at most 40 cooperative 45s segments with 90s outer caps; two live
arms maximum. Identity, numeric or replay failures stop work with state retained.
The registered budget cannot silently grow. Confirmation uses bounded lazy units;
fixed G0/Zeb/Demon integration faceoffs each retain the 60s cap.

## Registration and runtime

[Public preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/2dfc408d0971961d300da2297d1b7e43ddb894b0/lem/gus-exposure-budget-2026-10-18) preceded training. All 11 files
(12,201,421 bytes) passed fresh anonymous hashes. Registration SHA256:
`bcf1e93c7204d49044c66dedc0a71c5858082ec1067da146fae774f4e619b835`.

All 1,664 reserved physical deals exclude 600,622 prior deals, including every
reserved root and tournament schedule from the preceding study. All new reservations
become future exclusions. Root audit confirms split uniqueness and fixed seating.

Runtime source is `77e0dc44` (the unchanged trainer implementation
from `1dff0a63`, with 146 Python tests and GREEN gates). Both bounded runs completed
all 128 rounds. Protocol and recovery are in [[gus-success-probability-planning]]
and the `gus-training` skill. The ongoing investigation is
[#223](https://github.com/jasonyandell/mk5-main/issues/223).

Recorded telemetry: [GRPO](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/4b35a82a)
and [rehearsal](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/e462da85).

The [frozen evaluation protocol](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/a9548387906b37716763dcea9af5aca208a961f3/lem/gus-exposure-budget-2026-10-18/protocol) was published before confirmation:
all 16 files passed fresh anonymous hashes. It pins the five-label comparison,
independent selection reconstruction, probability tolerances, actual lazy replay
checks and separate short/long native captures. All four selections froze before
confirmation; the five distinct inference states
produced 5,120 actual trajectories. The disclosed audit correction below preserves
the original evaluation protocol.

## Fresh confirmation

| Label | Selected round | Dev greedy /128 | Fresh greedy /512 | Fresh sampled /512 |
|---|---:|---:|---:|---:|
|initial_parent|0|—|222|189|
|grpo_short|32|55|222|187|
|grpo_long|88|55|217|189|
|rehearsal_short|24|55|220|189|
|rehearsal_long|40|56|217|193|

Long GRPO versus parent is **−0.98 percentage points**, paired 95% interval
**[−3.13, +1.17]**: twelve wins gained and seventeen lost. Versus short GRPO,
the difference is −0.98pp [−2.54, +0.59]; versus long rehearsal it is 0.00pp
[−1.56, +1.56]. Sampled long-GRPO success ties the parent, 0.00pp
[−1.56, +1.56], and trails long rehearsal by −0.78pp [−1.95, +0.20].
These intervals do not establish a gain or a meaningful degradation.

All original-task retention bounds pass. Parent tile NLL/splat CE are
1.329962/2.349329; selected long GRPO reaches 1.309835/2.317055. Better imitation
losses did not establish better play. The development panel selected round 88
for long GRPO and round 40 for long rehearsal; neither is the final round 128.
Confirmation did not reselect checkpoints or promote an apparent winner.

## Training signal and output movement

Across 1,024 GRPO training roots, 511 groups lose every draw, 229 win every draw,
and 284 have mixed outcomes. Thus 72.27% of groups have zero relative reward
advantage. Six of 128 rounds have no mixed group. Only 8,746 of 25,900 unsettled
discretionary decisions carry nonzero reward credit (33.77%). All-loss groups
still average 5.42 distinct learner paths before settlement; entropy on active
choices averages .924 nats and largest-action probability .602. This does not
establish exploration collapse, nor prove those deals unwinnable.

Four native captures each cover the same 11,116 frozen development rows. Between
trained rounds 32 and 128, GRPO changes 9.85% [9.04, 10.61] of choices on 4,895
unsettled discretionary rows; rehearsal changes 8.66% [7.99, 9.41]. At round 128,
the two arms disagree on 7.07% [6.17, 7.92] of those choices. These paired
development diagnostics show policy movement, not stronger decisions. Captures
are trained boundaries, distinct from selected models; `clean-neuron` still
requires held-out causal evidence for unit meanings.

The first saved mixed group contains four wins and four losses, giving advantages
+1/−1. Its literal opening prefix is:

```text
start bid:30 you trump:4 your hand 1-0 2-2 3-2 4-3 4-4 5-1 5-4 play begins you
```

One successful sampled hand opens with 4-4. Its legal probability moves from
.886419 to .886845 across the two optimizer updates. The entire batch, KL and
rehearsal contribute; this is not isolated causal credit for that move. The
archive retains exact tokens, legal distributions and all eight outcomes.

## Replay, cost and audit correction

Independent replay verifies all 16,384 training and 5,120 confirmation hands,
114,688 common-reference decision distributions and 256 formation checkpoints.
Both 32/128 selection windows reconstruct from their own development histories.
Paired fixed-seat actions agree until the first learner divergence on 8,192
training pairs; all 64 first-round hands match completely.

Maximum CPU/MPS log-probability errors are 3.72e-5 for the reference and 3.91e-5
for formation outputs, below the unchanged 5e-5 tolerance. This is numerical
agreement. Separately, 20 actual lazy CPU units reproduce identical bytes in
reverse request order. All 1,664 reserved roots regenerate and remain excluded
from future training, including confirmation roots.

The frozen audit initially stopped because regenerated Python tuple hands/bids
were compared directly to JSON lists. A one-line JSON normalization fixes the
container comparison; all semantic fields still match exactly. The
[disclosed correction](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/89d0101fdc3e43e7dff3c5d4a5d7905221eb4c58/lem/gus-exposure-budget-2026-10-18/audit-correction)
was published before the corrected audit ran, after confirmation scores were
read. The original protocol, script and failure log remain intact. No games,
selection, statistics or numeric tolerances changed. The corrected audit passed.

| Training arm | Retired host instructions | CPU seconds | Summed segment wall seconds |
|---|---:|---:|---:|
|exposure-grpo|52.376 trillion|3029.4|1707.6|
|exposure-rehearsal|52.356 trillion|3023.1|1693.0|

Each arm completed 31 cooperative segments and retained 128 formation checkpoints.
Host instructions exclude MPS work. CPU/wall durations describe this run;
deterministic player budgets define the games.

## Fixed-competitor integration and publication

The twelve faceoffs completed 24 games, 282 hands and 7,896 independently replayed plays.
Maximum faceoff wall time was 4.68s, below the 60s cap.
Each unique selected model faced fixed G0, native Zeb and fenced Demon on a
shared schedule per opponent. One duplicate pair per matchup is integration
evidence, not a reliable strength ranking. Demon never supplies labels.

[Canonical tournament records](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/e52348943799ab761619b79bb4581e115879d048) are published and fresh-anonymous SHA256 verified: 47 combined records, 16 newly added. No completed games were rerun.

[Complete public evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/b1e119fa560dbe714a4d2310a9d3367a1cfa9ec7/lem/gus-exposure-budget-2026-10-18) contains 20 release files (1,033,725,296 bytes) and 7,041 archive members. Every release file and member passed fresh anonymous SHA256 verification.

- [grpo_long selected model](https://huggingface.co/jasonyandell/gus/tree/b961e87f6b19b94828918e7287074f30902731cc) — exact tensor/both-head CPU logit parity and fresh anonymous hashes.
- [grpo_short selected model](https://huggingface.co/jasonyandell/gus/tree/37873edafedf25e3577199f6fa874513eb9cdfd7) — exact tensor/both-head CPU logit parity and fresh anonymous hashes.
- [rehearsal_long selected model](https://huggingface.co/jasonyandell/gus/tree/85000d1e36624ad39e591307e5a191cd3cf844b0) — exact tensor/both-head CPU logit parity and fresh anonymous hashes.
- [rehearsal_short selected model](https://huggingface.co/jasonyandell/gus/tree/a8dc5d9a4a62ebadcec8d3fe0c555bb52fea06e5) — exact tensor/both-head CPU logit parity and fresh anonymous hashes.

The archive retains initialization, both selected windows, final models,
optimizer/RNG recovery, all formation, literal trajectories, native captures,
timing receipts and restore paths. The wider fixed-opponent investigation stays
open in [#223](https://github.com/jasonyandell/mk5-main/issues/223).

## Next question

Can a learned success-value baseline reduce noisy sequential credit assignment
without changing information inputs, fixed players or verifier authority?
[[gus-success-probability-planning]] records the proposed training-only diagnostic
and textbook actor-critic rationale. This comparison does not implement a critic,
prove that it helps, or justify another automatic exposure increase.
