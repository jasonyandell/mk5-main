---
title: "Gus population reward training and depth ladder through 757k parameters"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: active
---

## Question and authorization

Jason requested execution of [[gus-population-iteration]] until a plateau, then
more transformer layers through **709k–800k parameters** (2026-10-18; #213).
[[gus-tiny-gpt]] begins with the published G0 pretrained weights. The original
own-hand/public-history inputs, 46-token encoding and 28 tile outputs stay fixed.
The last five completed Gus generations remain active alongside permanently
pinned Zeb, including mixed partners. All formation snapshots remain archived.

## Registered stopping and growth rule

| Blocks | Width | Heads | Parameters |
|---|---:|---:|---:|
| 2 | 64 | 2 | 107,392 |
| 4 | 64 | 2 | 207,360 |
| 8 | 64 | 2 | 407,296 |
| 12 | 64 | 2 | 607,232 |
| 15 | 64 | 2 | 757,184 |

Each generation runs eight fresh 512-hand collection/update blocks: 4,096
hands, four PPO passes per block unless the registered KL guard stops earlier.
Other seats independently sample 25% Zeb, 75% uniform over the recent-five Gus
pool. The learner alone samples its legal softmax and receives policy gradients;
other Gus seats use legal argmax. The pool is fixed for a generation. Fresh
AdamW state starts each generation; optimizer state persists across its blocks.
Exact reward/loss settings are in [[gus-population-iteration]].

Before any learning, freeze **256 development hands** and a separate **512-hand
confirmation panel**, with balanced learner seats and bidder roles. Each table's
other seats are fixed G0/Zeb, sampled with the same 75%/25% mixture. All panel
deals are excluded from the original 500k corpus and every training generation.
The candidate controls one seat, using legal argmax; measure team contract wins.
The fixed table makes scores comparable as the training population changes.

Operational plateau: **four consecutive completed generations without gaining
at least two percentage points** over the last material-improvement anchor.
Small improvements can accumulate to reset that anchor. Separately retain the
highest development score and its weights, even if its gain is below two points.
This is a practical development stopping rule, not evidence of zero remaining
improvement or a capacity ceiling. Repeated adaptive panel use is exploratory.

At a plateau, grow the best development specimen to the next registered depth.
Append blocks with zero attention and MLP output projections: each new residual
block initially acts as identity. Preserve existing tensors, save this growth
initialization, then train the whole model. At 15 blocks, continue until the
same plateau rule holds. Only then evaluate the selected specimen and G0 on the
fresh 512-hand confirmation panel, with a whole-hand paired bootstrap interval.
No checkpoint selection follows that confirmation outcome.

Full-game H2H stays **one duplicate pair against the parent and one against
Zeb per generation**. Those four games are smoke tests, not promotion evidence.
The separate single-hand panel is the learning/stopping instrument. All smokes
are audited against source, training and reserved-panel physical deals.

## Implementation and validation

`lem/observer/population.py`: visible-prefix features, mixed-seat collector,
independent public-record replay, team reward and policy-only PPO.
`reward.py`: one bounded collection/update block or one panel evaluation.
`campaign.py`: persistent generation state, recent-five pool, separate best and
progress anchors, depth growth, W&B and resumable subprocess orchestration.
`release.py`: versioned model releases and allowlisted full trajectories.
After the controller finishes reserved confirmation,
`python -u -m lem.observer.release --root scratch/lem/gus-population/campaign`
publishes the preselected final-depth specimen, confirmation evidence and
`campaign-selected-20261018` model tag. Publication checks the requested size,
completed final-depth plateau, unchanged selected checkpoint and complete panel;
anonymous downloads verify every release file and the confirmation replay.

Work units run through `scripts/cpu-slot --threads 2` with the default 300-second
cap; the orchestrator holds no slot while waiting or uploading. No game chooses
an action from elapsed wall time. The deterministic seeds/specs and model/field
hashes are retained. A terminal job failure preserves receipts for inspection.
Each work unit records the executing Git commit and observer source-file hashes;
the archive includes those receipts, cost logs and the G0 panel's replay evidence.

Tests cover team reward signs, mixed-seat ownership, causal input/own-legality
invariance to hidden hands, stored action-probability parity, policy-gradient
direction and forced-move zero gradient, identity-preserving growth, the 757k
parameter count, disjoint deal generation, pool retention, plateau patience,
and equality after block-boundary resume. Existing observer/referee tests remain.

A fixed random sample of 512 original validation deals supplies next-tile
retention diagnostics. These are behavior-prediction scores, separate from team
return; no supervised auxiliary loss is used in the initial pilot. The original
source corpus is already archived by [[lem-observer-corpus-squeeze-2026-10-17]].

## Evidence and result boundary

Campaign directory: `scratch/lem/gus-population/campaign`.
Every generation is published under a `gus-gNNN` tag of
[jasonyandell/gus](https://huggingface.co/jasonyandell/gus); full block checkpoints,
input/output decisions, game replays, metrics and panel specifications go to
`lem/gus-depth-campaign-2026-10-18/` in the existing research dataset.
W&B project: `jasonyandell-forge42/gus-population`, with hands seen as the axis.

Registration preceded the first population-training outcome. G1 completed all
4,096 training hands, eight updates, the 256-hand panel and both two-game smokes.
Every recorded hand replayed lawfully; the smokes had no source/training/panel
deal overlap. The campaign continues; G1 alone does not establish a plateau.

| Specimen | Blocks / parameters | Development team wins | Difference from G0 |
|---|---|---:|---:|
| G0 | 2 / 107,392 | 51.95% | reference |
| G1 | 2 / 107,392 | 50.00% | −1.95 pp [−5.08, +1.17] |

The interval is the registered whole-hand paired bootstrap on the repeatedly
used development panel; it is exploratory. G1 did not improve the development
score. The separate confirmation panel remains unused. Both full-game smokes
split 1–1; they are integration checks, not evidence of equal strength.

G1's eight training blocks used 178.60 seconds for collection, 5.07 seconds for
optimizer work and 187.47 seconds total inside the worker processes. Development
evaluation used 11.43 seconds. These exclude shared-slot waiting and publication;
wall times are descriptive, with CPU instructions retained in the cost logs.

[G1 model](https://huggingface.co/jasonyandell/gus/tree/6e4f09e4dbd5c3aeca7bcb76545c8d4d6a6f4f14)
and [formation/panel/smoke evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1362807a377256a42bacb73a8620eae38bd91728/lem/gus-depth-campaign-2026-10-18)
are public. The generation archive is 16,105,574 bytes, plus shared panel files;
fresh anonymous downloads verified the exported weights and a raw hand replay.
[W&B G1](https://forge.coreweave.com/wandb/jasonyandell-forge42/gus-population/runs/f9414e5b).

The hypothesis is improvement against a fixed mixed table after outcome learning;
no native-unit meaning or clean-share claim follows from a reward increase.
Scope: this sequence model, original-own-hand and within-hand public history,
this frozen/rolling policy population, bid-30 pip contracts, and no persistent
player identity or cross-hand memory. [[clean-neurons-42]] remains an independent
interpretability test over the preserved formation evidence.
