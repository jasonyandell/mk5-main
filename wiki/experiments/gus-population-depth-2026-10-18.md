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
After the controller finishes reserved confirmation, run
`scripts/cpu-slot --threads 2 -- scratch/venv/bin/python -u -m lem.observer.audit --root scratch/lem/gus-population/campaign`, then
`scratch/venv/bin/python -u -m lem.observer.release --root scratch/lem/gus-population/campaign`
publishes the preselected final-depth specimen, confirmation evidence and
`campaign-selected-20261018` model tag. Publication checks the requested size,
completed final-depth plateau, unchanged selected checkpoint and complete panel;
anonymous downloads verify every release file and the confirmation replay.
An offline Hub round-trip test checks selected-versus-latest weight identity,
adapter/safetensors parity, example logits, retry idempotency and refusal to
publish before campaign completion.
The archive audit independently reconstructs every saved learner prefix from
own initial tiles and public plays, checks legal outputs/rewards and checkpoint
chains, verifies the rolling population and deal exclusions, and requires
actual finite weight changes. A corruption test updates a polluted input's
checksum and confirms that the semantic audit still rejects it. Final publication
requires an audit covering every completed generation's exact checkpoint.
The development audit also replays reported wins and verifies that any hand's
first changed action belongs to the learner, before which all frozen players
must behave identically. A regression test rejects an otherwise lawful replay
in which a frozen seat changes first.
`plot.py` renders the completed-generation score/parameter chart as PNG and PDF,
with exact values in `progress.json`. Matplotlib runs in a separate plotting
environment, leaving the training runtime unchanged. Rerender after confirmation;
final publication rejects a stale chart and archives the refreshed figure/data.

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

Registration preceded the first population-training outcome. G1–G16 each completed all
4,096 training hands, eight collection/update blocks, the 256-hand panel and both two-game smokes.
Every recorded hand replayed lawfully; the smokes had no source/training/panel
deal overlap. Four generations without a two-point gain triggered the registered
plateau at depths two, four, eight and twelve; the campaign now continues at depth fifteen.

| Specimen | Blocks / parameters | Development team wins | Difference from G0 |
|---|---|---:|---:|
| G0 | 2 / 107,392 | 51.95% | reference |
| G1 | 2 / 107,392 | 50.00% | −1.95 pp [−5.08, +1.17] |
| G2 | 2 / 107,392 | 50.00% | −1.95 pp [−5.08, +1.17] |
| G3 | 2 / 107,392 | 52.73% | +0.78 pp [−2.73, +4.30] |
| G4 | 2 / 107,392 | 52.73% | +0.78 pp [−2.73, +4.30] |
| G5 | 4 / 207,360 | 52.73% | +0.78 pp [−3.13, +4.69] |
| G6 | 4 / 207,360 | 51.17% | −0.78 pp [−4.69, +3.13] |
| G7 | 4 / 207,360 | 51.95% | 0.00 pp [−3.91, +3.91] |
| G8 | 4 / 207,360 | 50.78% | −1.17 pp [−5.08, +3.13] |
| G9 | 8 / 407,296 | 52.34% | +0.39 pp [−3.13, +4.30] |
| G10 | 8 / 407,296 | 53.13% | +1.17 pp [−2.34, +4.69] |
| G11 | 8 / 407,296 | 52.73% | +0.78 pp [−2.73, +4.30] |
| G12 | 8 / 407,296 | 51.17% | −0.78 pp [−3.91, +2.34] |
| G13 | 12 / 607,232 | 53.13% | +1.17 pp [−2.73, +5.47] |
| G14 | 12 / 607,232 | 50.00% | −1.95 pp [−5.86, +1.95] |
| G15 | 12 / 607,232 | 48.83% | −3.13 pp [−6.64, +0.78] |
| G16 | 12 / 607,232 | 46.88% | −5.08 pp [−9.38, −0.78] |

The interval is the registered whole-hand paired bootstrap on the repeatedly
used development panel; it is exploratory. G1 did not improve the development
score. The separate confirmation panel remains unused. G1's two full-game smokes
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

G2 completed the same registered budget and checks, with no development gain.
[G2 model](https://huggingface.co/jasonyandell/gus/tree/965cb7d33e38e2a5f27518e50c5a2dc46b233b54)
and [G2 evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0d4f472e43c0a7bca309f93a2c46a55fca732b51/lem/gus-depth-campaign-2026-10-18/g002)
passed anonymous weight/replay verification.

G3 supplied the retained best weights for the first growth. G4 tied its score;
the registered material-improvement anchor remained G0's 51.95%, so four
consecutive non-material generations triggered growth. The +0.78-point best
development difference does not establish a strength improvement.
[G3 model](https://huggingface.co/jasonyandell/gus/tree/gus-g003),
[G4 model](https://huggingface.co/jasonyandell/gus/tree/1f367791e382020165226eac78b307049d62e5d9)
and [complete first-depth evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/262a24d063a40d1476c6cfa135dafacecf5ae836/lem/gus-depth-campaign-2026-10-18).

G5 begins at four blocks and **207,360 parameters**. Its initialization produced
bit-identical MPS logits to G3 at every position of 512 fixed validation-view
sequences (maximum absolute difference zero). The independent archive audit
verified all **16,384 hands and 114,688 learner decisions** from G1–G4.
[Audits and growth initialization](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/a8bedd32415179d0ac82630fe6138350a36c46c7/lem/gus-depth-campaign-2026-10-18)
were anonymously downloaded and hash-verified. G5 training is tracked in
[W&B](https://forge.coreweave.com/wandb/jasonyandell-forge42/gus-population/runs/f5e1911c).
Its first completed iteration matched the inherited development score;
[G5 weights](https://huggingface.co/jasonyandell/gus/tree/83646a18c39876666bc24fa0aa24be3cd26f54c0)
and [full formation](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/7601f75e1b40f0085508dbdad93f2abb9c1775a2/lem/gus-depth-campaign-2026-10-18/g005)
passed anonymous verification.

Equal win rates do not imply equal policies. G5 changed the play of 184 of the
256 development hands versus G0: 14 losses became wins and 12 wins became
losses. The net two hands produce the +0.78-point difference. Every first
divergence belonged to the learner. The expanded audit verified all 20,480
training hands and 143,360 learner decisions through G5, including the exact
evaluation outcomes and fixed-player consistency.
[Audit through G5](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/476380e3046a8f0d15f0a59bf5eb42bd63633fbc/lem/gus-depth-campaign-2026-10-18/audit-through-g005.json).

G6 scored 51.17%, leaving the retained best at 52.73% and the depth-four
non-improvement counter at two.
[G6 weights](https://huggingface.co/jasonyandell/gus/tree/b9b216ee129a159999ba76d5f9ff54a9883aa7a9)
and [formation](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/eb0704eebed4dca10d1cbb2920c3985a7aa9dd3b/lem/gus-depth-campaign-2026-10-18/g006)
passed anonymous verification. No selection uses the reserved confirmation panel.

G7 recovered to 51.95% and G8 fell to 50.78%; four depth-four generations failed
to exceed the inherited best. G9 therefore grew to **407,296 parameters** from
the retained G5 initialization, which preserves G3's function. None of the
reward-trained depth-four candidates surpassed that inherited score.
[Complete second-depth evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/03a8dba92737989f40ffa4874be177a4216c8c10/lem/gus-depth-campaign-2026-10-18).

The eight-layer initialization was bit-identical to its parent over all logits
at all positions of the same 512 MPS probe sequences. An independent audit
verified all **32,768 training hands and 229,376 learner decisions** through G8,
including evaluation replay and frozen-player consistency.
[Growth and audit receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/7d10e82e95ff088d05268739f3bdfb6da92e1be1/lem/gus-depth-campaign-2026-10-18).

G9's first completed iteration scored 52.34%, below the retained 52.73% best.
[G9 weights](https://huggingface.co/jasonyandell/gus/tree/434ed1511ef7cbc5207717dd38b680d9b308a373)
and [formation](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/e25f23a90979842af6de85ddfc2576eebbeb6669/lem/gus-depth-campaign-2026-10-18/g009)
passed anonymous verification.

G10 set the retained development high at **53.13%**, a small increase below the
two-point reset threshold. G11 scored 52.73% and G12 scored 51.17%, completing
four depth-eight generations without a material gain. G10's trained weights,
not an identity initialization, supplied the next growth.
[G10 model](https://huggingface.co/jasonyandell/gus/tree/ac3fbdaeb81d7f5ed8de35eaf20800a6ee35315f),
[G11 model](https://huggingface.co/jasonyandell/gus/tree/09bd64fe5a06f0ab44bfe94321003caacb66a893),
[G12 model](https://huggingface.co/jasonyandell/gus/tree/b286cc4ce528d1ef8ab99a44f334c9e01ca01447)
and [third-depth formation](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/4c485efe7044427720a79bd776ae0e2aada4219c/lem/gus-depth-campaign-2026-10-18).

G13 starts at twelve layers and **607,232 parameters**. Its initialization was
bit-identical to G10 over all logits on the 512-sequence MPS probe. The complete
audit through G12 verified **49,152 hands and 344,064 learner decisions**, plus
the development outcomes and frozen-player consistency.
[Growth, audit and twelve-generation chart](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0f00771337756fc96829487dc495fd0ae6b7b7d3/lem/gus-depth-campaign-2026-10-18)
passed anonymous download verification. The chart uses a common percentage scale
and separates observed scores from the retained maximum; confirmation is reserved.

G13 tied the retained high; G14–G16 scored below it. The twelve-layer phase
therefore reached its fourth consecutive non-material iteration. The retained
specimen is G13's identity-grown initialization of G10. The fresh confirmation
panel remains unused.
[G13 weights](https://huggingface.co/jasonyandell/gus/tree/gus-g013),
[G14 weights](https://huggingface.co/jasonyandell/gus/tree/gus-g014),
[G15 weights](https://huggingface.co/jasonyandell/gus/tree/gus-g015) and
[formation through G15](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/786465330fe38507b4efa7d8b968a2d91cf53fac/lem/gus-depth-campaign-2026-10-18)
passed anonymous weight/replay verification.

G17 begins at the final depth: **15 layers and 757,184 parameters**. All existing
tensors were copied exactly; the three new blocks' output projections are zero.
All logits at all positions are bit-identical to the parent on 512 fixed MPS
probe sequences. The independent archive audit verified **65,536 hands and
458,752 learner decisions** through G16, including public input reconstruction,
rewards, checkpoint chains and development-table consistency.
[G16 model](https://huggingface.co/jasonyandell/gus/tree/84c9fd01fbd087aaca25cd10ce29d8b372334f40),
[formation through G16](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/9a3bf045972a7627def14280ee14e0f462e9baea/lem/gus-depth-campaign-2026-10-18),
and [audit and 757k growth receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/72a6539f3ac9677f89f28d6515754d353d3639bc/lem/gus-depth-campaign-2026-10-18)
passed anonymous download verification. The final-depth phase still must reach
its registered plateau before the reserved confirmation and final selection
publication; reaching the parameter target alone does not finish the campaign.

## Measured iteration time

A generation is 4,096 fresh training hands followed by evaluation, not another
pass over the original pretraining corpus. Collection and optimization alternate
in eight blocks. The original [[lem-observer-corpus-squeeze-2026-10-17]] pretraining
took 21 minutes 44 seconds; the smaller outcome-learning iterations have these
measured costs:

| Work | G1, 107k | G9, 407k | G15, 607k |
|---|---:|---:|---:|
| Collect 4,096 hands | 178.60 s | 191.84 s | 182.07 s |
| Optimize, 448 updates | 5.07 s | 14.43 s | 18.31 s |
| All eight training workers, including local overhead | 187.47 s | 212.12 s | 208.05 s |
| 256-hand development panel plus prediction diagnostics | 11.43 s | 11.50 s | 12.09 s |
| Two parent games plus two Zeb games | 6.98 s | 7.53 s | 9.22 s |
| Training workers + evaluation + smokes | 205.88 s | 231.15 s | 229.35 s |

At G15, optimizer work averages 40.9 ms per update. Experience generation,
including the frozen players' moves, dominates the iteration. The total excludes
slot waiting, subprocess startup outside the reported worker timers, W&B and
Hugging Face publication. It is descriptive wall time on the shared machine,
not a player thinking budget or an end-to-end latency guarantee. CPU instructions
remain in each work unit's cost log. Exact receipts are each generation's
`block-*/complete.json`, `development/report.json` and `smoke-*/report.json` in the
linked formation archive.

The hypothesis is improvement against a fixed mixed table after outcome learning;
no native-unit meaning or clean-share claim follows from a reward increase.
Scope: this sequence model, original-own-hand and within-hand public history,
this frozen/rolling policy population, bid-30 pip contracts, and no persistent
player identity or cross-hand memory. [[clean-neurons-42]] remains an independent
interpretability test over the preserved formation evidence.
