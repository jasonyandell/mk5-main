---
title: "Gus population reward training and depth ladder through 757k parameters"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---

## Completed outcome

Twenty reward-learning generations reached the registered plateau at all five
depths, ending at **15 layers and 757,184 parameters**. The fresh confirmation
found no established playing-strength improvement over G0. All trained
specimens, formation checkpoints and learner inputs/outputs are public.

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
The final model card explicitly distinguishes an identity-grown selection from
a fully trained candidate and links the last trained final-depth specimen.
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

## Fresh confirmation and selected specimen

Selection was fixed before the fresh 512-hand panel. The same deals and frozen
G0/Zeb partners/opponents were used for both policies, with one learner-controlled
seat and legal argmax. The panel remained excluded from pretraining and all
reward-training hands.

| Policy | Team wins | Win rate |
|---|---:|---:|
| G0 pretrained | 235 / 512 | 45.90% |
| Selected Gus | 234 / 512 | 45.70% |

Paired difference: **−0.20 percentage points**, 95% whole-hand bootstrap interval
**[−2.73, +2.34]** (2,000 resamples, seed 42). This does not establish improvement
or equivalence. The interval concerns this fixed mixed-table setting, not a
universal tournament rating. [Confirmation report](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/final.json).

The selected file is **G17's initialization**, preserving G10's learned eight-layer
function inside a fifteen-layer architecture. Seven appended residual blocks
have zero output projections. This is the registered best-weight retention rule,
not evidence that the selected policy learned to use all fifteen layers. G17–G20
were fully trained at fifteen layers; all fifteen blocks changed weights, and
all initially zero output projections became nonzero. Those trained candidates
remain separately available for [[clean-neurons-42]]. No selection was changed
after confirmation. [Selection and growth audit](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/completion-checks.json).

The selected policy changed play in **371 of 512** confirmation hands. Twenty-two
G0 losses became wins; twenty-three wins became losses. Every first divergence
belonged to the learner, with all frozen players identical until its changed
choice. A near-zero net score therefore does not imply an unchanged policy.

On the separate, fixed original-validation prediction probe (57,344 next-tile
targets), G0 accuracy/NLL were **52.95% / 1.2601**; selected Gus was
**48.67% / 1.4173**. These measure retention of recorded behavior, not team wins.
The reward objective included no supervised auxiliary loss.

## Development trajectory

Every generation completed 4,096 new hands and eight collection/update blocks.
These 256-hand scores were repeatedly reused for selection; their paired
intervals are exploratory. The retained maximum rose from 51.95% to 53.13%,
first attained by G10. No depth obtained a two-point material gain; each
therefore stopped after four generations under the predeclared rule.

| Specimen | Blocks / parameters | Development team wins | Difference from G0; paired 95% interval |
|---|---|---:|---:|
| G0 | 2 / 107,392 | 51.95% | reference |
| [G1](https://huggingface.co/jasonyandell/gus/tree/gus-g001) | 2 / 107,392 | 50.00% | -1.95 pp [-5.08, +1.17] |
| [G2](https://huggingface.co/jasonyandell/gus/tree/gus-g002) | 2 / 107,392 | 50.00% | -1.95 pp [-5.08, +1.17] |
| [G3](https://huggingface.co/jasonyandell/gus/tree/gus-g003) | 2 / 107,392 | 52.73% | +0.78 pp [-2.73, +4.30] |
| [G4](https://huggingface.co/jasonyandell/gus/tree/gus-g004) | 2 / 107,392 | 52.73% | +0.78 pp [-2.73, +4.30] |
| [G5](https://huggingface.co/jasonyandell/gus/tree/gus-g005) | 4 / 207,360 | 52.73% | +0.78 pp [-3.12, +4.69] |
| [G6](https://huggingface.co/jasonyandell/gus/tree/gus-g006) | 4 / 207,360 | 51.17% | -0.78 pp [-4.69, +3.12] |
| [G7](https://huggingface.co/jasonyandell/gus/tree/gus-g007) | 4 / 207,360 | 51.95% | +0.00 pp [-3.91, +3.91] |
| [G8](https://huggingface.co/jasonyandell/gus/tree/gus-g008) | 4 / 207,360 | 50.78% | -1.17 pp [-5.08, +3.12] |
| [G9](https://huggingface.co/jasonyandell/gus/tree/gus-g009) | 8 / 407,296 | 52.34% | +0.39 pp [-3.12, +4.30] |
| [G10](https://huggingface.co/jasonyandell/gus/tree/gus-g010) | 8 / 407,296 | 53.12% | +1.17 pp [-2.34, +4.69] |
| [G11](https://huggingface.co/jasonyandell/gus/tree/gus-g011) | 8 / 407,296 | 52.73% | +0.78 pp [-2.73, +4.30] |
| [G12](https://huggingface.co/jasonyandell/gus/tree/gus-g012) | 8 / 407,296 | 51.17% | -0.78 pp [-3.91, +2.34] |
| [G13](https://huggingface.co/jasonyandell/gus/tree/gus-g013) | 12 / 607,232 | 53.12% | +1.17 pp [-2.73, +5.47] |
| [G14](https://huggingface.co/jasonyandell/gus/tree/gus-g014) | 12 / 607,232 | 50.00% | -1.95 pp [-5.86, +1.95] |
| [G15](https://huggingface.co/jasonyandell/gus/tree/gus-g015) | 12 / 607,232 | 48.83% | -3.12 pp [-6.64, +0.78] |
| [G16](https://huggingface.co/jasonyandell/gus/tree/gus-g016) | 12 / 607,232 | 46.88% | -5.08 pp [-9.38, -0.78] |
| [G17](https://huggingface.co/jasonyandell/gus/tree/gus-g017) | 15 / 757,184 | 53.12% | +1.17 pp [-2.73, +5.08] |
| [G18](https://huggingface.co/jasonyandell/gus/tree/gus-g018) | 15 / 757,184 | 51.95% | +0.00 pp [-3.91, +3.91] |
| [G19](https://huggingface.co/jasonyandell/gus/tree/gus-g019) | 15 / 757,184 | 49.61% | -2.34 pp [-5.87, +1.17] |
| [G20](https://huggingface.co/jasonyandell/gus/tree/gus-g020) | 15 / 757,184 | 49.22% | -2.73 pp [-6.64, +1.17] |

[Completed score and parameter chart](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/resolve/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/progress.png) separates each trained generation
from the retained maximum and reports the independent confirmation below it.

## Formation and completion checks

The independent archive audit reconstructed **81,920 training hands and 573,440
learner decisions**, including literal visible-prefix tokens, own legality,
replayed team rewards, immutable population hashes, optimizer settings and all
checkpoint chains. Training performed **8,960 optimizer updates**. Each of the
sixteen seat/bidder-role combinations occurred equally within every generation.
Reward-training deals were unique and excluded the original source corpus and
both reserved panels; the two panels were mutually disjoint.
[Full formation audit](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/audit.json).

All four growths copied existing tensors exactly and preserved every tile logit
at every position of 512 fixed MPS probe sequences. The separate completion
check recomputed every stopping decision, retained-best transition and growth
parent; replayed both confirmation policies and the paired interval; and checked
that critical training and pinned engine sources stayed unchanged.
[Completion checker and receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/verification).

The full-game runs were exactly **80 integration-smoke games**: two against the
parent and two against pinned Zeb per generation. Every play and score was
replayed; none overlapped any source, training or panel deal. They do not supply
a strength estimate. W&B smoke summaries were copied from those saved reports
and read back for all twenty runs, with no additional games.
[Tracking receipt](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/wandb-smoke-verification.json).

Anonymous Hub metadata verification checked all **1,466 required formation files**
(1,031,082,800 bytes) by size and Git/LFS content hash, plus every generation and
selection tag. This was an object-metadata check, not a fresh download of every
formation byte. Each generation separately verified downloaded weights and a raw
replay; final publication freshly downloaded and verified every model-release
file and the confirmation replay. The supplemental checker scripts and receipts
were also freshly downloaded and hash-verified.
[Archive verification](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/remote-archive-verification.json).
The 59-test Lem suite, wiki, license and conflict-marker gates passed.

## Published artifacts

- [Selected Gus](https://huggingface.co/jasonyandell/gus/tree/e6b88165cc017594f09f62d55c2d28ea55b450d3), tag `campaign-selected-20261018`: best development
  specimen fixed before confirmation; identity-grown G10 policy.
- [G20](https://huggingface.co/jasonyandell/gus/tree/gus-g020): last fully trained
  757k candidate. G17–G20 tags preserve all four final-depth training outcomes.
- [Complete formation and confirmation archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18): twenty initial
  checkpoints, 160 optimizer checkpoints, input/output tensors, full games,
  fixed panels, growth checks, costs and publication receipts.
- [W&B final training run](https://forge.coreweave.com/wandb/jasonyandell-forge42/gus-population/runs/3ea12bc4);
  project `jasonyandell-forge42/gus-population` contains all twenty runs.
- [[gus-depth-campaign-evidence-2026-10-18]] records immutable release and source pins.

## Measured iteration time

A generation collects fresh experience; it is not another epoch over the original
pretraining corpus. The original [[lem-observer-corpus-squeeze-2026-10-17]]
pretraining took 21 minutes 44 seconds. Reward-generation costs were:

| Work | G1, 107k | G9, 407k | G20, 757k |
|---|---:|---:|---:|
| Collect 4,096 hands | 178.60 s | 191.84 s | 175.89 s |
| Optimize, 448 updates | 5.07 s | 14.43 s | 22.60 s |
| All eight training workers, including local overhead | 187.47 s | 212.12 s | 207.56 s |
| 256-hand panel plus prediction diagnostics | 11.43 s | 11.50 s | 11.44 s |
| Two parent games plus two Zeb games | 6.98 s | 7.53 s | 8.32 s |
| Training workers + evaluation + smokes | 205.87 s | 231.15 s | 227.32 s |

The final two 512-hand confirmation evaluations took **41.34 seconds** combined,
including prediction diagnostics. These descriptive wall times exclude slot
waiting, subprocess startup outside the worker timers and publication. CPU
instructions remain in the archived cost logs. No player used time as its
thinking budget; experience generation dominated the iteration.

## Interpretation boundary

This completed campaign tests one training recipe and operational stopping rule,
with original-own-hand and within-hand public-history inputs, the fixed/rolling
policy populations, bid-30 pip contracts and one learner seat. It does not
establish a capacity ceiling, an optimal population, equilibrium, human-player
modeling, native neuron meanings, or clean share. Nello, splat prediction,
learned bidding and persistent player identity remain separate work in
[[lem-nanogpt-foundations]]. The preserved formation is ready for interpretation;
no interpretability claim follows from reaching the parameter target.
