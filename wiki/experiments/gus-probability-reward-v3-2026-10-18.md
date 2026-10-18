---
title: "Gus: three-seed probability reward and fixed-competitor transfer"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Result and next boundary

The [[gus-tiny-gpt]] probability-reward change did not improve on binary
best-set reward in the registered three-seed comparison. Both predeclared
seed42 representatives improved fresh finite-model endgame regret over the
splat-pretrained parent, with original-task retention. Short full-game checks
showed no win-count improvement. Neither is promoted as a general player.

This completes the comparison selected in [[gus-success-probability-planning]],
not Jason's broader autonomous training goal. [Issue #222](https://github.com/jasonyandell/mk5-main/issues/222)
continues to full-hand verified-outcome post-training with fixed nonlearner
policies. The existing [[gus-splat-pretraining-2026-10-18]] supplies the pretrained
representation; the exact curriculum covers only one late discretionary choice.

## Registered comparison

Training source: [`2bbf665e`](https://github.com/jasonyandell/mk5-main/tree/2bbf665ea4ced1069df17286c2666b9b3aa43b0e).
The [training archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/2df7f25c63ad8eef0f64e402eb5501a07e488079/lem/gus-probability-reward-v3-2026-10-18)
contains registration, commands, full formation and optimizer/RNG state,
selection, fresh confirmation and frozen native representations. All 145 files,
392,258,656 bytes, passed fresh anonymous SHA256 verification.

Six runs used the same original splat checkpoint, 185,652 parameters, unchanged
clean inputs/outputs, 2,048 train and 512 development questions, and fixed
lowest/highest/uniform exact continuation model. Seeds42/43/44 paired the question
and rehearsal streams. Each ran256 rounds with two updates, question batch64,
replay batch32, AdamW1e-5, KL0.02 and rehearsal weight1. Original-task loss bounds
were +0.02 each. Cooperative45s segments and90s outer caps remained resumable;
each completed training segment took10.0–12.7s. Timings are descriptive, not
planner budgets; instruction receipts accompany them.

The sole changed factor was `--reward best` versus `--reward success` in exact
expected-reward mode. Checkpoint selection included initialization, required
retention, and minimized exact development regret. Success reward required a
mean paired regret reduction of at least0.2 percentage points and improvement
in at least two of three seeds. The deployment representative was always
seed42, never the best seed after inspection. Confirmation was generated under
a new namespace with588,781 prior/supplemental deals excluded, then read only
after method selection. No confirmation-driven reselection followed.

| Seed | Binary-best regret | Probability-reward regret | Probability reduction |
|---|---:|---:|---:|
| 42 | 2.290% | 2.486% | −0.196 pp |
| 43 | 2.490% | 2.553% | −0.063 pp |
| 44 | 2.348% | 2.311% | +0.037 pp |

Mean change was −0.074pp; only one seed improved. The control was retained.
All selected checkpoints met retention. This is a finite-budget result, not
an algorithm ranking. Both rewards agree on the optimal move set; they weight
mistakes differently. Holding KL and rehearsal coefficients fixed also changes
their relative strength when probability gaps replace unit reward gaps.

## Fresh exact-model confirmation

One independently reserved physical deal per question; 512 questions. Success
and regret are exact conditional on the declared finite model, not empirical
full-game win rates. Paired95% intervals resample complete physical deals.

| Model, predeclared seed42 | Best-move accuracy | Success probability | Regret |
|---|---:|---:|---:|
| Splat-pretrained | 73.24% | 59.23% | 4.685 pp |
| Binary-best post-training | 78.91% | 61.07% | 2.850 pp |
| Probability post-training | 77.54% | 61.01% | 2.913 pp |

Binary-best regret reduction: **1.835pp [0.906, 2.744]**. Probability reward:
**1.772pp [0.989, 2.631]**. Probability versus binary-best: **−0.063pp
[−0.543, +0.427]**, no established difference. Both retain original prediction
losses; probability reward has slightly better tile NLL here. Saved `policy_return`
also distinguishes the stochastic policy from the deployed argmax.

Public model specimens, all files freshly downloaded/hashed and both output
heads checked against source tensors:

- [Binary-best v3](https://huggingface.co/jasonyandell/gus/tree/9fd90ffabff2068c1c649b66e99e1c56f0b9a629), tag `rlvr-best-v3-20261018`, round224.
- [Probability v3](https://huggingface.co/jasonyandell/gus/tree/b3ff20eaa103472f6dff08ef476acedc3515a4f0), tag `rlvr-success-v3-20261018`, round160.

The general model main revision is unchanged. The archive contains all six runs,
not just these representatives. W&B examples: [binary-best seed42](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/d9644fa2)
and [probability seed42](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/da981886).

## Actual play: fixed competitors and paired seeds

The existing [[tournament]] now seats joint Gus through a separately pinned
adapter, with explicit tile/splat deployment head. Original tile-only participant
pins remain valid. [[the-demon]] is a separate common-bid30 competitor, preserving
its100-world root-reuse cardplay and source trees. Its native auction is replaced
by the common bidder, so ordinary-auction performance claims do not transfer.
No internal Demon estimate enters training; [[strategy-fusion-fence]] remains.

All three model versions played the same thirteen potential deals/auctions for
each competitor, reserved before results and excluded from prior training/test
data. Only the candidate partnership changed. The Demon's `paired-deal-v1` seeds
exclude candidate names and claim timestamps, preserving other-player draws
across models. No field weights or deterministic budgets evolved. Each matchup
was one duplicate pair; all18 games finished, maximum faceoff wall time5.81s.
An independent replay checked199 hands,5,572 legal plays, every schedule, score
and terminal result. No timeout or incomplete hand was scored as a loss.

| Candidate | G0 | Native Zeb | Demon |
|---|---:|---:|---:|
| Splat-pretrained | 1/2 | 0/2 | 0/2 |
| Binary-best post-training | 1/2 | 0/2 | 0/2 |
| Probability post-training | 1/2 | 0/2 | 0/2 |

One pair per matchup cannot estimate population strength or rank Zeb against
Demon. The complete paired mark differences and small-panel rating fit remain
in the evidence. The result motivates broader outcome-training coverage; it
does not justify changing opponents or declaring model capacity exhausted.

[Game evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/293f87034683b02e736bc22aa9dc29e476290c5e/lem/gus-probability-v3-tournament-2026-10-18)
contains46 files,569,148 bytes, all freshly downloaded and hashed. The6 participant
and9 faceoff records also reached the
[canonical tournament store](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/027d6729f3f97b85c5c0ae8143faf156e772d229),
with fresh anonymous checks. Adapter/harness source is
[`ecd610b7`](https://github.com/jasonyandell/mk5-main/tree/ecd610b7).
The repository gate passed118 tournament tests,8 fence tests, wiki and licenses.
