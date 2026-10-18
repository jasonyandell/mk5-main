---
title: "Gus: three quick retention and capacity iterations"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and verdict

This is the **three-iteration, five-arm retention study**, separate from the
older twenty-generation population/depth campaign. [[gus-tiny-gpt]] can learn
more exact endgame answers without repeating the prediction-loss deterioration
in [[gus-rlvr-silas-2026-10-18]]. The selected method directly sums expected
binary verifier reward and rehearses the original supervised task. The original
185,652-parameter model remains sufficient for this bounded experiment; one
extra block earns no development-regret improvement. No general-player
promotion follows. [Issue #219](https://github.com/jasonyandell/mk5-main/issues/219).

The fixed protocol predates learning: registration SHA256
`03ffd1e246afddfac58f24a9067124e87ec6874022d6777d1e5e08e4684b7a88`.
Implementation is commit `3a6a4af6a02e4a07d5d09139cb0b130a4859fb7c`;
`lem/observer/rlvr/train.py`, `rehearsal.py` and `data.py` carry the mechanisms.
The public archive below contains the registration, commands and all receipts.

## The pieces and why they fit

| Piece | Choice and criterion | Alternative and boundary |
|---|---|---|
| Representation | Existing history-conditioned GPT and unchanged clean prefix | Added inputs would confound a learning-method comparison; no hidden-world or verifier features |
| Verifier | Enumerate both legal answers; exact rational team success under the declared finite population | Unrestricted 42 is not solved; the uniform legal-world belief ignores pre-root behavior likelihood |
| Retention | Original tile CE plus splat histogram CE on train-only rehearsal | Freezing the splat head alone cannot protect its changing shared backbone |
| Learning baseline | Complete best-set SFT with the same parent and question/update budget | Cheaper complete labels make this a necessary comparator to RL machinery |
| Sampled RL | GRPO groups of eight, normalized rewards, clipped old-policy ratio, fixed-parent KL | Useful when answers must be sampled; here all two rewards are already available |
| Selected RL objective | Direct finite sum of binary exact-best reward plus fixed-parent KL | No learned critic or sampled-action estimator is needed; this is a different objective from GRPO's normalized clipped surrogate |
| Capacity | Append one identity-initialized residual block; preserve both initial heads | Accept only a material development gain within the registered instruction-cost bound |
| Selection | Retention constraints first, exact development regret second; include initialization | Choosing by endgame accuracy alone previously hid deterioration of the old task |
| Confirmation | Frozen repeated panel plus a separately reserved fresh 512-deal readout | Four actual games remain an adapter/transfer diagnostic, not a strength estimate |

[Experience replay](https://arxiv.org/abs/1811.11682) motivates preserving old
experience during new learning. This implementation replays **supervised**
examples, not stale RL action probabilities through an on-policy update.
[DeepSeekMath](https://arxiv.org/abs/2402.03300) supplies GRPO's group-relative
baseline; [PPO](https://arxiv.org/abs/1707.06347) supplies its clipped-ratio
constraint. The finite-sum objective is a direct policy-gradient calculation
([Sutton et al., 1999](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)).
[Net2Net](https://arxiv.org/abs/1511.05641) motivates function-preserving growth;
Gus uses zero-output attention/MLP residual branches, not that paper's specific
ReLU deepening construction. These are method-selection criteria, not claims
that any source prescribes this complete recipe. The Cammarata article and its
actual learning advice remain in [[gus-modern-learning-recovery-2026-10-18]].

For legal actions A, let r(a) be one for every exact-best answer and zero
otherwise. The selected loss is:

`L = -sum_a pi(a|x) r(a) + 0.02 KL(pi || parent) + tile_CE + splat_CE`.

The first two terms use exact-curriculum questions; the last two use an
independent rehearsal minibatch. Gradients from both tasks meet in the shared
backbone. The splat head receives its original supervised gradient. The reward
is binary correctness, **not** rational Q itself: a large and small regret gap
receive the same reward labels. Evaluation still measures the actual rational
success-probability gap. Ties give both actions reward one and zero reward-term
gradient. The optimizer uses floating point after exact rational grading.

## One actual input, output and derivative

The archived `worked-example.json` contains this complete decoded prefix:

```text
start bid:30 you trump:3 your hand 3-0 3-3 4-3 5-2 6-2 6-5 6-6 play begins
 you 6-2 left 2-2 partner 5-3 right 6-0
 partner 0-0 right 1-0 you 3-3 left 5-0
 you 5-2 left 5-4 partner 5-1 right 4-2
 left 4-1 partner 4-0 right 6-4 you 3-0
 you 6-6 left 2-0 partner 2-1 right 6-1 you
```

The two legal remaining tiles are **4-3** and **6-5**. Across 60 legal hidden
worlds and the fixed population, their exact team-success values are **41/90**
and **31/48**, a gap of **137/720**. The binary rewards are therefore `[0, 1]`.
Parent legal probabilities `[0.74825, 0.25175]` become
`[0.12547, 0.87453]` at the selected checkpoint. These are initial/final
measurements on one training question, not a held-out result or a single update.

For legal logit z_j, the reward-only derivative is
`dL/dz_j = pi_j (E_pi[r] - r_j)`. At the parent it is
`[+0.18837, -0.18837]`: gradient descent lowers the wrong answer's logit and
raises the right answer's. KL and rehearsal add their own gradients before
clipping and AdamW. This explains the computation without inventing an
internal reasoning trace; the exact reasoning occurs in the external verifier.

## Controlled recipe and selection

All arms start at the original joint-pretrained parent SHA256
`f261e0f257047565cb481f88c5480e39e219d951173710c3dbf858b4b9cd563c`.
The exact curriculum reuses 2,048 train and 512 development physical deals;
its already-read old test is not used for this study's selection or confirmation.
Rehearsal draws 128 original train and 32 original validation deals from each
of seven source caches, retaining all four views: **896 train + 224 retention
validation deals**. This small panel balances sources; it does not estimate the
original pours' frequency-weighted distribution. All deals are disjoint from
the exact curriculum, fresh confirmation and 234 reserved benchmark deals.

Each arm runs 256 rounds, 64 questions/round, two optimizer updates/round,
rehearsal batch32 when enabled, LR1e-5, AdamW betas(0.9,0.95), matrix decay0.1,
gradient norm cap1, KL weight0.02. GRPO uses group8 and ratio clip0.2.
Question, sampled-answer and rehearsal streams have separate saved generators.
Question streams match all arms; rehearsal streams match all rehearsal arms.

Initialization and every32-round checkpoint are eligible. Both original-task
validation losses must stay within +0.02 of their parent values; among feasible
checkpoints, minimize exact development regret, then maximize accuracy, then
prefer the earlier checkpoint. Capacity must improve regret by at least0.5pp
at no more than1.75× measured CPU instructions. Final confirmation requires
at least1pp regret reduction, a paired95% interval excluding zero, and retention.

| Iteration / arm | Parameters | Selected round | Development regret, pp | Train-segment seconds | CPU instructions, billions |
|---|---:|---:|---:|---:|---:|
| 1: GRPO, no rehearsal loss | 185,652 | 32 | 2.650 | 8.25 | 71.87 |
| 1: GRPO + rehearsal | 185,652 | 224 | 2.291 | 13.56 | 91.61 |
| 1: SFT + rehearsal | 185,652 | 224 | 2.625 | 14.43 | 90.67 |
| 2: exact reward + rehearsal | 185,652 | 224 | **2.290** | 10.24 | 90.34 |
| 3: exact reward + rehearsal, extra block | 235,636 | 224 | 2.290 | 13.26 | 114.59 |

Exact and GRPO development regret differ by only0.00085pp; this does not
establish meaningful algorithm superiority. The registered rule selects exact;
its retention losses are also better. The deeper model costs **26.84%** more
CPU instructions and adds **26.92%** parameters with **zero** development-regret
gain. It is rejected. CPU instructions cover host work, not accelerator work;
MPS time and wall time are descriptive, not deterministic thinking budgets.
These are short, single-seed comparisons, not a hyperparameter search or a
capacity ceiling. Selecting early also protects the no-rehearsal control.

## Fixed benchmark and fresh confirmation

All five selected checkpoints ran on [[gus-frozen-benchmark-2026-10-18]], with
full logits and native traces retained. This is a repeated development check,
not the fresh confirmation. The main slice has4,895 unsettled discretionary
rows; its exact subset has471 decisions, including287 ties. Uniform legal play
gets80.47% exact accuracy and5.336pp regret. Paired intervals cluster duplicate
game pairs, not individual tokens.

| Fixed exact subset | Parent | Selected | Paired change, 95% interval |
|---|---:|---:|---:|
| Exact-best accuracy | 81.53% | 86.84% | +5.31pp [1.40,9.28] |
| Exact regret | 4.827pp | 3.312pp | −1.515pp [−2.830,−0.429] |
| Raw tile NLL, main slice | 1.0799 | 1.0194 | −0.0604 [−0.0906,−0.0281] |
| Raw illegal probability, main slice | 2.458% | 2.763% | +0.305pp [0.159,0.465] |

Legal masking prevents impossible plays but does not erase the raw prediction
diagnostic. Relative to G0, the selected exact-regret change is
−1.189pp [−2.686,+0.052]: this comparison does **not** exclude zero.

The new namespace `gus-retention-v2-confirmation-20261018` reserves512 fresh
physical deals before selection. Only the parent and selected model were read
on it, after all selection; no gradients or reselection followed. Uniform-legal
accuracy is67.58%, regret6.046pp, with180 tied questions.

| Fresh 512-deal measure | Parent | Selected |
|---|---:|---:|
| Exact-best accuracy | 73.24% | **81.05%** |
| Exact regret | 4.218pp | **2.379pp** |
| Retention-validation tile NLL | 1.32996 | **1.32913** |
| Retention-validation splat CE | 2.34933 | **2.32235** |
| Retention-validation raw tile accuracy | 50.482% | 50.163% |

Paired regret reduction is **1.839pp [0.936,2.844]**; paired accuracy gain is
**7.813pp [3.711,12.109]**. The registered material-gain and retention tests pass.
Retention concerns the two loss constraints, not improvement of every metric.
No comparison between this fresh cohort and the older cohort is treated as a
paired model gain. All claims remain conditional on the verifier's population.

Two fixed actual-game anchors against Zeb finish in4.39s and3.69s: selected
Gus wins1/4 versus G0's recorded2/4, with a −0.5 marks/game difference from that
reference. Every move and score replays correctly. Four games cannot estimate
strength; neither these results nor exact-model gains promote an incumbent.
The transfer boundary remains early/middle-hand decisions and eventually fresh
paired full games under the same fixed nonlearner policies. The current control
is recorded in [[gus-tiny-gpt]]: expanding decision coverage does not authorize
changing the opponent model. Nello is still pending.

## Recovery, interpretation and publication

Every training launch uses `scripts/cpu-slot --threads 2`, a45s cooperative
budget inside a90s outer cap, at most four segments per arm. All five finish in
one segment; no jobs remain active. SIGINT/SIGTERM, a stop file and time budget
save at a completed round. Full checkpoints retain optimizer, all RNG/samplers,
best-selection history, reference/data/source hashes and W&B identity. Resume
with the identical command plus `--resume`. Inference weights alone cannot do
that. Actual signal interruption followed by bitwise CPU recovery passes for
GRPO, SFT and exact mode with rehearsal and growth; cross-hardware bitwise
identity is not claimed. The relevant gate passes108 Lem tests.

The audit reconstructs every rehearsal tensor/label from the source caches,
checks all split/reserved/confirmation exclusions, recomputes16 fresh exact
certificates, checks equal question/rehearsal streams, hashes all representation
arrays and replays both anchor pairs. Formation and native traces are retained
for the `clean-neuron` skill. No neuron semantics or causal mechanisms are
claimed by this training experiment.

[Complete public CC0 study](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/8dbd48324261a676ee256805ad323d0078c6101d/lem/gus-retention-iterations-2026-10-18): **161 files,
736,807,691 bytes**, all freshly downloaded anonymously and SHA256-verified.
[Verification receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/33ca657507ad35c4e134dfccc1bd01330dd61978/lem/gus-retention-iterations-2026-10-18/verification) pin that audit and the model export.
The archive includes all five arms' full checkpoints, original parent and exact
curriculum, compact rehearsal, fresh confirmation, frozen representations,
anchor games, source-pinned scripts, registration, selection, tests and costs.

[Selected model](https://huggingface.co/jasonyandell/gus/tree/rlvr-rehearsal-v2-20261018), immutable revision
`fa23b0628eb89687af373877d3e8cc0bcccdb635`, uses `head="tile"` and full source checkpoint SHA256
`7fc41750e26fb8de55c322185c7821e9bea0a294706f1fc3d3c4711bd51dc334`. All exported files pass fresh anonymous hash checks;
safetensors and the inference checkpoint reproduce source tensors and both
heads' CPU logits exactly. This is a separate research tag; the general-player
Hub main revision is unchanged.

W&B runs: [control](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/b1b7b55f),
[GRPO rehearsal](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/53b11107),
[SFT rehearsal](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/9a77bcd8),
[selected exact rehearsal](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/fe7de6f1),
[extra block](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/581e6c4a).

The `gus-training` skill now carries retention-aware selection, the direct
finite-reward option and criteria for bounded capacity tests. This protocol
stops after the registered three iterations. A broader curriculum remains a
future experiment rather than an unbounded automatic continuation.
